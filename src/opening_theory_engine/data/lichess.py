"""Streamed access to the Lichess open database.

The monthly exports are zstandard-compressed PGN, sometimes tens of
gigabytes once unpacked. Nothing here ever decompresses a file to disk or
holds more than one game in memory: games stream straight out of the
compressed file and a hard cap bounds how many are scanned.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import chess.pgn
import httpx
import zstandard
from tqdm import tqdm

from opening_theory_engine.config import DataConfig
from opening_theory_engine.utils import ensure_dir, get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class GameFilter:
    """Header-level quality gate for one game."""

    min_elo: int
    max_elo: int
    min_base_time_s: int

    @classmethod
    def from_config(cls, config: DataConfig) -> GameFilter:
        return cls(
            min_elo=config.min_elo,
            max_elo=config.max_elo,
            min_base_time_s=config.min_base_time_s,
        )

    def passes(self, headers: chess.pgn.Headers) -> bool:
        if headers.get("Variant", "Standard") != "Standard":
            return False
        eco = headers.get("ECO", "?")
        opening = headers.get("Opening", "?")
        if eco in ("", "?") or opening in ("", "?"):
            return False
        try:
            white_elo = int(headers["WhiteElo"])
            black_elo = int(headers["BlackElo"])
        except (KeyError, ValueError):
            return False
        for elo in (white_elo, black_elo):
            if not self.min_elo <= elo <= self.max_elo:
                return False
        return self._base_time_seconds(headers.get("TimeControl", "")) >= self.min_base_time_s

    @staticmethod
    def _base_time_seconds(time_control: str) -> int:
        base = time_control.split("+", maxsplit=1)[0]
        return int(base) if base.isdigit() else 0


def download_database(url: str, raw_dir: Path, quiet: bool = False) -> Path:
    """Fetch one monthly export if it is not already present.

    ``url`` may also be a local file path (plain ``.pgn`` or ``.pgn.zst``),
    which the tests and offline runs use to skip the network entirely.
    """
    if not url.lower().startswith(("http://", "https://")):
        local = Path(url)
        if local.is_file():
            _LOG.info("using local PGN file %s", local)
            return local
        raise FileNotFoundError(f"local PGN path does not exist: {url}")
    file_name = Path(urlparse(url).path).name
    destination = ensure_dir(raw_dir) / file_name
    if destination.is_file() and destination.stat().st_size > 0:
        _LOG.info("using cached %s", destination)
        return destination

    _LOG.info("downloading %s", url)
    partial = destination.with_suffix(destination.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        progress = tqdm(total=total, unit="B", unit_scale=True, desc=file_name, disable=quiet)
        with open(partial, "wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
                progress.update(len(chunk))
        progress.close()
    partial.replace(destination)
    return destination


@contextmanager
def _open_pgn_text(path: Path):
    """Open a PGN file as text, transparently handling zstandard compression."""
    if path.suffix == ".zst":
        with open(path, "rb") as raw:
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
            yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    else:
        with open(path, encoding="utf-8", errors="replace") as handle:
            yield handle


def iter_rated_games(
    pgn_path: Path,
    game_filter: GameFilter,
    max_games_scanned: int,
    quiet: bool = False,
) -> Iterator[chess.pgn.Game]:
    """Yield games that pass the filter, scanning at most ``max_games_scanned``."""
    with _open_pgn_text(pgn_path) as text:
        progress = tqdm(total=max_games_scanned, desc="scanning games", disable=quiet)
        scanned = 0
        kept = 0
        while scanned < max_games_scanned:
            game = chess.pgn.read_game(text)
            if game is None:
                break
            scanned += 1
            progress.update(1)
            if game_filter.passes(game.headers):
                kept += 1
                yield game
        progress.close()
        _LOG.info("scanned %d games, %d passed the filter", scanned, kept)
