"""Annotate positions with engine facts and motifs, resume-safely.

Engine analysis is by far the slowest stage, so every result is cached to
an append-only JSONL file keyed by position, move and engine settings. A
run that dies half way (a closed laptop lid, a recycled Colab session)
picks up where it stopped instead of re-analysing anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import chess.engine
import pandas as pd
from tqdm import tqdm

from opening_theory_engine.data.motifs import detect_motifs
from opening_theory_engine.engine.analysis import Analyser
from opening_theory_engine.utils import ensure_dir, get_logger, sha1_hex

_LOG = get_logger(__name__)


class AnnotationCache:
    """Append-only JSONL cache of engine annotations."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: dict[str, dict[str, Any]] = {}
        if path.is_file():
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._records[record["key"]] = record["value"]
            _LOG.info("annotation cache holds %d entries", len(self._records))

    def __len__(self) -> int:
        return len(self._records)

    def get(self, key: str) -> dict[str, Any] | None:
        return self._records.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._records[key] = value
        ensure_dir(self._path.parent)
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


class PositionAnnotator:
    """Runs the engine over a DataFrame of positions and attaches the facts."""

    def __init__(
        self, analyser: Analyser, cache: AnnotationCache, depth: int, multipv: int
    ) -> None:
        self._analyser = analyser
        self._cache = cache
        self._fingerprint = f"{analyser.name}|d{depth}|mpv{multipv}"

    def annotate_frame(self, positions: pd.DataFrame, quiet: bool = False) -> pd.DataFrame:
        """Return the input rows with ``facts`` and ``motifs`` columns added."""
        records: list[dict[str, Any]] = []
        skipped = 0
        rows = positions.to_dict("records")
        for row in tqdm(rows, desc="annotating", disable=quiet):
            annotation = self._annotate_row(row)
            if annotation is None:
                skipped += 1
                continue
            records.append({**row, **annotation})
        if skipped:
            _LOG.warning("skipped %d positions the engine could not analyse", skipped)
        if not records:
            raise RuntimeError("annotation produced no rows")
        return pd.DataFrame(records)

    def _annotate_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        key = sha1_hex(f"{row['fen_before']}|{row['move_uci']}|{self._fingerprint}")
        cached = self._cache.get(key)
        if cached is None:
            board = chess.Board(row["fen_before"])
            move = chess.Move.from_uci(row["move_uci"])
            try:
                facts = self._analyser.evaluate_move(board, move)
                motifs = detect_motifs(board, move)
            except chess.engine.EngineTerminatedError:
                raise
            except (chess.engine.EngineError, ValueError) as exc:
                _LOG.warning("skipping %s %s: %s", row["fen_before"], row["move_uci"], exc)
                return None
            cached = {"facts": facts.to_dict(), "motifs": motifs.to_dict()}
            self._cache.put(key, cached)
        return cached
