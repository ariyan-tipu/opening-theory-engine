"""Draw candidate opening positions out of filtered games.

Each game contributes at most a couple of positions from the opening
window, and the pool is deduplicated on (position, move) and rebalanced
across ECO families with pandas so no single opening dominates the
training set.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import chess
import chess.pgn
import pandas as pd

from opening_theory_engine.config import DataConfig
from opening_theory_engine.utils import get_logger, sha1_hex

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class RawPosition:
    """One (position, played move) pair with its game metadata."""

    game_id: str
    eco: str
    opening: str
    white_elo: int
    black_elo: int
    ply: int
    turn_white: bool
    fen_before: str
    move_uci: str
    move_san: str


def positions_from_game(
    game: chess.pgn.Game,
    config: DataConfig,
    rng: random.Random,
) -> list[RawPosition]:
    """Sample up to ``max_positions_per_game`` positions from the opening window."""
    headers = game.headers
    game_id = _game_id(headers)
    try:
        white_elo = int(headers["WhiteElo"])
        black_elo = int(headers["BlackElo"])
    except (KeyError, ValueError):
        return []

    candidates: list[RawPosition] = []
    board = game.board()
    for ply, move in enumerate(game.mainline_moves(), start=1):
        if ply > config.max_ply:
            break
        if ply >= config.min_ply and move in board.legal_moves:
            candidates.append(
                RawPosition(
                    game_id=game_id,
                    eco=headers.get("ECO", "?"),
                    opening=headers.get("Opening", "Unknown opening"),
                    white_elo=white_elo,
                    black_elo=black_elo,
                    ply=ply,
                    turn_white=board.turn == chess.WHITE,
                    fen_before=board.fen(),
                    move_uci=move.uci(),
                    move_san=board.san(move),
                )
            )
        if move not in board.legal_moves:
            break
        board.push(move)

    picked = min(config.max_positions_per_game, len(candidates))
    return rng.sample(candidates, picked) if picked else []


def collect_positions(
    games: Iterable[chess.pgn.Game],
    config: DataConfig,
    seed: int,
) -> pd.DataFrame:
    """Accumulate, deduplicate and ECO-balance positions into a DataFrame."""
    rng = random.Random(seed)
    rows: list[dict] = []
    enough = config.max_unique_positions * 3
    for game in games:
        rows.extend(asdict(position) for position in positions_from_game(game, config, rng))
        if len(rows) >= enough:
            break

    if not rows:
        raise RuntimeError("no positions survived the filters; loosen the data config")

    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["fen_before", "move_uci"], keep="first")
    frame = _balance_eco_families(frame, config.max_unique_positions, seed)
    frame = frame.sort_values(["game_id", "ply"]).reset_index(drop=True)
    frame["position_id"] = [
        sha1_hex(f"{fen}|{uci}")[:16]
        for fen, uci in zip(frame["fen_before"], frame["move_uci"], strict=True)
    ]
    _LOG.info(
        "collected %d unique positions across %d ECO families",
        len(frame),
        frame["eco"].str[0].nunique(),
    )
    return frame


def _balance_eco_families(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Cap the pool, drawing evenly from ECO letters A to E where possible."""
    if len(frame) <= cap:
        return frame
    letters = frame["eco"].str[0]
    families = sorted(letters.unique())
    quota = max(1, cap // len(families))

    taken = []
    for family in families:
        group = frame[letters == family]
        taken.append(group.sample(n=min(quota, len(group)), random_state=seed))
    balanced = pd.concat(taken)

    shortfall = cap - len(balanced)
    if shortfall > 0:
        leftovers = frame.drop(balanced.index)
        balanced = pd.concat(
            [balanced, leftovers.sample(n=min(shortfall, len(leftovers)), random_state=seed)]
        )
    return balanced


def _game_id(headers: chess.pgn.Headers) -> str:
    site = headers.get("Site", "")
    if "/" in site:
        tail = site.rstrip("/").rsplit("/", maxsplit=1)[-1]
        if tail:
            return tail
    return sha1_hex(str(headers))[:12]
