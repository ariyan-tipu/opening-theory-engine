"""Immutable records of what the engine actually said about a move.

Everything downstream (templates, verifier, scorer, evaluation) works from
``MoveFacts``. Scores are stored from White's point of view; mate scores are
mapped through ``MATE_SCORE_CP`` so a forced mate lands in the outermost
evaluation band. ``cp_loss`` is from the mover's point of view and never
negative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EngineLine:
    """One multipv line: the moves and the score if both sides follow it."""

    rank: int
    moves_uci: tuple[str, ...]
    moves_san: tuple[str, ...]
    cp_white: int


@dataclass(frozen=True)
class MoveFacts:
    """Ground truth for one (position, played move) pair."""

    fen_before: str
    move_uci: str
    move_san: str
    turn_white: bool
    cp_best_white: int
    cp_after_white: int
    cp_loss: int
    is_best: bool
    best_move_uci: str
    best_move_san: str
    pv_uci: tuple[str, ...]
    pv_san: tuple[str, ...]
    reply_pv_uci: tuple[str, ...]
    reply_pv_san: tuple[str, ...]
    material_delta_cp: int

    @property
    def expected_line_san(self) -> tuple[str, ...]:
        """The continuation an explanation should quote: played move, then replies."""
        return (self.move_san, *self.reply_pv_san)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        for key, value in record.items():
            if isinstance(value, tuple):
                record[key] = list(value)
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> MoveFacts:
        data = dict(record)
        for key in ("pv_uci", "pv_san", "reply_pv_uci", "reply_pv_san"):
            data[key] = tuple(data.get(key) or ())
        return cls(
            fen_before=data["fen_before"],
            move_uci=data["move_uci"],
            move_san=data["move_san"],
            turn_white=bool(data["turn_white"]),
            cp_best_white=int(data["cp_best_white"]),
            cp_after_white=int(data["cp_after_white"]),
            cp_loss=int(data["cp_loss"]),
            is_best=bool(data["is_best"]),
            best_move_uci=data["best_move_uci"],
            best_move_san=data["best_move_san"],
            pv_uci=data["pv_uci"],
            pv_san=data["pv_san"],
            reply_pv_uci=data["reply_pv_uci"],
            reply_pv_san=data["reply_pv_san"],
            material_delta_cp=int(data["material_delta_cp"]),
        )
