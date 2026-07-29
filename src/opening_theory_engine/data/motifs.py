"""Deterministic tactical and positional motif detection.

These flags do three jobs: they are features in the Pandas dataset build,
they decide which sentences the explanation templates are allowed to use,
and they are the ground truth the verifier checks motif claims against.
One implementation, three consumers, no room for the story and the check
to drift apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import chess

from opening_theory_engine.engine.material import PIECE_VALUES_CP, static_exchange_cp

CENTRE_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)
_MINOR_PIECES = (chess.KNIGHT, chess.BISHOP)
_FORK_TARGET_MIN_CP = 300


@dataclass(frozen=True)
class MotifFlags:
    """What is verifiably true about one move, beyond raw evaluation."""

    is_capture: bool
    gives_check: bool
    is_castle: bool
    is_promotion: bool
    is_centre_pawn_push: bool
    develops_minor: bool
    wins_material_see: bool
    creates_pin: bool
    is_fork: bool
    centre_attacks_delta: int
    development_delta: int

    @property
    def fights_for_centre(self) -> bool:
        return self.is_centre_pawn_push or self.centre_attacks_delta > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> MotifFlags:
        return cls(**{key: record[key] for key in cls.__dataclass_fields__})


def detect_motifs(board: chess.Board, move: chess.Move) -> MotifFlags:
    """Compute all motif flags for a legal move on the given board."""
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {move.uci()} in {board.fen()}")
    mover = board.turn
    piece = board.piece_at(move.from_square)
    if piece is None:
        raise ValueError(f"no piece on {chess.square_name(move.from_square)}")

    after = board.copy(stack=False)
    after.push(move)

    return MotifFlags(
        is_capture=board.is_capture(move),
        gives_check=board.gives_check(move),
        is_castle=board.is_castling(move),
        is_promotion=move.promotion is not None,
        is_centre_pawn_push=piece.piece_type == chess.PAWN and move.to_square in CENTRE_SQUARES,
        develops_minor=_develops_minor(piece, move, mover),
        wins_material_see=static_exchange_cp(board, move) >= PIECE_VALUES_CP[chess.PAWN],
        creates_pin=_creates_pin(board, after, mover),
        is_fork=_is_fork(after, move.to_square, mover),
        centre_attacks_delta=_centre_attacks(after, mover) - _centre_attacks(board, mover),
        development_delta=_developed_minors(after, mover) - _developed_minors(board, mover),
    )


def _back_rank(colour: chess.Color) -> int:
    return 0 if colour == chess.WHITE else 7


def _develops_minor(piece: chess.Piece, move: chess.Move, mover: chess.Color) -> bool:
    return (
        piece.piece_type in _MINOR_PIECES
        and chess.square_rank(move.from_square) == _back_rank(mover)
        and chess.square_rank(move.to_square) != _back_rank(mover)
    )


def _developed_minors(board: chess.Board, colour: chess.Color) -> int:
    developed = 0
    for piece_type in _MINOR_PIECES:
        for square in board.pieces(piece_type, colour):
            if chess.square_rank(square) != _back_rank(colour):
                developed += 1
    return developed


def _centre_attacks(board: chess.Board, colour: chess.Color) -> int:
    return sum(len(board.attackers(colour, square)) for square in CENTRE_SQUARES)


def _creates_pin(before: chess.Board, after: chess.Board, mover: chess.Color) -> bool:
    enemy = not mover
    for square, piece in after.piece_map().items():
        if piece.color != enemy or piece.piece_type == chess.KING:
            continue
        if not after.is_pinned(enemy, square):
            continue
        was_pinned = before.piece_at(square) == piece and before.is_pinned(enemy, square)
        if not was_pinned:
            return True
    return False


def _is_fork(after: chess.Board, landing_square: chess.Square, mover: chess.Color) -> bool:
    enemy = not mover
    valuable_targets = 0
    for square in after.attacks(landing_square):
        target = after.piece_at(square)
        if target is None or target.color != enemy:
            continue
        if PIECE_VALUES_CP[target.piece_type] >= _FORK_TARGET_MIN_CP or (
            target.piece_type == chess.KING
        ):
            valuable_targets += 1
    return valuable_targets >= 2
