"""A deterministic engine stand-in for tests and CI.

Scores are a material count plus a small tempo bonus, and lines follow a
greedy capture-first rule. It is obviously not chess-strong; it exists so
the data pipeline, verifier and API can run instantly with no binary and
give byte-identical results everywhere.
"""

from __future__ import annotations

import chess

from opening_theory_engine.engine.facts import EngineLine, MoveFacts
from opening_theory_engine.engine.material import (
    material_balance_cp,
    material_swing_cp,
    static_exchange_cp,
)

_TEMPO_BONUS_CP = 8


class StubAnalyser:
    """Drop-in :class:`~opening_theory_engine.engine.analysis.Analyser` fake."""

    def __init__(self, multipv: int = 2, lookahead_plies: int = 3) -> None:
        self._multipv = multipv
        self._lookahead = lookahead_plies

    @property
    def name(self) -> str:
        return "stub-analyser-1"

    def top_lines(self, board: chess.Board) -> list[EngineLine]:
        scored = self._scored_candidates(board)
        lines = []
        for rank, (move, cp_white, continuation) in enumerate(scored[: self._multipv], start=1):
            moves = (move, *continuation)
            lines.append(
                EngineLine(
                    rank=rank,
                    moves_uci=tuple(item.uci() for item in moves),
                    moves_san=_san_line(board, moves),
                    cp_white=cp_white,
                )
            )
        if not lines:
            raise ValueError(f"no legal moves in {board.fen()}")
        return lines

    def evaluate_move(self, board: chess.Board, move: chess.Move) -> MoveFacts:
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {move.uci()} in {board.fen()}")
        best = self.top_lines(board)[0]
        move_san = board.san(move)

        after = board.copy(stack=False)
        after.push(move)
        continuation = self._greedy_continuation(after)
        cp_after_white = self._score_after_line(after, continuation)

        mover_is_white = board.turn == chess.WHITE
        mover_best = best.cp_white if mover_is_white else -best.cp_white
        mover_after = cp_after_white if mover_is_white else -cp_after_white

        reply_pv_uci = tuple(item.uci() for item in continuation)
        return MoveFacts(
            fen_before=board.fen(),
            move_uci=move.uci(),
            move_san=move_san,
            turn_white=mover_is_white,
            cp_best_white=best.cp_white,
            cp_after_white=cp_after_white,
            cp_loss=max(0, mover_best - mover_after),
            is_best=move.uci() == best.moves_uci[0],
            best_move_uci=best.moves_uci[0],
            best_move_san=best.moves_san[0],
            pv_uci=best.moves_uci,
            pv_san=best.moves_san,
            reply_pv_uci=reply_pv_uci,
            reply_pv_san=_san_line(after, continuation),
            material_delta_cp=material_swing_cp(board, move, reply_pv_uci),
        )

    def close(self) -> None:  # matches the Analyser protocol; nothing to release
        return

    def _scored_candidates(
        self, board: chess.Board
    ) -> list[tuple[chess.Move, int, tuple[chess.Move, ...]]]:
        candidates = sorted(board.legal_moves, key=lambda move: self._move_order_key(board, move))
        scored = []
        for move in candidates[:8]:
            after = board.copy(stack=False)
            after.push(move)
            continuation = self._greedy_continuation(after)
            cp_white = self._score_after_line(after, continuation)
            scored.append((move, cp_white, continuation))
        mover_sign = 1 if board.turn == chess.WHITE else -1
        scored.sort(key=lambda item: (-mover_sign * item[1], item[0].uci()))
        return scored

    def _move_order_key(self, board: chess.Board, move: chess.Move) -> tuple[int, int, str]:
        exchange = static_exchange_cp(board, move)
        gives_check = board.gives_check(move)
        return (-exchange, 0 if gives_check else 1, move.uci())

    def _greedy_continuation(self, board: chess.Board) -> tuple[chess.Move, ...]:
        working = board.copy(stack=False)
        line = []
        for _ in range(self._lookahead):
            if working.is_game_over():
                break
            move = min(
                working.legal_moves,
                key=lambda candidate: self._move_order_key(working, candidate),
            )
            line.append(move)
            working.push(move)
        return tuple(line)

    def _score_after_line(self, board: chess.Board, continuation: tuple[chess.Move, ...]) -> int:
        working = board.copy(stack=False)
        for move in continuation:
            working.push(move)
        if working.is_checkmate():
            return -2000 if working.turn == chess.WHITE else 2000
        tempo = _TEMPO_BONUS_CP if working.turn == chess.WHITE else -_TEMPO_BONUS_CP
        return material_balance_cp(working) + tempo


def _san_line(board: chess.Board, moves: tuple[chess.Move, ...]) -> tuple[str, ...]:
    sans = []
    working = board.copy(stack=False)
    for move in moves:
        if move not in working.legal_moves:
            break
        sans.append(working.san(move))
        working.push(move)
    return tuple(sans)
