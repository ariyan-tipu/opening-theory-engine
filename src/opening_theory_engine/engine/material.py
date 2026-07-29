"""Material arithmetic on boards: balances, swings and a simple exchange check."""

from __future__ import annotations

import chess

PIECE_VALUES_CP: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 310,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def material_balance_cp(board: chess.Board) -> int:
    """Material count in centipawns from White's point of view."""
    balance = 0
    for piece_type, value in PIECE_VALUES_CP.items():
        balance += value * len(board.pieces(piece_type, chess.WHITE))
        balance -= value * len(board.pieces(piece_type, chess.BLACK))
    return balance


def material_swing_cp(
    board_before: chess.Board,
    move: chess.Move,
    reply_pv_uci: tuple[str, ...],
    max_plies: int = 4,
) -> int:
    """Material change for the mover once the engine's expected replies settle.

    Plays the move and then up to ``max_plies`` plies of the engine's
    continuation, then compares material with the starting position. This is
    what backs claims like "wins a pawn": a capture that gets recaptured
    immediately should not count as winning anything.
    """
    base = material_balance_cp(board_before)
    board = board_before.copy(stack=False)
    board.push(move)
    for uci in reply_pv_uci[:max_plies]:
        reply = chess.Move.from_uci(uci)
        if reply not in board.legal_moves:
            break
        board.push(reply)
    delta_white = material_balance_cp(board) - base
    return delta_white if board_before.turn == chess.WHITE else -delta_white


def _captured_value_cp(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return PIECE_VALUES_CP[chess.PAWN]
    captured = board.piece_type_at(move.to_square)
    return PIECE_VALUES_CP[captured] if captured is not None else 0


def static_exchange_cp(board: chess.Board, move: chess.Move) -> int:
    """Approximate static exchange value of a capture, in centipawns.

    Plays out the cheapest legal recapture on the target square until no
    recapture remains, then folds the gain list back so that either side may
    stop capturing when continuing loses material. Using real legal moves
    means pins are respected without any extra machinery. Non-captures
    score zero.
    """
    if not board.is_capture(move):
        return 0
    square = move.to_square
    gains = [_captured_value_cp(board, move)]
    working = board.copy(stack=False)
    working.push(move)
    while True:
        recaptures = [
            candidate
            for candidate in working.legal_moves
            if candidate.to_square == square and working.is_capture(candidate)
        ]
        if not recaptures:
            break
        cheapest = min(
            recaptures,
            key=lambda candidate: (
                PIECE_VALUES_CP[working.piece_type_at(candidate.from_square) or chess.KING],
                candidate.uci(),
            ),
        )
        gains.append(_captured_value_cp(working, cheapest))
        working.push(cheapest)
    declined = 0
    for value in reversed(gains[1:]):
        declined = max(0, value - declined)
    return gains[0] - declined
