"""Material balance, static exchange and settled swings."""

import chess

from opening_theory_engine.engine.material import (
    material_balance_cp,
    material_swing_cp,
    static_exchange_cp,
)


def test_startpos_is_balanced():
    assert material_balance_cp(chess.Board()) == 0


def test_missing_pawn_shows_up():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert material_balance_cp(board) == -100


def test_see_zero_for_non_capture():
    board = chess.Board()
    assert static_exchange_cp(board, chess.Move.from_uci("e2e4")) == 0


def test_see_free_pawn():
    board = chess.Board("k7/8/8/3p4/4P3/8/8/K7 w - - 0 1")
    assert static_exchange_cp(board, chess.Move.from_uci("e4d5")) == 100


def test_see_defended_pawn_is_even():
    board = chess.Board("k7/8/4p3/3p4/4P3/8/8/K7 w - - 0 1")
    assert static_exchange_cp(board, chess.Move.from_uci("e4d5")) == 0


def test_see_queen_grabs_defended_pawn_loses_queen():
    board = chess.Board("k7/8/4p3/3p4/8/8/3Q4/K7 w - - 0 1")
    assert static_exchange_cp(board, chess.Move.from_uci("d2d5")) == -800


def test_material_swing_recapture_cancels_out():
    board = chess.Board("k7/8/4p3/3p4/4P3/8/8/K7 w - - 0 1")
    move = chess.Move.from_uci("e4d5")
    assert material_swing_cp(board, move, ("e6d5",)) == 0


def test_material_swing_free_pawn_stays_won():
    board = chess.Board("k7/8/8/3p4/4P3/8/8/K7 w - - 0 1")
    move = chess.Move.from_uci("e4d5")
    assert material_swing_cp(board, move, ("a8b8",)) == 100
