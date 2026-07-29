"""Motif detection on hand-built positions."""

import chess
import pytest

from opening_theory_engine.data.motifs import detect_motifs


def test_centre_pawn_push():
    motifs = detect_motifs(chess.Board(), chess.Move.from_uci("e2e4"))
    assert motifs.is_centre_pawn_push
    assert motifs.fights_for_centre
    assert not motifs.develops_minor
    assert not motifs.is_capture


def test_developing_move():
    motifs = detect_motifs(chess.Board(), chess.Move.from_uci("g1f3"))
    assert motifs.develops_minor
    assert motifs.development_delta == 1


def test_castling():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    motifs = detect_motifs(board, chess.Move.from_uci("e1g1"))
    assert motifs.is_castle


def test_gives_check():
    board = chess.Board("k7/8/8/8/8/8/1Q6/K7 w - - 0 1")
    motifs = detect_motifs(board, chess.Move.from_uci("b2b7"))
    assert motifs.gives_check


def test_knight_fork_of_king_and_rook():
    board = chess.Board("2r1k3/8/8/1N6/8/8/8/4K3 w - - 0 1")
    motifs = detect_motifs(board, chess.Move.from_uci("b5d6"))
    assert motifs.is_fork
    assert motifs.gives_check


def test_rook_move_creates_absolute_pin():
    board = chess.Board("4k3/8/8/4n3/8/8/8/R5K1 w - - 0 1")
    motifs = detect_motifs(board, chess.Move.from_uci("a1e1"))
    assert motifs.creates_pin


def test_winning_capture_by_see():
    board = chess.Board("k7/8/8/3p4/4P3/8/8/K7 w - - 0 1")
    motifs = detect_motifs(board, chess.Move.from_uci("e4d5"))
    assert motifs.is_capture
    assert motifs.wins_material_see


def test_illegal_move_rejected():
    with pytest.raises(ValueError, match="illegal move"):
        detect_motifs(chess.Board(), chess.Move.from_uci("e2e5"))


def test_round_trip_dict():
    motifs = detect_motifs(chess.Board(), chess.Move.from_uci("e2e4"))
    from opening_theory_engine.data.motifs import MotifFlags

    assert MotifFlags.from_dict(motifs.to_dict()) == motifs
