"""Tests that need a real Stockfish binary.

Skipped automatically when no binary is available; CI installs one, and
locally ``ote engine fetch`` or STOCKFISH_PATH makes them run.
"""

import chess
import pytest

from opening_theory_engine.alignment.bands import VerdictBand, verdict_from_cp_loss
from opening_theory_engine.config import EngineConfig
from opening_theory_engine.engine.analysis import (
    EngineNotFoundError,
    StockfishAnalyser,
    resolve_engine_path,
)


def _engine_available() -> bool:
    try:
        resolve_engine_path(None)
    except EngineNotFoundError:
        return False
    return True


pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(not _engine_available(), reason="no Stockfish binary available"),
]


@pytest.fixture(scope="module")
def analyser():
    with StockfishAnalyser(EngineConfig(depth=8, multipv=2, threads=1, hash_mb=32)) as engine:
        yield engine


def test_startpos_e4_looks_sane(analyser):
    board = chess.Board()
    facts = analyser.evaluate_move(board, chess.Move.from_uci("e2e4"))
    assert facts.cp_loss >= 0
    assert abs(facts.cp_after_white) < 150, "start position should be near equal"
    assert facts.pv_san, "expected a principal variation"
    assert facts.move_san == "e4"
    # The reply line must replay legally from the position after the move.
    after = chess.Board()
    after.push_uci("e2e4")
    for uci in facts.reply_pv_uci:
        move = chess.Move.from_uci(uci)
        assert move in after.legal_moves
        after.push(move)


def test_walking_into_mate_is_a_blunder(analyser):
    # 1. e4 e5 2. Bc4 Nc6 3. Qh5 and now Nf6 allows Qxf7 mate.
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3")
    facts = analyser.evaluate_move(board, chess.Move.from_uci("g8f6"))
    assert facts.cp_loss > 300
    assert verdict_from_cp_loss(facts.cp_loss, facts.is_best) == VerdictBand.BLUNDER
    assert facts.cp_after_white > 400, "mate for White should band as winning"


def test_top_lines_are_ranked(analyser):
    lines = analyser.top_lines(chess.Board())
    assert len(lines) == 2
    assert lines[0].rank == 1
    assert lines[0].moves_san
