"""The explain-verify-regenerate loop, driven by the corruptible stub."""

import pytest

from opening_theory_engine.agent.explainer import StubExplainer
from opening_theory_engine.agent.loop import ExplainVerifyLoop, parse_move

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_clean_explainer_passes_first_time(stub_analyser):
    loop = ExplainVerifyLoop(StubExplainer(stub_analyser), stub_analyser)
    outcome = loop.run(START_FEN, "e4")
    assert outcome.report.passed()
    assert outcome.attempts == 1


def test_hallucination_triggers_regeneration(stub_analyser):
    explainer = StubExplainer(stub_analyser, corrupt_first=True)
    loop = ExplainVerifyLoop(explainer, stub_analyser, max_attempts=2)
    outcome = loop.run(START_FEN, "e4")
    assert outcome.attempts == 2
    assert outcome.report.passed()


def test_single_attempt_returns_flagged_result(stub_analyser):
    explainer = StubExplainer(stub_analyser, corrupt_first=True)
    loop = ExplainVerifyLoop(explainer, stub_analyser, max_attempts=1)
    outcome = loop.run(START_FEN, "e4")
    assert outcome.attempts == 1
    assert not outcome.report.passed()
    assert outcome.report.n_incorrect >= 1


def test_uci_and_san_move_forms(stub_analyser):
    loop = ExplainVerifyLoop(StubExplainer(stub_analyser), stub_analyser)
    assert loop.run(START_FEN, "e2e4").facts.move_san == "e4"
    assert loop.run(START_FEN, "e4").facts.move_uci == "e2e4"


def test_invalid_inputs_raise_value_error(stub_analyser):
    loop = ExplainVerifyLoop(StubExplainer(stub_analyser), stub_analyser)
    with pytest.raises(ValueError):
        loop.run("not a fen", "e4")
    with pytest.raises(ValueError):
        loop.run(START_FEN, "e5")  # not legal for White from the start position
    with pytest.raises(ValueError):
        loop.run(START_FEN, "zz9")


def test_parse_move_rejects_illegal_uci():
    import chess

    board = chess.Board()
    with pytest.raises(ValueError, match="not legal"):
        parse_move(board, "e7e5")


def test_max_attempts_must_be_positive(stub_analyser):
    with pytest.raises(ValueError):
        ExplainVerifyLoop(StubExplainer(stub_analyser), stub_analyser, max_attempts=0)
