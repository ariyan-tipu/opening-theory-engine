"""Band edges and phrase parsing."""

from opening_theory_engine.alignment.bands import (
    EvalBand,
    VerdictBand,
    eval_band_from_cp_white,
    parse_eval,
    parse_verdict,
    verdict_from_cp_loss,
)


def test_verdict_band_edges():
    assert verdict_from_cp_loss(0, is_best=True) == VerdictBand.BEST
    assert verdict_from_cp_loss(10, is_best=False) == VerdictBand.BEST
    assert verdict_from_cp_loss(11, is_best=False) == VerdictBand.GOOD
    assert verdict_from_cp_loss(50, is_best=False) == VerdictBand.GOOD
    assert verdict_from_cp_loss(51, is_best=False) == VerdictBand.INACCURACY
    assert verdict_from_cp_loss(100, is_best=False) == VerdictBand.INACCURACY
    assert verdict_from_cp_loss(101, is_best=False) == VerdictBand.MISTAKE
    assert verdict_from_cp_loss(300, is_best=False) == VerdictBand.MISTAKE
    assert verdict_from_cp_loss(301, is_best=False) == VerdictBand.BLUNDER


def test_is_best_overrides_loss():
    assert verdict_from_cp_loss(500, is_best=True) == VerdictBand.BEST


def test_eval_band_edges():
    assert eval_band_from_cp_white(400) == EvalBand.WHITE_WINNING
    assert eval_band_from_cp_white(399) == EvalBand.WHITE_CLEARLY_BETTER
    assert eval_band_from_cp_white(150) == EvalBand.WHITE_CLEARLY_BETTER
    assert eval_band_from_cp_white(149) == EvalBand.WHITE_SLIGHTLY_BETTER
    assert eval_band_from_cp_white(50) == EvalBand.WHITE_SLIGHTLY_BETTER
    assert eval_band_from_cp_white(49) == EvalBand.EQUAL
    assert eval_band_from_cp_white(-49) == EvalBand.EQUAL
    assert eval_band_from_cp_white(-50) == EvalBand.BLACK_SLIGHTLY_BETTER
    assert eval_band_from_cp_white(-150) == EvalBand.BLACK_CLEARLY_BETTER
    assert eval_band_from_cp_white(-400) == EvalBand.BLACK_WINNING
    assert eval_band_from_cp_white(10_000) == EvalBand.WHITE_WINNING


def test_parse_verdict_accepts_markdown_and_synonyms():
    assert parse_verdict("**Good move.**") == VerdictBand.GOOD
    assert parse_verdict("This is the best move") == VerdictBand.BEST
    assert parse_verdict("a blunder!") == VerdictBand.BLUNDER
    assert parse_verdict("an inaccuracy") == VerdictBand.INACCURACY
    assert parse_verdict("no idea what this is") is None


def test_parse_eval_accepts_synonyms():
    assert parse_eval("White is slightly better") == EvalBand.WHITE_SLIGHTLY_BETTER
    assert parse_eval("the position is roughly equal") == EvalBand.EQUAL
    assert parse_eval("Black has a clear advantage") == EvalBand.BLACK_CLEARLY_BETTER
    assert parse_eval("banana") is None
