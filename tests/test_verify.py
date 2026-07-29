"""Verifier behaviour on synthetic ground truth."""

from conftest import make_facts, make_motifs
from opening_theory_engine.alignment.claims import ClaimParser
from opening_theory_engine.alignment.verify import CheckOutcome, Verifier


def _verify(text: str, facts=None, motifs=None, **verifier_kwargs):
    parser = ClaimParser()
    verifier = Verifier(**verifier_kwargs)
    return verifier.verify(parser.parse(text), facts or make_facts(), motifs or make_motifs())


GOOD_TEXT = (
    "Verdict: best move\n"
    "Position after: roughly equal\n"
    "Engine line: 1. e4 e5 2. Nf3\n"
    "Why: A standard choice in this opening."
)


def test_clean_explanation_passes():
    report = _verify(GOOD_TEXT)
    assert report.passed()
    assert report.verdict_distance == 0
    assert report.eval_distance == 0
    assert not report.pv_illegal
    assert report.pv_match_plies == 3


def test_wrong_verdict_fails():
    report = _verify(GOOD_TEXT.replace("best move", "blunder"))
    assert not report.passed()
    assert report.verdict_distance == 4
    assert any(
        check.label == "verdict" and check.outcome == CheckOutcome.INCORRECT
        for check in report.checks
    )


def test_false_material_claim_fails():
    text = GOOD_TEXT.replace("A standard choice in this opening.", "It wins a pawn.")
    report = _verify(text)  # facts say material_delta_cp == 0
    assert not report.passed()
    assert report.n_incorrect == 1


def test_true_material_claim_passes():
    text = GOOD_TEXT.replace("A standard choice in this opening.", "It wins a pawn.")
    report = _verify(text, facts=make_facts(material_delta_cp=100))
    assert report.passed()


def test_true_motif_claim_passes():
    text = GOOD_TEXT.replace(
        "A standard choice in this opening.", "It fights for the central squares."
    )
    report = _verify(text, motifs=make_motifs(is_centre_pawn_push=True))
    assert report.passed()


def test_illegal_line_is_flagged():
    report = _verify(GOOD_TEXT.replace("1. e4 e5 2. Nf3", "1. e4 Ke7"))
    assert report.pv_illegal
    assert not report.passed()


def test_line_must_start_with_played_move():
    report = _verify(GOOD_TEXT.replace("1. e4 e5 2. Nf3", "1. d4 d5"))
    assert not report.passed()


def test_better_was_agreeing_with_engine_is_correct():
    facts = make_facts(is_best=False, cp_loss=80, best_move_uci="d2d4", best_move_san="d4")
    text = (
        "Verdict: inaccuracy\n"
        "Position after: roughly equal\n"
        "Engine line: 1. e4 e5 2. Nf3\n"
        "Why: Better was d4."
    )
    report = _verify(text, facts=facts)
    assert report.passed()


def test_better_was_suggesting_the_played_move_is_wrong():
    facts = make_facts(is_best=False, cp_loss=80, best_move_uci="d2d4", best_move_san="d4")
    report = _verify(
        GOOD_TEXT.replace("A standard choice in this opening.", "Better was e4."), facts=facts
    )
    assert report.n_incorrect >= 1


def test_better_was_other_legal_move_is_unverifiable():
    facts = make_facts(is_best=False, cp_loss=80, best_move_uci="d2d4", best_move_san="d4")
    text = GOOD_TEXT.replace("A standard choice in this opening.", "Better was Nc3.")
    report = _verify(text, facts=facts)
    assert any(check.outcome == CheckOutcome.UNVERIFIABLE for check in report.checks)


def test_tolerance_zero_is_strict():
    text = GOOD_TEXT.replace("roughly equal", "White is slightly better")
    lenient = _verify(text)
    strict = _verify(text, eval_tolerance=0)
    assert lenient.passed()
    assert not strict.passed()
