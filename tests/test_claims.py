"""Claim parsing: labelled sections, SAN tokens, lexicon and negation."""

from opening_theory_engine.alignment.bands import EvalBand, VerdictBand
from opening_theory_engine.alignment.claims import ClaimKind, ClaimParser

WELL_FORMED = (
    "Verdict: good move\n"
    "Position after: roughly equal\n"
    "Engine line: 1. e4 e5 2. Nf3\n"
    "Why: It fights for the central squares. It wins a pawn."
)


def test_well_formed_explanation_parses():
    parsed = ClaimParser().parse(WELL_FORMED)
    assert parsed.format_ok
    assert parsed.verdict == VerdictBand.GOOD
    assert parsed.eval_band == EvalBand.EQUAL
    assert parsed.line_tokens == ("e4", "e5", "Nf3")
    kinds = {claim.kind for claim in parsed.claims}
    assert ClaimKind.CENTRE in kinds
    assert ClaimKind.WINS_PAWN in kinds


def test_markdown_noise_is_tolerated():
    text = WELL_FORMED.replace("Verdict:", "**Verdict:**")
    parsed = ClaimParser().parse(text)
    assert parsed.format_ok
    assert parsed.verdict == VerdictBand.GOOD


def test_missing_section_breaks_format():
    text = "Verdict: good move\nPosition after: roughly equal\nEngine line: 1. e4"
    parsed = ClaimParser().parse(text)
    assert not parsed.format_ok


def test_black_move_numbers_are_stripped():
    text = WELL_FORMED.replace("1. e4 e5 2. Nf3", "4... Nf6 5. O-O Be7")
    parsed = ClaimParser().parse(text)
    assert parsed.line_tokens == ("Nf6", "O-O", "Be7")


def test_negated_claim_is_not_extracted():
    text = WELL_FORMED.replace("It wins a pawn.", "It does not win a pawn.")
    parsed = ClaimParser().parse(text)
    kinds = {claim.kind for claim in parsed.claims}
    assert ClaimKind.WINS_PAWN not in kinds


def test_better_was_captures_the_move():
    text = WELL_FORMED.replace("It wins a pawn.", "Better was Nc3.")
    parsed = ClaimParser().parse(text)
    better = [claim for claim in parsed.claims if claim.kind == ClaimKind.BETTER_WAS]
    assert len(better) == 1
    assert better[0].move_san == "Nc3"


def test_engine_prefers_variant_is_caught():
    text = WELL_FORMED.replace("It wins a pawn.", "The engine prefers d4 here.")
    parsed = ClaimParser().parse(text)
    better = [claim for claim in parsed.claims if claim.kind == ClaimKind.BETTER_WAS]
    assert len(better) == 1
    assert better[0].move_san == "d4"


def test_motif_keywords():
    text = WELL_FORMED.replace(
        "It fights for the central squares. It wins a pawn.",
        "Castling brings the king to safety. The move sets up a pin and creates a fork.",
    )
    parsed = ClaimParser().parse(text)
    kinds = {claim.kind for claim in parsed.claims}
    assert {ClaimKind.CASTLES, ClaimKind.PIN, ClaimKind.FORK} <= kinds


def test_each_claim_kind_reported_once():
    text = WELL_FORMED.replace("It wins a pawn.", "It wins a pawn. Later it wins a pawn again.")
    parsed = ClaimParser().parse(text)
    pawn_claims = [claim for claim in parsed.claims if claim.kind == ClaimKind.WINS_PAWN]
    assert len(pawn_claims) == 1
