"""JAX scorer against its numpy oracle, plus ranking sanity."""

import numpy as np

from conftest import make_facts, make_motifs
from opening_theory_engine.alignment.claims import ClaimParser
from opening_theory_engine.alignment.scoring import (
    ScoreWeights,
    reference_scores_numpy,
    score_reports,
)
from opening_theory_engine.alignment.verify import Verifier

CLEAN = (
    "Verdict: best move\n"
    "Position after: roughly equal\n"
    "Engine line: 1. e4 e5 2. Nf3\n"
    "Why: A standard choice."
)
WRONG_CLAIM = CLEAN.replace("A standard choice.", "It wins a pawn.")
WRONG_VERDICT = CLEAN.replace("best move", "blunder")
BROKEN_FORMAT = "Verdict: best move\nWhy: fine."


def _reports():
    parser = ClaimParser()
    verifier = Verifier()
    facts, motifs = make_facts(), make_motifs()
    return [
        verifier.verify(parser.parse(text), facts, motifs)
        for text in (CLEAN, WRONG_CLAIM, WRONG_VERDICT, BROKEN_FORMAT)
    ]


def test_jax_matches_numpy_reference():
    reports = _reports()
    jax_scores = score_reports(reports)
    numpy_scores = reference_scores_numpy(reports)
    np.testing.assert_allclose(jax_scores, numpy_scores, rtol=1e-5, atol=1e-6)


def test_ranking_prefers_clean_explanations():
    scores = score_reports(_reports())
    clean, wrong_claim, wrong_verdict, broken = scores
    assert clean > wrong_claim
    assert clean > wrong_verdict
    assert clean > broken


def test_custom_weights_change_scores():
    reports = _reports()
    default = score_reports(reports)
    harsher = score_reports(reports, ScoreWeights(incorrect=5.0))
    assert harsher[1] < default[1]


def test_empty_batch():
    assert score_reports([]).shape == (0,)
