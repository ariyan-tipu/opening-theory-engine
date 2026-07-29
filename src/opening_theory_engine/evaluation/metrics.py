"""Metrics for comparing systems.

Factual metrics come from the same verifier used everywhere else, so a
number in the results table means exactly what it means in serving.
Fluency is reported through simple proxies (length, bigram diversity,
n-gram repetition); this project does not pretend to measure prose
quality with a judge model.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from statistics import mean

from opening_theory_engine.alignment.scoring import score_reports
from opening_theory_engine.alignment.verify import VerificationReport


def fluency_stats(texts: Sequence[str]) -> dict[str, float]:
    """Cheap, transparent proxies for how the text reads."""
    token_lists = [text.split() for text in texts]
    lengths = [len(tokens) for tokens in token_lists]

    distinct_2_values = []
    repeated = 0
    for tokens in token_lists:
        bigrams = list(pairwise(tokens))
        if bigrams:
            distinct_2_values.append(len(set(bigrams)) / len(bigrams))
        four_grams = list(zip(tokens, tokens[1:], tokens[2:], tokens[3:], strict=False))
        if four_grams and len(set(four_grams)) < len(four_grams):
            repeated += 1

    return {
        "mean_tokens": round(mean(lengths), 2) if lengths else 0.0,
        "distinct_2": round(mean(distinct_2_values), 4) if distinct_2_values else 0.0,
        "repetition_rate": round(repeated / len(texts), 4) if texts else 0.0,
    }


def aggregate_reports(reports: Sequence[VerificationReport]) -> dict[str, float]:
    """Roll verification reports up into the numbers the results table shows."""
    if not reports:
        return {}
    total = len(reports)

    verdict_distances = [r.verdict_distance for r in reports if r.verdict_distance is not None]
    eval_distances = [r.eval_distance for r in reports if r.eval_distance is not None]
    with_line = [r for r in reports if r.pv_tokens > 0]

    n_correct = sum(r.n_correct for r in reports)
    n_incorrect = sum(r.n_incorrect for r in reports)
    checkable = n_correct + n_incorrect

    return {
        "responses": total,
        "format_rate": round(sum(r.format_ok for r in reports) / total, 4),
        "verdict_exact": _rate(verdict_distances, lambda d: d == 0),
        "verdict_within_one": _rate(verdict_distances, lambda d: d <= 1),
        "eval_exact": _rate(eval_distances, lambda d: d == 0),
        "eval_within_one": _rate(eval_distances, lambda d: d <= 1),
        "line_illegal_rate": (
            round(sum(r.pv_illegal for r in with_line) / len(with_line), 4) if with_line else 0.0
        ),
        "mean_pv_match_plies": (
            round(mean(r.pv_match_plies for r in with_line), 2) if with_line else 0.0
        ),
        "claim_precision": round(n_correct / checkable, 4) if checkable else 0.0,
        "hallucination_rate": round(mean(r.n_incorrect for r in reports), 4),
        "verified_pass_rate": round(sum(r.passed() for r in reports) / total, 4),
        "composite_score_mean": round(float(score_reports(reports).mean()), 4),
    }


def _rate(distances: Sequence[int], predicate) -> float:
    if not distances:
        return 0.0
    return round(sum(1 for value in distances if predicate(value)) / len(distances), 4)
