"""Vectorised candidate scoring with JAX.

Where JAX earns its place here: scoring is pure array arithmetic over
hundreds of candidates at once (several sampled explanations for each of
hundreds of positions), which is exactly the functional, jit plus vmap
shape JAX is built for. The per-candidate rule is written once as scalar
maths (``_score_single``), ``vmap`` lifts it over the batch and ``jit``
compiles the whole thing. To be clear about scale: NumPy would cope fine
with a batch this size, but the JAX version expresses the intent better
and the identical code runs on an accelerator when the batch grows.

String work (parsing claims out of text) stays in ordinary Python where it
belongs; only the numeric features arrive here.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import NamedTuple

# The batch scorer is deliberately small, so JAX runs on the CPU. Pinning the
# CPU backend before JAX initialises keeps this pure numerical code off the GPU,
# where it would otherwise contend with PyTorch and, on hosts like Colab, hit a
# CUDA plugin that a fresh install can leave mismatched with the JAX framework.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from opening_theory_engine.alignment.verify import VerificationReport

_MISSING_VERDICT_DISTANCE = 4.0
_MISSING_EVAL_DISTANCE = 6.0


class CandidateFeatures(NamedTuple):
    """Numeric features of one candidate explanation, batched as arrays."""

    format_ok: jnp.ndarray | np.ndarray
    verdict_distance: jnp.ndarray | np.ndarray
    eval_distance: jnp.ndarray | np.ndarray
    pv_match_ratio: jnp.ndarray | np.ndarray
    pv_illegal: jnp.ndarray | np.ndarray
    n_correct: jnp.ndarray | np.ndarray
    n_incorrect: jnp.ndarray | np.ndarray


class ScoreWeights(NamedTuple):
    """How much each factual dimension moves the score."""

    format_ok: float = 1.0
    verdict: float = 1.5
    eval: float = 1.0
    pv: float = 0.75
    correct: float = 0.15
    incorrect: float = 1.0


def features_from_reports(reports: Sequence[VerificationReport]) -> CandidateFeatures:
    """Flatten verification reports into aligned numpy arrays."""

    def _distance(value: int | None, missing: float) -> float:
        return float(value) if value is not None else missing

    return CandidateFeatures(
        format_ok=np.array([1.0 if r.format_ok else 0.0 for r in reports], dtype=np.float32),
        verdict_distance=np.array(
            [_distance(r.verdict_distance, _MISSING_VERDICT_DISTANCE) for r in reports],
            dtype=np.float32,
        ),
        eval_distance=np.array(
            [_distance(r.eval_distance, _MISSING_EVAL_DISTANCE) for r in reports],
            dtype=np.float32,
        ),
        pv_match_ratio=np.array(
            [r.pv_match_plies / max(r.pv_tokens, 1) for r in reports], dtype=np.float32
        ),
        pv_illegal=np.array([1.0 if r.pv_illegal else 0.0 for r in reports], dtype=np.float32),
        n_correct=np.array([float(r.n_correct) for r in reports], dtype=np.float32),
        n_incorrect=np.array([float(r.n_incorrect) for r in reports], dtype=np.float32),
    )


def _score_single(features: CandidateFeatures, weights: ScoreWeights) -> jnp.ndarray:
    """Score one candidate. Written as scalar maths; vmap batches it."""
    verdict_penalty = features.verdict_distance / _MISSING_VERDICT_DISTANCE
    eval_penalty = features.eval_distance / _MISSING_EVAL_DISTANCE
    return (
        weights.format_ok * features.format_ok
        - weights.verdict * verdict_penalty
        - weights.eval * eval_penalty
        + weights.pv * (features.pv_match_ratio - features.pv_illegal)
        + weights.correct * jnp.minimum(features.n_correct, 4.0)
        - weights.incorrect * features.n_incorrect
    )


_score_batch = jax.jit(jax.vmap(_score_single, in_axes=(0, None)))


def score_reports(
    reports: Sequence[VerificationReport],
    weights: ScoreWeights | None = None,
) -> np.ndarray:
    """Score a batch of verification reports in one compiled pass."""
    if not reports:
        return np.zeros(0, dtype=np.float32)
    features = features_from_reports(reports)
    device_features = CandidateFeatures(*(jnp.asarray(array) for array in features))
    scores = _score_batch(device_features, weights or ScoreWeights())
    return np.asarray(scores)


def reference_scores_numpy(
    reports: Sequence[VerificationReport],
    weights: ScoreWeights | None = None,
) -> np.ndarray:
    """Plain numpy mirror of the JAX scorer, used by tests as an oracle."""
    weights = weights or ScoreWeights()
    features = features_from_reports(reports)
    return (
        weights.format_ok * features.format_ok
        - weights.verdict * (features.verdict_distance / _MISSING_VERDICT_DISTANCE)
        - weights.eval * (features.eval_distance / _MISSING_EVAL_DISTANCE)
        + weights.pv * (features.pv_match_ratio - features.pv_illegal)
        + weights.correct * np.minimum(features.n_correct, 4.0)
        - weights.incorrect * features.n_incorrect
    ).astype(np.float32)
