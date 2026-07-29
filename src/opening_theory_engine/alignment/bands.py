"""Shared verdict and evaluation bands.

This module is the single source of truth for how centipawn numbers map to
the words the model is allowed to use. The dataset templates render from
these bands, the claim parser reads them back out of model output, and the
verifier and scorer measure distances in band space. Keeping all of that in
one place is what makes the train, verify and align loop consistent.
"""

from __future__ import annotations

import re
from enum import IntEnum
from typing import TypeVar

MATE_SCORE_CP = 10_000


class VerdictBand(IntEnum):
    """How good the played move was, relative to the engine's best."""

    BEST = 0
    GOOD = 1
    INACCURACY = 2
    MISTAKE = 3
    BLUNDER = 4


class EvalBand(IntEnum):
    """Who stands better after the move, from White's point of view."""

    WHITE_WINNING = 0
    WHITE_CLEARLY_BETTER = 1
    WHITE_SLIGHTLY_BETTER = 2
    EQUAL = 3
    BLACK_SLIGHTLY_BETTER = 4
    BLACK_CLEARLY_BETTER = 5
    BLACK_WINNING = 6


VERDICT_PHRASES: dict[VerdictBand, str] = {
    VerdictBand.BEST: "best move",
    VerdictBand.GOOD: "good move",
    VerdictBand.INACCURACY: "inaccuracy",
    VerdictBand.MISTAKE: "mistake",
    VerdictBand.BLUNDER: "blunder",
}

EVAL_PHRASES: dict[EvalBand, str] = {
    EvalBand.WHITE_WINNING: "White is winning",
    EvalBand.WHITE_CLEARLY_BETTER: "White is clearly better",
    EvalBand.WHITE_SLIGHTLY_BETTER: "White is slightly better",
    EvalBand.EQUAL: "roughly equal",
    EvalBand.BLACK_SLIGHTLY_BETTER: "Black is slightly better",
    EvalBand.BLACK_CLEARLY_BETTER: "Black is clearly better",
    EvalBand.BLACK_WINNING: "Black is winning",
}

# Phrases accepted when reading model output back. Longest match wins, so
# "the best move" resolves before "best".
_VERDICT_SYNONYMS: dict[str, VerdictBand] = {
    "the best move": VerdictBand.BEST,
    "best move": VerdictBand.BEST,
    "engine's first choice": VerdictBand.BEST,
    "best": VerdictBand.BEST,
    "a good move": VerdictBand.GOOD,
    "good move": VerdictBand.GOOD,
    "solid move": VerdictBand.GOOD,
    "reasonable move": VerdictBand.GOOD,
    "good": VerdictBand.GOOD,
    "an inaccuracy": VerdictBand.INACCURACY,
    "inaccuracy": VerdictBand.INACCURACY,
    "inaccurate": VerdictBand.INACCURACY,
    "dubious": VerdictBand.INACCURACY,
    "a mistake": VerdictBand.MISTAKE,
    "mistake": VerdictBand.MISTAKE,
    "an error": VerdictBand.MISTAKE,
    "error": VerdictBand.MISTAKE,
    "a blunder": VerdictBand.BLUNDER,
    "blunder": VerdictBand.BLUNDER,
}

_EVAL_SYNONYMS: dict[str, EvalBand] = {
    "white is winning": EvalBand.WHITE_WINNING,
    "white has a winning advantage": EvalBand.WHITE_WINNING,
    "white is completely winning": EvalBand.WHITE_WINNING,
    "white is clearly better": EvalBand.WHITE_CLEARLY_BETTER,
    "white is much better": EvalBand.WHITE_CLEARLY_BETTER,
    "white has a clear advantage": EvalBand.WHITE_CLEARLY_BETTER,
    "white is slightly better": EvalBand.WHITE_SLIGHTLY_BETTER,
    "white has a slight edge": EvalBand.WHITE_SLIGHTLY_BETTER,
    "white has a small advantage": EvalBand.WHITE_SLIGHTLY_BETTER,
    "white stands slightly better": EvalBand.WHITE_SLIGHTLY_BETTER,
    "white is a little better": EvalBand.WHITE_SLIGHTLY_BETTER,
    "roughly equal": EvalBand.EQUAL,
    "about equal": EvalBand.EQUAL,
    "equal": EvalBand.EQUAL,
    "level": EvalBand.EQUAL,
    "balanced": EvalBand.EQUAL,
    "even": EvalBand.EQUAL,
    "black is slightly better": EvalBand.BLACK_SLIGHTLY_BETTER,
    "black has a slight edge": EvalBand.BLACK_SLIGHTLY_BETTER,
    "black has a small advantage": EvalBand.BLACK_SLIGHTLY_BETTER,
    "black stands slightly better": EvalBand.BLACK_SLIGHTLY_BETTER,
    "black is a little better": EvalBand.BLACK_SLIGHTLY_BETTER,
    "black is clearly better": EvalBand.BLACK_CLEARLY_BETTER,
    "black is much better": EvalBand.BLACK_CLEARLY_BETTER,
    "black has a clear advantage": EvalBand.BLACK_CLEARLY_BETTER,
    "black is winning": EvalBand.BLACK_WINNING,
    "black has a winning advantage": EvalBand.BLACK_WINNING,
    "black is completely winning": EvalBand.BLACK_WINNING,
}


def verdict_from_cp_loss(cp_loss: int, is_best: bool) -> VerdictBand:
    """Band the centipawn loss of a move against the engine's best choice."""
    if is_best or cp_loss <= 10:
        return VerdictBand.BEST
    if cp_loss <= 50:
        return VerdictBand.GOOD
    if cp_loss <= 100:
        return VerdictBand.INACCURACY
    if cp_loss <= 300:
        return VerdictBand.MISTAKE
    return VerdictBand.BLUNDER


def eval_band_from_cp_white(cp_white: int) -> EvalBand:
    """Band a White-perspective centipawn score into who stands better."""
    if cp_white >= 400:
        return EvalBand.WHITE_WINNING
    if cp_white >= 150:
        return EvalBand.WHITE_CLEARLY_BETTER
    if cp_white >= 50:
        return EvalBand.WHITE_SLIGHTLY_BETTER
    if cp_white > -50:
        return EvalBand.EQUAL
    if cp_white > -150:
        return EvalBand.BLACK_SLIGHTLY_BETTER
    if cp_white > -400:
        return EvalBand.BLACK_CLEARLY_BETTER
    return EvalBand.BLACK_WINNING


def _normalise(text: str) -> str:
    cleaned = re.sub(r"[*_`]", "", text.strip().lower())
    return re.sub(r"[.!?]+$", "", cleaned).strip()


def parse_verdict(text: str) -> VerdictBand | None:
    """Read a verdict band out of free text, or ``None`` if nothing matches."""
    return _parse_from_synonyms(text, _VERDICT_SYNONYMS)


def parse_eval(text: str) -> EvalBand | None:
    """Read an evaluation band out of free text, or ``None`` if nothing matches."""
    return _parse_from_synonyms(text, _EVAL_SYNONYMS)


_BandT = TypeVar("_BandT", VerdictBand, EvalBand)


def _parse_from_synonyms(text: str, synonyms: dict[str, _BandT]) -> _BandT | None:
    cleaned = _normalise(text)
    if not cleaned:
        return None
    exact = synonyms.get(cleaned)
    if exact is not None:
        return exact
    for phrase in sorted(synonyms, key=len, reverse=True):
        if phrase in cleaned:
            return synonyms[phrase]
    return None
