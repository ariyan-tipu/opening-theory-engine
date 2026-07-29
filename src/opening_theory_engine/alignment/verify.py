"""Check parsed claims against engine ground truth.

This verifier is deterministic on purpose. Given the same explanation and
the same engine facts it always returns the same report, which makes it
usable in three places with identical behaviour: building DPO preference
pairs, the held-out evaluation, and the live serving loop that flags
hallucinated claims before a response goes out.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import chess

from opening_theory_engine.alignment.bands import (
    EVAL_PHRASES,
    VERDICT_PHRASES,
    eval_band_from_cp_white,
    verdict_from_cp_loss,
)
from opening_theory_engine.alignment.claims import ClaimKind, ParsedExplanation, TextClaim
from opening_theory_engine.data.motifs import MotifFlags
from opening_theory_engine.engine.facts import MoveFacts


class CheckOutcome(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class ClaimCheck:
    """One claim, what we expected, what the model said, and the outcome."""

    label: str
    outcome: CheckOutcome
    expected: str
    observed: str

    def to_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "outcome": self.outcome.value,
            "expected": self.expected,
            "observed": self.observed,
        }


@dataclass(frozen=True)
class VerificationReport:
    """The verifier's full result for one explanation."""

    format_ok: bool
    verdict_distance: int | None
    eval_distance: int | None
    pv_match_plies: int
    pv_tokens: int
    pv_illegal: bool
    checks: tuple[ClaimCheck, ...]

    @property
    def n_correct(self) -> int:
        return sum(1 for check in self.checks if check.outcome == CheckOutcome.CORRECT)

    @property
    def n_incorrect(self) -> int:
        return sum(1 for check in self.checks if check.outcome == CheckOutcome.INCORRECT)

    @property
    def n_unverifiable(self) -> int:
        return sum(1 for check in self.checks if check.outcome == CheckOutcome.UNVERIFIABLE)

    @property
    def factual_accuracy(self) -> float | None:
        checkable = self.n_correct + self.n_incorrect
        return self.n_correct / checkable if checkable else None

    def passed(self) -> bool:
        return self.format_ok and self.n_incorrect == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_ok": self.format_ok,
            "verdict_distance": self.verdict_distance,
            "eval_distance": self.eval_distance,
            "pv_match_plies": self.pv_match_plies,
            "pv_tokens": self.pv_tokens,
            "pv_illegal": self.pv_illegal,
            "passed": self.passed(),
            "factual_accuracy": self.factual_accuracy,
            "checks": [check.to_dict() for check in self.checks],
        }


# Material windows in centipawns, from the mover's point of view. "Wins a
# pawn" is only counted correct when the settled material swing looks like
# one pawn, not a whole piece and not a mirage that gets recaptured.
_MATERIAL_WINDOWS: dict[ClaimKind, tuple[int, int]] = {
    ClaimKind.WINS_PAWN: (60, 180),
    ClaimKind.WINS_EXCHANGE: (130, 280),
    ClaimKind.WINS_PIECE: (240, 100_000),
    ClaimKind.WINS_MATERIAL: (60, 100_000),
    ClaimKind.LOSES_MATERIAL: (-100_000, -60),
}


class Verifier:
    """Deterministic claim checking with configurable band tolerances."""

    def __init__(self, verdict_tolerance: int = 1, eval_tolerance: int = 1) -> None:
        self._verdict_tolerance = verdict_tolerance
        self._eval_tolerance = eval_tolerance

    def verify(
        self,
        parsed: ParsedExplanation,
        facts: MoveFacts,
        motifs: MotifFlags,
    ) -> VerificationReport:
        checks: list[ClaimCheck] = []

        expected_verdict = verdict_from_cp_loss(facts.cp_loss, facts.is_best)
        verdict_distance: int | None = None
        if parsed.verdict is not None:
            verdict_distance = abs(int(parsed.verdict) - int(expected_verdict))
            checks.append(
                ClaimCheck(
                    label="verdict",
                    outcome=self._banded(verdict_distance, self._verdict_tolerance),
                    expected=VERDICT_PHRASES[expected_verdict],
                    observed=VERDICT_PHRASES[parsed.verdict],
                )
            )

        expected_eval = eval_band_from_cp_white(facts.cp_after_white)
        eval_distance: int | None = None
        if parsed.eval_band is not None:
            eval_distance = abs(int(parsed.eval_band) - int(expected_eval))
            checks.append(
                ClaimCheck(
                    label="position after",
                    outcome=self._banded(eval_distance, self._eval_tolerance),
                    expected=EVAL_PHRASES[expected_eval],
                    observed=EVAL_PHRASES[parsed.eval_band],
                )
            )

        pv_match, pv_illegal = self._check_line(parsed, facts, checks)

        for claim in parsed.claims:
            checks.append(self._check_text_claim(claim, facts, motifs))

        return VerificationReport(
            format_ok=parsed.format_ok,
            verdict_distance=verdict_distance,
            eval_distance=eval_distance,
            pv_match_plies=pv_match,
            pv_tokens=len(parsed.line_tokens),
            pv_illegal=pv_illegal,
            checks=tuple(checks),
        )

    @staticmethod
    def _banded(distance: int, tolerance: int) -> CheckOutcome:
        return CheckOutcome.CORRECT if distance <= tolerance else CheckOutcome.INCORRECT

    def _check_line(
        self,
        parsed: ParsedExplanation,
        facts: MoveFacts,
        checks: list[ClaimCheck],
    ) -> tuple[int, bool]:
        """Legality is a hard claim; agreement with the engine line is graded.

        A legal line that diverges from the engine's continuation is not a
        lie, so divergence only affects the score through ``pv_match_plies``.
        An illegal move or a line that starts with a different move than the
        one being explained is factually wrong.
        """
        if not parsed.line_tokens:
            return 0, False

        board = chess.Board(facts.fen_before)
        legal_plies = 0
        illegal = False
        for token in parsed.line_tokens:
            try:
                board.push(board.parse_san(token))
                legal_plies += 1
            except ValueError:
                illegal = True
                break

        expected = facts.expected_line_san
        match_plies = 0
        for token, expected_san in zip(parsed.line_tokens, expected, strict=False):
            if token.rstrip("+#") != expected_san.rstrip("+#"):
                break
            match_plies += 1

        starts_with_move = bool(parsed.line_tokens) and (
            parsed.line_tokens[0].rstrip("+#") == facts.move_san.rstrip("+#")
        )
        if illegal or not starts_with_move:
            outcome = CheckOutcome.INCORRECT
            observed = (
                "contains an illegal move" if illegal else "does not start with the played move"
            )
        else:
            outcome = CheckOutcome.CORRECT
            observed = f"legal line, matches the engine for {match_plies} plies"
        checks.append(
            ClaimCheck(
                label="engine line",
                outcome=outcome,
                expected=" ".join(expected[:6]),
                observed=observed,
            )
        )
        return match_plies, illegal

    def _check_text_claim(
        self,
        claim: TextClaim,
        facts: MoveFacts,
        motifs: MotifFlags,
    ) -> ClaimCheck:
        if claim.kind in _MATERIAL_WINDOWS:
            low, high = _MATERIAL_WINDOWS[claim.kind]
            holds = low <= facts.material_delta_cp <= high
            return ClaimCheck(
                label=claim.kind.value,
                outcome=CheckOutcome.CORRECT if holds else CheckOutcome.INCORRECT,
                expected=f"material swing {facts.material_delta_cp} cp",
                observed=claim.raw,
            )
        if claim.kind == ClaimKind.BETTER_WAS:
            return self._check_better_was(claim, facts)

        flag_by_kind = {
            ClaimKind.GIVES_CHECK: motifs.gives_check,
            ClaimKind.CASTLES: motifs.is_castle,
            ClaimKind.DEVELOPS: motifs.develops_minor,
            ClaimKind.CENTRE: motifs.fights_for_centre,
            ClaimKind.FORK: motifs.is_fork,
            ClaimKind.PIN: motifs.creates_pin,
        }
        holds = flag_by_kind.get(claim.kind, False)
        return ClaimCheck(
            label=claim.kind.value,
            outcome=CheckOutcome.CORRECT if holds else CheckOutcome.INCORRECT,
            expected=f"{claim.kind.value} is {'true' if holds else 'false'} for this move",
            observed=claim.raw,
        )

    @staticmethod
    def _check_better_was(claim: TextClaim, facts: MoveFacts) -> ClaimCheck:
        suggested = (claim.move_san or "").rstrip("+#")
        board = chess.Board(facts.fen_before)
        try:
            move = board.parse_san(suggested)
        except ValueError:
            return ClaimCheck(
                label=claim.kind.value,
                outcome=CheckOutcome.INCORRECT,
                expected=f"a legal alternative such as {facts.best_move_san}",
                observed=f"{claim.raw} (not a legal move here)",
            )
        if facts.is_best or move.uci() == facts.move_uci:
            return ClaimCheck(
                label=claim.kind.value,
                outcome=CheckOutcome.INCORRECT,
                expected="no better move exists" if facts.is_best else "a different move",
                observed=claim.raw,
            )
        if move.uci() == facts.best_move_uci:
            return ClaimCheck(
                label=claim.kind.value,
                outcome=CheckOutcome.CORRECT,
                expected=facts.best_move_san,
                observed=claim.raw,
            )
        return ClaimCheck(
            label=claim.kind.value,
            outcome=CheckOutcome.UNVERIFIABLE,
            expected=f"engine best is {facts.best_move_san}",
            observed=f"{claim.raw} (legal, but not the engine's first choice)",
        )
