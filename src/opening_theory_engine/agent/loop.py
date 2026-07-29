"""Explain, verify, and regenerate once if the facts do not hold up.

Two components with one clear job each: the explainer proposes text, the
verifier checks every claim against the engine. If a response fails
verification the loop resamples at a different temperature and, if
nothing passes, returns the least-wrong attempt with its full
verification report attached, so a caller always sees exactly what was
checked and what failed.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from opening_theory_engine.agent.explainer import ExplainContext, Explainer
from opening_theory_engine.alignment.claims import ClaimParser, ParsedExplanation
from opening_theory_engine.alignment.verify import VerificationReport, Verifier
from opening_theory_engine.data.motifs import MotifFlags, detect_motifs
from opening_theory_engine.engine.analysis import Analyser
from opening_theory_engine.engine.facts import MoveFacts


def parse_move(board: chess.Board, text: str) -> chess.Move:
    """Accept a move in SAN or UCI and insist it is legal here."""
    cleaned = text.strip()
    try:
        return board.parse_san(cleaned)
    except chess.IllegalMoveError as exc:
        raise ValueError(f"'{text}' is not legal in this position") from exc
    except chess.AmbiguousMoveError as exc:
        raise ValueError(f"'{text}' is ambiguous in this position") from exc
    except chess.InvalidMoveError:
        pass  # not SAN at all; try UCI next
    try:
        move = chess.Move.from_uci(cleaned.lower())
    except chess.InvalidMoveError as exc:
        raise ValueError(f"'{text}' is not a move in SAN or UCI form") from exc
    if move not in board.legal_moves:
        raise ValueError(f"'{text}' is not legal in this position")
    return move


@dataclass(frozen=True)
class ExplanationOutcome:
    """What the loop hands back: the text plus the evidence trail."""

    explanation: str
    parsed: ParsedExplanation
    report: VerificationReport
    facts: MoveFacts
    motifs: MotifFlags
    attempts: int


class ExplainVerifyLoop:
    """Wires an explainer and a verifier together with bounded retries."""

    def __init__(
        self,
        explainer: Explainer,
        analyser: Analyser,
        verifier: Verifier | None = None,
        max_attempts: int = 2,
        temperatures: tuple[float, ...] = (0.3, 0.0),
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._explainer = explainer
        self._analyser = analyser
        self._verifier = verifier or Verifier()
        self._parser = ClaimParser()
        self._max_attempts = max_attempts
        self._temperatures = temperatures

    def run(
        self,
        fen: str,
        move: str,
        opening: str = "Unknown opening",
        eco: str = "?",
    ) -> ExplanationOutcome:
        board = chess.Board(fen)
        parsed_move = parse_move(board, move)
        context = ExplainContext(
            fen_before=board.fen(),
            move_san=board.san(parsed_move),
            move_uci=parsed_move.uci(),
            turn_white=board.turn == chess.WHITE,
            opening=opening,
            eco=eco,
        )
        facts = self._analyser.evaluate_move(board, parsed_move)
        motifs = detect_motifs(board, parsed_move)

        best: ExplanationOutcome | None = None
        for attempt in range(1, self._max_attempts + 1):
            temperature = self._temperatures[min(attempt - 1, len(self._temperatures) - 1)]
            text = self._explainer.explain(context, temperature=temperature)
            parsed = self._parser.parse(text)
            report = self._verifier.verify(parsed, facts, motifs)
            outcome = ExplanationOutcome(
                explanation=text,
                parsed=parsed,
                report=report,
                facts=facts,
                motifs=motifs,
                attempts=attempt,
            )
            if report.passed():
                return outcome
            if best is None or _less_wrong(report, best.report):
                best = outcome
        assert best is not None  # max_attempts >= 1 guarantees at least one outcome
        return best


def _less_wrong(candidate: VerificationReport, incumbent: VerificationReport) -> bool:
    """Order failed attempts: fewer wrong claims first, then better accuracy."""
    if candidate.n_incorrect != incumbent.n_incorrect:
        return candidate.n_incorrect < incumbent.n_incorrect
    return (candidate.factual_accuracy or 0.0) > (incumbent.factual_accuracy or 0.0)
