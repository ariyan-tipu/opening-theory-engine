"""Request and response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from opening_theory_engine import __version__


class ExplainRequest(BaseModel):
    """A position, the move that was played and optional opening context.

    Every field is length-bounded. python-chess rejects anything malformed
    anyway, but it only gets to say so after the whole body has been parsed
    into memory, so the cheap check goes first. A FEN runs to about 90
    characters at its longest and a move in SAN or UCI to seven, so these
    limits are generous rather than tight.
    """

    fen: str = Field(
        description="FEN of the position before the move",
        examples=["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
        min_length=1,
        max_length=120,
    )
    move: str = Field(
        description="The played move in SAN or UCI",
        examples=["e4", "e2e4"],
        min_length=1,
        max_length=10,
    )
    opening: str = Field(default="Unknown opening", max_length=120)
    eco: str = Field(default="?", max_length=3)


class ClaimCheckOut(BaseModel):
    label: str
    outcome: str
    expected: str
    observed: str


class VerificationOut(BaseModel):
    passed: bool
    format_ok: bool
    factual_accuracy: float | None
    verdict_distance: int | None
    eval_distance: int | None
    pv_match_plies: int
    pv_illegal: bool
    checks: list[ClaimCheckOut]


class EngineFactsOut(BaseModel):
    best_move_san: str
    cp_best_white: int
    cp_after_white: int
    cp_loss: int
    is_best: bool
    expected_line_san: list[str]
    material_delta_cp: int


class ExplainResponse(BaseModel):
    explanation: str
    attempts: int
    verification: VerificationOut
    engine: EngineFactsOut
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str = __version__
    explainer: str
    engine: str
