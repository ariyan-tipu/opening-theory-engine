"""The FastAPI application.

Backends are chosen by environment variables (prefix ``OTE_``), so the
same container image runs as a stub for CI smoke tests and as the real
model in front of a GPU or a patient CPU. The engine process is a single
shared resource guarded by a lock; requests queue rather than spawning
engines, which keeps memory flat and predictable.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic_settings import BaseSettings, SettingsConfigDict

from opening_theory_engine import __version__
from opening_theory_engine.agent.explainer import Explainer, StubExplainer
from opening_theory_engine.agent.loop import ExplainVerifyLoop
from opening_theory_engine.config import EngineConfig
from opening_theory_engine.engine.analysis import Analyser, StockfishAnalyser
from opening_theory_engine.engine.stub import StubAnalyser
from opening_theory_engine.serving.schemas import (
    ClaimCheckOut,
    EngineFactsOut,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    VerificationOut,
)
from opening_theory_engine.utils import get_logger

_LOG = get_logger(__name__)


class ServeSettings(BaseSettings):
    """Serving configuration, read from OTE_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="OTE_")

    backend: Literal["stub", "hf"] = "stub"
    engine: Literal["stockfish", "stub"] = "stub"
    base_id: str = "Qwen/Qwen2.5-3B-Instruct"
    adapter_path: str | None = None
    four_bit: bool = False
    max_new_tokens: int = 220
    max_attempts: int = 2
    engine_depth: int = 12
    engine_threads: int = 1
    engine_hash_mb: int = 64


def _build_analyser(settings: ServeSettings) -> Analyser:
    if settings.engine == "stub":
        return StubAnalyser()
    return StockfishAnalyser(
        EngineConfig(
            depth=settings.engine_depth,
            threads=settings.engine_threads,
            hash_mb=settings.engine_hash_mb,
        )
    )


def _build_explainer(settings: ServeSettings, analyser: Analyser) -> Explainer:
    if settings.backend == "stub":
        return StubExplainer(analyser)
    from opening_theory_engine.agent.explainer import HfExplainer

    return HfExplainer(
        base_id=settings.base_id,
        adapter_path=settings.adapter_path,
        four_bit=settings.four_bit,
        max_new_tokens=settings.max_new_tokens,
    )


def create_app(settings: ServeSettings | None = None) -> FastAPI:
    """Application factory used by uvicorn and the tests."""
    resolved = settings or ServeSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        analyser = _build_analyser(resolved)
        explainer = _build_explainer(resolved, analyser)
        app.state.analyser = analyser
        app.state.explainer = explainer
        app.state.loop = ExplainVerifyLoop(
            explainer=explainer,
            analyser=analyser,
            max_attempts=resolved.max_attempts,
        )
        app.state.lock = threading.Lock()
        _LOG.info("serving with explainer=%s engine=%s", explainer.name, analyser.name)
        try:
            yield
        finally:
            analyser.close()

    app = FastAPI(
        title="Opening theory engine",
        description=(
            "Explanations of chess opening moves from a fine-tuned language model, "
            "with every response verified against engine analysis before it is returned."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": "opening-theory-engine", "docs": "/docs", "health": "/health"}

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            explainer=app.state.explainer.name,
            engine=app.state.analyser.name,
        )

    @app.post("/explain", response_model=ExplainResponse)
    async def explain(request: ExplainRequest) -> ExplainResponse:
        started = time.perf_counter()

        def _run() -> object:
            with app.state.lock:
                return app.state.loop.run(
                    fen=request.fen,
                    move=request.move,
                    opening=request.opening,
                    eco=request.eco,
                )

        try:
            outcome = await run_in_threadpool(_run)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        report = outcome.report
        facts = outcome.facts
        return ExplainResponse(
            explanation=outcome.explanation,
            attempts=outcome.attempts,
            verification=VerificationOut(
                passed=report.passed(),
                format_ok=report.format_ok,
                factual_accuracy=report.factual_accuracy,
                verdict_distance=report.verdict_distance,
                eval_distance=report.eval_distance,
                pv_match_plies=report.pv_match_plies,
                pv_illegal=report.pv_illegal,
                checks=[ClaimCheckOut(**check.to_dict()) for check in report.checks],
            ),
            engine=EngineFactsOut(
                best_move_san=facts.best_move_san,
                cp_best_white=facts.cp_best_white,
                cp_after_white=facts.cp_after_white,
                cp_loss=facts.cp_loss,
                is_best=facts.is_best,
                expected_line_san=list(facts.expected_line_san[:6]),
                material_delta_cp=facts.material_delta_cp,
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    return app
