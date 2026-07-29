"""Engine analysis: a thin, typed layer over Stockfish plus a test stub."""

from opening_theory_engine.engine.analysis import (
    Analyser,
    EngineNotFoundError,
    StockfishAnalyser,
    resolve_engine_path,
)
from opening_theory_engine.engine.facts import EngineLine, MoveFacts
from opening_theory_engine.engine.stub import StubAnalyser

__all__ = [
    "Analyser",
    "EngineLine",
    "EngineNotFoundError",
    "MoveFacts",
    "StockfishAnalyser",
    "StubAnalyser",
    "resolve_engine_path",
]
