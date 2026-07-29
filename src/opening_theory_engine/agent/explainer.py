"""Explainer backends.

Two implementations of one small interface: the real model, and a stub
that renders straight from engine facts. The stub exists so the API, the
tests and CI can exercise the whole explain-verify-regenerate loop
deterministically; its optional ``corrupt_first`` mode injects a false
material claim on the first attempt, which is how the tests prove the
verifier actually catches hallucinations rather than waving them through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import chess

from opening_theory_engine.data.motifs import detect_motifs
from opening_theory_engine.data.prompts import build_prompt_messages
from opening_theory_engine.data.templates import TemplateRenderer
from opening_theory_engine.engine.analysis import Analyser


@dataclass(frozen=True)
class ExplainContext:
    """Everything an explainer may use about the position being explained."""

    fen_before: str
    move_san: str
    move_uci: str
    turn_white: bool
    opening: str = "Unknown opening"
    eco: str = "?"

    def messages(self) -> list[dict[str, str]]:
        return build_prompt_messages(
            opening=self.opening,
            eco=self.eco,
            fen_before=self.fen_before,
            move_san=self.move_san,
            turn_white=self.turn_white,
        )


class Explainer(Protocol):
    """Proposes an explanation for one move. May be wrong; the verifier decides."""

    @property
    def name(self) -> str: ...

    def explain(self, context: ExplainContext, temperature: float) -> str: ...


class HfExplainer:
    """The fine-tuned model, loaded once and sampled per request."""

    def __init__(
        self,
        base_id: str,
        adapter_path: str | None = None,
        four_bit: bool = False,
        max_new_tokens: int = 220,
    ) -> None:
        from opening_theory_engine.modelling.hf import load_model, load_tokenizer

        self._base_id = base_id
        self._adapter_path = adapter_path
        self._max_new_tokens = max_new_tokens
        self._tokenizer = load_tokenizer(base_id)
        self._model = load_model(base_id, four_bit=four_bit, adapter_path=adapter_path)

    @property
    def name(self) -> str:
        suffix = " + adapter" if self._adapter_path else ""
        return f"hf:{self._base_id}{suffix}"

    def explain(self, context: ExplainContext, temperature: float) -> str:
        from opening_theory_engine.modelling.hf import GenerationSettings, generate_texts

        settings = GenerationSettings(
            max_new_tokens=self._max_new_tokens,
            temperature=temperature,
        )
        return generate_texts(
            self._model, self._tokenizer, [context.messages()], settings, batch_size=1
        )[0]


class StubExplainer:
    """Template renderer posing as a model. Deterministic, instant, no torch."""

    def __init__(self, analyser: Analyser, corrupt_first: bool = False, seed: int = 0) -> None:
        self._analyser = analyser
        self._renderer = TemplateRenderer(seed=seed)
        self._corrupt_first = corrupt_first
        self._calls = 0

    @property
    def name(self) -> str:
        return "stub-explainer"

    def explain(self, context: ExplainContext, temperature: float) -> str:
        del temperature  # deterministic by design
        board = chess.Board(context.fen_before)
        move = chess.Move.from_uci(context.move_uci)
        facts = self._analyser.evaluate_move(board, move)
        motifs = detect_motifs(board, move)
        text = self._renderer.render(facts, motifs, context.opening)
        self._calls += 1
        if self._corrupt_first and self._calls == 1:
            text += " It wins a piece."
        return text
