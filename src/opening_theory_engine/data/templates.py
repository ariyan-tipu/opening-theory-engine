"""Render training targets from engine facts.

The supervision here is distant but verifiable: every sentence a template can
emit is either backed by a fact we verified (material windows, motif
flags, the engine's own line) or carries no checkable claim at all. The
test suite enforces the invariant that a rendered explanation, parsed
back through the claim parser and checked by the verifier, produces zero
incorrect claims.
"""

from __future__ import annotations

import random

import chess

from opening_theory_engine.alignment.bands import (
    EVAL_PHRASES,
    VERDICT_PHRASES,
    VerdictBand,
    eval_band_from_cp_white,
    verdict_from_cp_loss,
)
from opening_theory_engine.data.motifs import MotifFlags
from opening_theory_engine.engine.facts import MoveFacts

_WINS_PIECE_CP = 240
_WINS_MATERIAL_CP = 180
_WINS_PAWN_CP = 60
_LOSES_MATERIAL_CP = -60

_MATERIAL_PIECE = ("It wins a piece.", "The tactic wins a piece.")
_MATERIAL_GENERIC = ("It wins material.", "The move wins material by force.")
_MATERIAL_PAWN = ("It wins a pawn.", "The move picks up a pawn.")
_MATERIAL_LOSS = ("It loses material.", "The move drops material.")
_BETTER_WAS = ("Better was {san}.", "The engine prefers {san} here.")
_CASTLE = (
    "Castling tucks the king away and connects the rooks.",
    "It brings the king to safety by castling.",
)
_CHECK = ("It gives check.", "It comes with check, which limits the reply.")
_FORK = ("The move creates a fork.", "It forks two enemy pieces at once.")
_PIN = ("It pins an enemy piece.", "The move sets up a pin.")
_DEVELOP = (
    "It develops a minor piece towards useful squares.",
    "It continues development without losing time.",
)
_CENTRE = (
    "It fights for the central squares.",
    "It stakes a claim in the centre.",
)
_GENERIC_NAMED = (
    "A standard choice in the {opening}.",
    "Both sides are following the main ideas of the {opening}.",
)
_GENERIC_PLAIN = (
    "This keeps to well known opening theory.",
    "A sensible move that follows sound opening principles.",
)
_UNKNOWN_OPENING = "Unknown opening"


class TemplateRenderer:
    """Turns one annotated position into the four-line target format."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def render(self, facts: MoveFacts, motifs: MotifFlags, opening: str) -> str:
        verdict = verdict_from_cp_loss(facts.cp_loss, facts.is_best)
        eval_band = eval_band_from_cp_white(facts.cp_after_white)
        return (
            f"Verdict: {VERDICT_PHRASES[verdict]}\n"
            f"Position after: {EVAL_PHRASES[eval_band]}\n"
            f"Engine line: {self._engine_line(facts)}\n"
            f"Why: {self._why(facts, motifs, verdict, opening)}"
        )

    def _engine_line(self, facts: MoveFacts) -> str:
        board = chess.Board(facts.fen_before)
        moves = [chess.Move.from_uci(facts.move_uci)]
        moves += [chess.Move.from_uci(uci) for uci in facts.reply_pv_uci[:4]]
        legal_prefix: list[chess.Move] = []
        working = board.copy(stack=False)
        for move in moves:
            if move not in working.legal_moves:
                break
            legal_prefix.append(move)
            working.push(move)
        return board.variation_san(legal_prefix)

    def _why(
        self,
        facts: MoveFacts,
        motifs: MotifFlags,
        verdict: VerdictBand,
        opening: str,
    ) -> str:
        sentences: list[str] = []

        material = self._material_sentence(facts)
        if material:
            sentences.append(material)

        if not facts.is_best and verdict >= VerdictBand.INACCURACY:
            sentences.append(self._pick(_BETTER_WAS).format(san=facts.best_move_san))

        motif_bank: list[tuple[str, ...]] = []
        if motifs.is_castle:
            motif_bank.append(_CASTLE)
        if motifs.gives_check:
            motif_bank.append(_CHECK)
        if motifs.is_fork:
            motif_bank.append(_FORK)
        if motifs.creates_pin:
            motif_bank.append(_PIN)
        if motifs.develops_minor:
            motif_bank.append(_DEVELOP)
        if motifs.fights_for_centre:
            motif_bank.append(_CENTRE)
        self._rng.shuffle(motif_bank)
        for bank in motif_bank:
            if len(sentences) >= 3:
                break
            sentences.append(self._pick(bank))

        if len(sentences) < 2:
            if opening and opening != _UNKNOWN_OPENING:
                sentences.append(self._pick(_GENERIC_NAMED).format(opening=opening))
            else:
                sentences.append(self._pick(_GENERIC_PLAIN))
        return " ".join(sentences[:3])

    def _material_sentence(self, facts: MoveFacts) -> str | None:
        delta = facts.material_delta_cp
        if delta >= _WINS_PIECE_CP:
            return self._pick(_MATERIAL_PIECE)
        if delta >= _WINS_MATERIAL_CP:
            return self._pick(_MATERIAL_GENERIC)
        if delta >= _WINS_PAWN_CP:
            return self._pick(_MATERIAL_PAWN)
        if delta <= _LOSES_MATERIAL_CP:
            return self._pick(_MATERIAL_LOSS)
        return None

    def _pick(self, bank: tuple[str, ...]) -> str:
        return self._rng.choice(bank)
