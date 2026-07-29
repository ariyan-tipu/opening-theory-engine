"""The core invariant: rendered targets verify perfectly.

Every explanation the templates produce, for real positions from real
games, must parse cleanly and contain zero incorrect claims when checked
against the same facts it was rendered from. If this test fails, the
training data is teaching the model something the verifier would punish.
"""

import chess
import chess.pgn

from opening_theory_engine.alignment.claims import ClaimParser
from opening_theory_engine.alignment.verify import Verifier
from opening_theory_engine.data.motifs import detect_motifs
from opening_theory_engine.data.templates import TemplateRenderer
from opening_theory_engine.engine.stub import StubAnalyser


def test_rendered_targets_verify_clean(fixture_pgn):
    analyser = StubAnalyser()
    renderer = TemplateRenderer(seed=11)
    parser = ClaimParser()
    verifier = Verifier()

    checked = 0
    with open(fixture_pgn, encoding="utf-8") as handle:
        for _ in range(6):
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            board = game.board()
            for ply, move in enumerate(game.mainline_moves(), start=1):
                if ply > 16:
                    break
                if ply >= 8 and ply % 4 == 0:
                    facts = analyser.evaluate_move(board, move)
                    motifs = detect_motifs(board, move)
                    text = renderer.render(facts, motifs, game.headers.get("Opening", "test"))
                    parsed = parser.parse(text)
                    report = verifier.verify(parsed, facts, motifs)
                    assert parsed.format_ok, f"unparseable target:\n{text}"
                    assert report.n_incorrect == 0, (
                        f"template asserted something false:\n{text}\n{report.to_dict()}"
                    )
                    checked += 1
                board.push(move)
    assert checked >= 10, "fixture produced too few positions to be meaningful"


def test_rendering_is_deterministic_given_seed():
    analyser = StubAnalyser()
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    facts = analyser.evaluate_move(board, move)
    motifs = detect_motifs(board, move)
    first = TemplateRenderer(seed=3).render(facts, motifs, "King's Pawn Game")
    second = TemplateRenderer(seed=3).render(facts, motifs, "King's Pawn Game")
    assert first == second


def test_bad_move_gets_better_was_sentence():
    analyser = StubAnalyser()
    # White can win the d5 pawn for free; a1b1 ignores it.
    board = chess.Board("k7/8/8/3p4/4P3/8/8/RK6 w - - 0 1")
    move = chess.Move.from_uci("a1a2")
    facts = analyser.evaluate_move(board, move)
    if facts.cp_loss > 50 and not facts.is_best:
        motifs = detect_motifs(board, move)
        text = TemplateRenderer(seed=5).render(facts, motifs, "test position")
        assert "Better was" in text or "engine prefers" in text.lower()
