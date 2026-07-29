"""Preference pair selection rules."""

import pandas as pd

from opening_theory_engine.alignment.pairs import PairBuilder, write_dpo_file

PROMPT = [{"role": "user", "content": "explain e4"}]


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["position_id", "prompt", "text", "score", "format_ok", "n_incorrect"],
    )


def test_picks_best_clean_versus_worst():
    frame = _frame(
        [
            ("p1", PROMPT, "excellent answer", 1.0, True, 0),
            ("p1", PROMPT, "fine answer", 0.4, True, 0),
            ("p1", PROMPT, "hallucinated answer", -1.2, True, 2),
        ]
    )
    pairs = PairBuilder(min_score_margin=0.5).build(frame)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["chosen"] == "excellent answer"
    assert row["rejected"] == "hallucinated answer"
    assert row["margin"] == 2.2


def test_uses_best_well_formed_when_none_clean():
    # No perfectly clean candidate, but the well-formed answer is preferable to
    # a broken one, so it still becomes the chosen target.
    frame = _frame(
        [
            ("p1", PROMPT, "formed but imperfect", 0.5, True, 1),
            ("p1", PROMPT, "broken", -0.5, False, 0),
        ]
    )
    pairs = PairBuilder(min_score_margin=0.1).build(frame)
    assert len(pairs) == 1
    assert pairs.iloc[0]["chosen"] == "formed but imperfect"
    assert pairs.iloc[0]["rejected"] == "broken"


def test_skips_when_no_well_formed_candidate():
    frame = _frame(
        [
            ("p1", PROMPT, "broken a", 0.5, False, 0),
            ("p1", PROMPT, "broken b", -0.5, False, 0),
        ]
    )
    assert PairBuilder(min_score_margin=0.1).build(frame).empty


def test_never_prefers_a_more_incorrect_answer():
    # The higher-scoring answer somehow carries more engine-flagged errors, so
    # no pair is emitted: we never teach the model to prefer the wronger one.
    frame = _frame(
        [
            ("p1", PROMPT, "high score more wrong", 0.9, True, 3),
            ("p1", PROMPT, "low score fewer wrong", 0.1, True, 1),
        ]
    )
    assert PairBuilder(min_score_margin=0.1).build(frame).empty


def test_margin_filter():
    frame = _frame(
        [
            ("p1", PROMPT, "a", 0.30, True, 0),
            ("p1", PROMPT, "b", 0.25, True, 0),
        ]
    )
    assert PairBuilder(min_score_margin=0.5).build(frame).empty


def test_missing_columns_rejected():
    import pytest

    with pytest.raises(ValueError, match="missing columns"):
        PairBuilder(0.1).build(pd.DataFrame({"position_id": []}))


def test_dpo_file_is_conversational(tmp_path):
    frame = _frame(
        [
            ("p1", PROMPT, "good", 1.0, True, 0),
            ("p1", PROMPT, "bad", -1.0, True, 3),
        ]
    )
    pairs = PairBuilder(min_score_margin=0.5).build(frame)
    path = tmp_path / "pairs.jsonl"
    count = write_dpo_file(pairs, path)
    assert count == 1
    import json

    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["prompt"] == PROMPT
    assert row["chosen"] == [{"role": "assistant", "content": "good"}]
    assert row["rejected"] == [{"role": "assistant", "content": "bad"}]
