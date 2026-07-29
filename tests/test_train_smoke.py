"""End-to-end training smoke: a few optimiser steps of every stage on CPU.

Gated behind OTE_RUN_TRAIN_TESTS=1 because it downloads a small model
(SmolLM2-135M) and takes a few minutes. CI runs it in a dedicated job;
locally it is opt-in. Quality is irrelevant here; what is being proven is
that the SFT, generation, pairing and DPO code paths run and produce
loadable artefacts with the real libraries.
"""

import os

import pytest

from conftest import make_test_config

pytestmark = [
    pytest.mark.train,
    pytest.mark.skipif(
        os.environ.get("OTE_RUN_TRAIN_TESTS") != "1",
        reason="set OTE_RUN_TRAIN_TESTS=1 to run training smoke tests",
    ),
]


def test_sft_candidates_pairs_dpo_eval(tmp_path, fixture_pgn):
    from opening_theory_engine.alignment.pairs import PairBuilder, write_dpo_file
    from opening_theory_engine.alignment.pipeline import build_preference_pairs
    from opening_theory_engine.data.pipeline import run_data_build
    from opening_theory_engine.evaluation.run import run_evaluation
    from opening_theory_engine.training.candidates import generate_candidates
    from opening_theory_engine.training.dpo import run_dpo
    from opening_theory_engine.training.sft import run_sft
    from opening_theory_engine.utils import read_jsonl

    config = make_test_config(tmp_path, pgn_url=str(fixture_pgn))

    bundle = run_data_build(config, use_stub_engine=True, quiet=True)
    assert bundle.split_counts.get("train", 0) > 0

    sft_adapter = run_sft(config)
    assert (sft_adapter / "adapter_config.json").is_file()

    candidates_path = generate_candidates(config)
    candidates = read_jsonl(candidates_path)
    assert len(candidates) == 2 * config.generation.max_positions

    # A two-step toy model will not produce format-clean candidates, so the
    # margin filter may legitimately reject everything. The pairing logic is
    # unit-tested elsewhere; for the DPO smoke we fall back to synthetic
    # pairs built from real prompts and targets.
    try:
        pairs_path = build_preference_pairs(config)
    except RuntimeError:
        import pandas as pd

        rows = read_jsonl(bundle.positions_path)[:4]
        frame = pd.DataFrame(
            {
                "position_id": [row["position_id"] for row in rows],
                "prompt": [
                    [
                        {"role": "system", "content": "explain the move"},
                        {"role": "user", "content": row["fen_before"]},
                    ]
                    for row in rows
                ],
                "text": [row["target"] for row in rows],
                "score": [1.0] * len(rows),
                "format_ok": [True] * len(rows),
                "n_incorrect": [0] * len(rows),
            }
        )
        rejected = frame.copy()
        rejected["text"] = rejected["text"] + " It wins a piece."
        rejected["score"] = -1.0
        rejected["n_incorrect"] = 1
        pairs = PairBuilder(min_score_margin=0.5).build(
            pd.concat([frame, rejected], ignore_index=True)
        )
        pairs_path = config.paths.processed_dir / "dpo_pairs.jsonl"
        write_dpo_file(pairs, pairs_path)
    assert read_jsonl(pairs_path), "no DPO pairs available for the smoke run"

    dpo_adapter = run_dpo(config)
    assert (dpo_adapter / "adapter_config.json").is_file()

    summary = run_evaluation(config, systems=["sft"])
    assert "sft" in summary["systems"]
    assert (config.paths.results_dir / "summary.md").is_file()
