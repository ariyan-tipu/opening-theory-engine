"""The data build end to end on the bundled fixture, engine stubbed."""

import json

from conftest import make_test_config
from opening_theory_engine.config import SplitsConfig
from opening_theory_engine.data.dataset import assign_split
from opening_theory_engine.data.pipeline import run_data_build
from opening_theory_engine.utils import read_jsonl


def test_full_data_build_offline(tmp_path, fixture_pgn):
    config = make_test_config(tmp_path, pgn_url=str(fixture_pgn))
    bundle = run_data_build(config, use_stub_engine=True, quiet=True)

    rows = read_jsonl(bundle.positions_path)
    assert rows, "no positions written"
    assert sum(bundle.split_counts.values()) == len(rows)

    # Split assignment must be game-level: one game never straddles splits.
    split_by_game = {}
    for row in rows:
        previous = split_by_game.setdefault(row["game_id"], row["split"])
        assert previous == row["split"], f"game {row['game_id']} leaked across splits"

    # Every row carries verifiable facts, motifs and a rendered target.
    sample = rows[0]
    assert {"facts", "motifs", "target", "split", "position_id"} <= set(sample)
    assert sample["facts"]["cp_loss"] >= 0
    assert sample["target"].startswith("Verdict:")

    sft_rows = read_jsonl(bundle.sft_path)
    train_count = sum(1 for row in rows if row["split"] == "train")
    assert len(sft_rows) == train_count
    first = sft_rows[0]
    assert first["prompt"][0]["role"] == "system"
    assert first["completion"][0]["role"] == "assistant"

    card = bundle.card_path.read_text(encoding="utf-8")
    assert "Unique positions" in card


def test_annotation_cache_makes_second_build_cheap(tmp_path, fixture_pgn):
    config = make_test_config(tmp_path, pgn_url=str(fixture_pgn))
    first = run_data_build(config, use_stub_engine=True, quiet=True)
    cache_file = config.paths.work_dir / "annotations_test.jsonl"
    cached_lines = cache_file.read_text(encoding="utf-8").count("\n")
    second = run_data_build(config, use_stub_engine=True, quiet=True)
    assert cached_lines == cache_file.read_text(encoding="utf-8").count("\n")
    assert first.split_counts == second.split_counts


def test_assign_split_is_deterministic():
    splits = SplitsConfig(train=0.7, align=0.2, eval=0.1)
    assert assign_split("abcd1234", splits, seed=7) == assign_split("abcd1234", splits, seed=7)
    buckets = {assign_split(f"game{i}", splits, seed=7) for i in range(200)}
    assert buckets == {"train", "align", "eval"}


def test_dataset_is_valid_json_lines(tmp_path, fixture_pgn):
    config = make_test_config(tmp_path, pgn_url=str(fixture_pgn))
    bundle = run_data_build(config, use_stub_engine=True, quiet=True)
    with open(bundle.sft_path, encoding="utf-8") as handle:
        for line in handle:
            json.loads(line)
