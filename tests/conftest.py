"""Shared fixtures: stub engine, synthetic facts and a tiny test profile."""

from __future__ import annotations

from pathlib import Path

import pytest

from opening_theory_engine.config import (
    DataConfig,
    DpoConfig,
    EngineConfig,
    EvaluationConfig,
    GenerationConfig,
    ModelConfig,
    PathsConfig,
    ProjectConfig,
    SftConfig,
    SplitsConfig,
    WandbConfig,
)
from opening_theory_engine.data.motifs import MotifFlags
from opening_theory_engine.engine.facts import MoveFacts
from opening_theory_engine.engine.stub import StubAnalyser

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_pgn() -> Path:
    return FIXTURES_DIR / "games_small.pgn"


@pytest.fixture()
def stub_analyser() -> StubAnalyser:
    return StubAnalyser()


def make_facts(**overrides) -> MoveFacts:
    """Synthetic MoveFacts for verifier tests: 1. e4 judged best, equal position."""
    base = {
        "fen_before": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "move_uci": "e2e4",
        "move_san": "e4",
        "turn_white": True,
        "cp_best_white": 30,
        "cp_after_white": 30,
        "cp_loss": 0,
        "is_best": True,
        "best_move_uci": "e2e4",
        "best_move_san": "e4",
        "pv_uci": ("e2e4", "e7e5", "g1f3"),
        "pv_san": ("e4", "e5", "Nf3"),
        "reply_pv_uci": ("e7e5", "g1f3"),
        "reply_pv_san": ("e5", "Nf3"),
        "material_delta_cp": 0,
    }
    base.update(overrides)
    return MoveFacts(**base)


def make_motifs(**overrides) -> MotifFlags:
    """MotifFlags with everything off unless a test switches it on."""
    base = {
        "is_capture": False,
        "gives_check": False,
        "is_castle": False,
        "is_promotion": False,
        "is_centre_pawn_push": False,
        "develops_minor": False,
        "wins_material_see": False,
        "creates_pin": False,
        "is_fork": False,
        "centre_attacks_delta": 0,
        "development_delta": 0,
    }
    base.update(overrides)
    return MotifFlags(**base)


def make_test_config(tmp_path: Path, pgn_url: str) -> ProjectConfig:
    """A complete profile small enough to run inside a test."""
    return ProjectConfig(
        profile="test",
        seed=7,
        paths=PathsConfig(
            data_dir=tmp_path / "data",
            artefacts_dir=tmp_path / "artefacts",
            results_dir=tmp_path / "results",
        ),
        engine=EngineConfig(depth=6, multipv=2, threads=1, hash_mb=16),
        data=DataConfig(
            pgn_url=pgn_url,
            max_games_scanned=100,
            min_elo=1800,
            max_elo=2400,
            min_base_time_s=180,
            min_ply=8,
            max_ply=24,
            max_positions_per_game=2,
            max_unique_positions=40,
            splits=SplitsConfig(train=0.5, align=0.3, eval=0.2),
        ),
        model=ModelConfig(base_id="HuggingFaceTB/SmolLM2-135M-Instruct", max_seq_len=512),
        sft=SftConfig(
            output_dir=tmp_path / "artefacts" / "sft",
            max_steps=2,
            learning_rate=2.0e-4,
            lora_r=4,
            lora_alpha=8,
        ),
        generation=GenerationConfig(
            temperatures=[0.2, 1.0],
            max_new_tokens=48,
            batch_size=1,
            max_positions=2,
        ),
        dpo=DpoConfig(
            output_dir=tmp_path / "artefacts" / "dpo",
            max_steps=1,
            learning_rate=5.0e-6,
        ),
        evaluation=EvaluationConfig(systems=["baseline"], max_positions=3, max_new_tokens=48),
        wandb=WandbConfig(mode="disabled"),
    )
