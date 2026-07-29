"""Typed configuration for every pipeline stage, loaded from YAML.

One YAML file describes one run profile (see ``configs/``). Everything is
validated up front by pydantic so a typo in a config fails loudly at load
time rather than three stages into a run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class PathsConfig(BaseModel):
    """Where each stage reads and writes, relative to the working directory."""

    data_dir: Path = Path("data")
    artefacts_dir: Path = Path("artefacts")
    results_dir: Path = Path("results")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


class EngineConfig(BaseModel):
    """Stockfish settings. Depth and thread caps keep local runs light."""

    path: str | None = None
    depth: int = Field(default=12, ge=1, le=40)
    multipv: int = Field(default=2, ge=1, le=5)
    threads: int = Field(default=1, ge=1, le=16)
    hash_mb: int = Field(default=64, ge=16, le=8192)


class SplitsConfig(BaseModel):
    """Fractions for the three game-level splits."""

    train: float = Field(gt=0, lt=1)
    align: float = Field(gt=0, lt=1)
    eval: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _sums_to_one(self) -> SplitsConfig:
        total = self.train + self.align + self.eval
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"splits must sum to 1.0, got {total}")
        return self


class DataConfig(BaseModel):
    """Which games to keep and how many positions to draw from them."""

    pgn_url: str
    max_games_scanned: int = Field(gt=0)
    min_elo: int = Field(default=1800, ge=0)
    max_elo: int = Field(default=2400, ge=0)
    min_base_time_s: int = Field(default=180, ge=0)
    min_ply: int = Field(default=8, ge=1)
    max_ply: int = Field(default=24, ge=1)
    max_positions_per_game: int = Field(default=2, ge=1)
    max_unique_positions: int = Field(gt=0)
    max_cp_before: int = Field(default=800, gt=0)
    splits: SplitsConfig

    @model_validator(mode="after")
    def _ranges_make_sense(self) -> DataConfig:
        if self.min_elo >= self.max_elo:
            raise ValueError("min_elo must be below max_elo")
        if self.min_ply >= self.max_ply:
            raise ValueError("min_ply must be below max_ply")
        return self


class ModelConfig(BaseModel):
    """Base model choice and how to load it."""

    base_id: str
    four_bit: bool = False
    max_seq_len: int = Field(default=512, ge=64)


class SftConfig(BaseModel):
    """Supervised fine-tuning hyperparameters."""

    output_dir: Path
    epochs: int = Field(default=1, ge=1)
    max_steps: int = -1
    per_device_batch: int = Field(default=1, ge=1)
    grad_accum: int = Field(default=1, ge=1)
    learning_rate: float = Field(gt=0)
    warmup_ratio: float = Field(default=0.03, ge=0, le=0.5)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, le=0.5)
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=0, ge=0)


class GenerationConfig(BaseModel):
    """Candidate sampling for the alignment stage: one candidate per temperature."""

    temperatures: list[float] = Field(min_length=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    max_new_tokens: int = Field(default=200, ge=16)
    batch_size: int = Field(default=4, ge=1)
    max_positions: int = Field(default=1000, ge=1)


class DpoConfig(BaseModel):
    """Direct preference optimisation hyperparameters and pair selection."""

    output_dir: Path
    beta: float = Field(default=0.1, gt=0)
    epochs: int = Field(default=1, ge=1)
    max_steps: int = -1
    per_device_batch: int = Field(default=1, ge=1)
    grad_accum: int = Field(default=1, ge=1)
    learning_rate: float = Field(gt=0)
    logging_steps: int = Field(default=10, ge=1)
    min_score_margin: float = Field(default=0.15, ge=0)


class EvaluationConfig(BaseModel):
    """Held-out comparison of baseline, SFT and DPO systems."""

    systems: list[Literal["baseline", "sft", "dpo"]]
    max_positions: int = Field(default=200, ge=1)
    max_new_tokens: int = Field(default=200, ge=16)
    verdict_tolerance: int = Field(default=1, ge=0)
    eval_band_tolerance: int = Field(default=1, ge=0)


class WandbConfig(BaseModel):
    """Weights and Biases settings. ``auto`` logs online when a key is present."""

    project: str = "opening-theory-engine"
    mode: Literal["auto", "online", "offline", "disabled"] = "disabled"


class ProjectConfig(BaseModel):
    """Root of one run profile."""

    profile: str
    seed: int = 715
    paths: PathsConfig = PathsConfig()
    engine: EngineConfig = EngineConfig()
    data: DataConfig
    model: ModelConfig
    sft: SftConfig
    generation: GenerationConfig
    dpo: DpoConfig
    evaluation: EvaluationConfig
    wandb: WandbConfig = WandbConfig()


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a run profile from a YAML file."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ProjectConfig.model_validate(raw)
