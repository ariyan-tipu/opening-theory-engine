"""Sample candidate explanations from the SFT model for alignment.

One candidate per configured temperature per position: low temperatures
give the model's confident answer, higher ones surface the failure modes
(wrong verdicts, invented material claims) that the verifier then catches
and DPO trains away. Positions come from the ``align`` split only, so
neither SFT training rows nor held-out evaluation rows are reused.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_theory_engine.config import ProjectConfig
from opening_theory_engine.data.prompts import build_prompt_messages
from opening_theory_engine.modelling.hf import (
    GenerationSettings,
    generate_texts,
    load_model,
    load_tokenizer,
)
from opening_theory_engine.utils import get_logger, read_jsonl, set_seed, write_jsonl

_LOG = get_logger(__name__)


def select_align_positions(config: ProjectConfig) -> pd.DataFrame:
    """The align-split rows this stage may use, capped and deterministic."""
    positions_path = config.paths.processed_dir / "positions_annotated.jsonl"
    if not positions_path.is_file():
        raise FileNotFoundError(f"{positions_path} not found; run 'ote data build' first")
    frame = pd.DataFrame(read_jsonl(positions_path))
    frame = frame[frame["split"] == "align"]
    if frame.empty:
        raise RuntimeError("no align-split positions; check the data build")
    cap = config.generation.max_positions
    if len(frame) > cap:
        frame = frame.sample(n=cap, random_state=config.seed)
    return frame.sort_values("position_id").reset_index(drop=True)


def generate_candidates(config: ProjectConfig, adapter_dir: Path | None = None) -> Path:
    """Write ``candidates.jsonl``: sampled explanations for every align position."""
    set_seed(config.seed)
    positions = select_align_positions(config)
    adapter = str(adapter_dir or config.sft.output_dir / "adapter")

    tokenizer = load_tokenizer(config.model.base_id)
    model = load_model(config.model.base_id, four_bit=config.model.four_bit, adapter_path=adapter)

    conversations = [
        build_prompt_messages(
            opening=row["opening"],
            eco=row["eco"],
            fen_before=row["fen_before"],
            move_san=row["move_san"],
            turn_white=bool(row["turn_white"]),
        )
        for row in positions.to_dict("records")
    ]

    rows = []
    for temperature in config.generation.temperatures:
        _LOG.info("sampling %d positions at temperature %.2f", len(conversations), temperature)
        settings = GenerationSettings(
            max_new_tokens=config.generation.max_new_tokens,
            temperature=temperature,
            top_p=config.generation.top_p,
        )
        texts = generate_texts(
            model,
            tokenizer,
            conversations,
            settings,
            batch_size=config.generation.batch_size,
        )
        rows.extend(
            {
                "position_id": row["position_id"],
                "temperature": temperature,
                "text": text,
            }
            for row, text in zip(positions.to_dict("records"), texts, strict=True)
        )

    candidates_path = config.paths.work_dir / "candidates.jsonl"
    count = write_jsonl(candidates_path, rows)
    _LOG.info("wrote %d candidates to %s", count, candidates_path)
    return candidates_path
