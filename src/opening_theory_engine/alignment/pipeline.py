"""Score sampled candidates against the engine reference and build DPO pairs.

This stage is pure CPU work: parse each candidate's claims, verify them
against the stored engine facts, score everything in one JAX pass, then
keep the pairs whose preference the evidence actually supports.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_theory_engine.alignment.claims import ClaimParser
from opening_theory_engine.alignment.pairs import PairBuilder, write_dpo_file
from opening_theory_engine.alignment.scoring import score_reports
from opening_theory_engine.alignment.verify import Verifier
from opening_theory_engine.config import ProjectConfig
from opening_theory_engine.data.motifs import MotifFlags
from opening_theory_engine.data.prompts import build_prompt_messages
from opening_theory_engine.engine.facts import MoveFacts
from opening_theory_engine.utils import get_logger, read_jsonl, write_json

_LOG = get_logger(__name__)


def build_preference_pairs(config: ProjectConfig) -> Path:
    """Verify and score every candidate, then write ``dpo_pairs.jsonl``."""
    processed = config.paths.processed_dir
    positions = {
        row["position_id"]: row for row in read_jsonl(processed / "positions_annotated.jsonl")
    }
    candidates = read_jsonl(config.paths.work_dir / "candidates.jsonl")
    if not candidates:
        raise RuntimeError("no candidates found; run 'ote generate candidates' first")

    parser = ClaimParser()
    verifier = Verifier(
        verdict_tolerance=config.evaluation.verdict_tolerance,
        eval_tolerance=config.evaluation.eval_band_tolerance,
    )

    reports = []
    kept_rows = []
    for candidate in candidates:
        position = positions.get(candidate["position_id"])
        if position is None:
            continue
        parsed = parser.parse(candidate["text"])
        report = verifier.verify(
            parsed,
            MoveFacts.from_dict(position["facts"]),
            MotifFlags.from_dict(position["motifs"]),
        )
        reports.append(report)
        kept_rows.append((candidate, position, report))

    scores = score_reports(reports)
    frame = pd.DataFrame(
        {
            "position_id": [candidate["position_id"] for candidate, _, _ in kept_rows],
            "prompt": [
                build_prompt_messages(
                    opening=position["opening"],
                    eco=position["eco"],
                    fen_before=position["fen_before"],
                    move_san=position["move_san"],
                    turn_white=bool(position["turn_white"]),
                )
                for _, position, _ in kept_rows
            ],
            "text": [candidate["text"] for candidate, _, _ in kept_rows],
            "score": scores,
            "format_ok": [report.format_ok for _, _, report in kept_rows],
            "n_incorrect": [report.n_incorrect for _, _, report in kept_rows],
        }
    )

    pairs = PairBuilder(config.dpo.min_score_margin).build(frame)
    if pairs.empty:
        raise RuntimeError(
            "no preference pairs cleared the margin; the SFT model may already be "
            "uniform, or generation needs more temperature spread"
        )
    pairs_path = processed / "dpo_pairs.jsonl"
    count = write_dpo_file(pairs, pairs_path)

    write_json(
        config.paths.work_dir / "pair_stats.json",
        {
            "candidates_scored": len(kept_rows),
            "pairs_written": count,
            "mean_margin": float(pairs["margin"].mean()),
            "mean_score_chosen": float(pairs["score_chosen"].mean()),
            "mean_score_rejected": float(pairs["score_rejected"].mean()),
        },
    )
    _LOG.info("wrote %d preference pairs to %s", count, pairs_path)
    return pairs_path
