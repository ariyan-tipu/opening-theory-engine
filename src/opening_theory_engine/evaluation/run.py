"""Compare baseline, SFT and DPO models on the held-out split.

Every system answers the same positions with greedy decoding, every answer
goes through the same claim parser and verifier, and the results land in
``results/`` as JSON plus a small markdown table. Nothing in here is
subjective: if the table says the DPO model hallucinates less, that is a
count of engine-checked claims, not an opinion.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pandas as pd

from opening_theory_engine.alignment.claims import ClaimParser
from opening_theory_engine.alignment.verify import Verifier
from opening_theory_engine.config import ProjectConfig
from opening_theory_engine.data.motifs import MotifFlags
from opening_theory_engine.data.prompts import build_prompt_messages
from opening_theory_engine.engine.facts import MoveFacts
from opening_theory_engine.evaluation.metrics import aggregate_reports, fluency_stats
from opening_theory_engine.utils import ensure_dir, get_logger, read_jsonl, set_seed, write_jsonl

_LOG = get_logger(__name__)

_SUMMARY_METRICS = (
    "format_rate",
    "verdict_exact",
    "verdict_within_one",
    "eval_within_one",
    "line_illegal_rate",
    "claim_precision",
    "hallucination_rate",
    "verified_pass_rate",
    "composite_score_mean",
)


def run_evaluation(config: ProjectConfig, systems: list[str] | None = None) -> dict[str, Any]:
    """Evaluate the requested systems and write results/summary files."""
    set_seed(config.seed)
    rows = _eval_rows(config)
    conversations = [
        build_prompt_messages(
            opening=row["opening"],
            eco=row["eco"],
            fen_before=row["fen_before"],
            move_san=row["move_san"],
            turn_white=bool(row["turn_white"]),
        )
        for row in rows
    ]
    parser = ClaimParser()
    verifier = Verifier(
        verdict_tolerance=config.evaluation.verdict_tolerance,
        eval_tolerance=config.evaluation.eval_band_tolerance,
    )

    results_dir = ensure_dir(config.paths.results_dir)
    summary: dict[str, Any] = {"profile": config.profile, "positions": len(rows), "systems": {}}
    for system in systems or list(config.evaluation.systems):
        _LOG.info("evaluating system: %s", system)
        texts = _generate_for_system(config, system, conversations)
        reports = [
            verifier.verify(
                parser.parse(text),
                MoveFacts.from_dict(row["facts"]),
                MotifFlags.from_dict(row["motifs"]),
            )
            for text, row in zip(texts, rows, strict=True)
        ]
        metrics = aggregate_reports(reports)
        metrics["fluency"] = fluency_stats(texts)
        summary["systems"][system] = metrics

        write_jsonl(
            results_dir / f"{system}_responses.jsonl",
            (
                {
                    "position_id": row["position_id"],
                    "prompt": conversation[-1]["content"],
                    "response": text,
                    "verification": report.to_dict(),
                }
                for row, conversation, text, report in zip(
                    rows, conversations, texts, reports, strict=True
                )
            ),
        )

    summary_path = results_dir / "summary.json"
    import json

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (results_dir / "summary.md").write_text(render_summary_table(summary), encoding="utf-8")
    _LOG.info("evaluation written to %s", results_dir)
    return summary


def _eval_rows(config: ProjectConfig) -> list[dict[str, Any]]:
    positions_path = config.paths.processed_dir / "positions_annotated.jsonl"
    if not positions_path.is_file():
        raise FileNotFoundError(f"{positions_path} not found; run 'ote data build' first")
    frame = pd.DataFrame(read_jsonl(positions_path))
    frame = frame[frame["split"] == "eval"]
    if frame.empty:
        raise RuntimeError("no eval-split positions; check the data build")
    cap = config.evaluation.max_positions
    if len(frame) > cap:
        frame = frame.sample(n=cap, random_state=config.seed)
    return frame.sort_values("position_id").to_dict("records")


def _generate_for_system(
    config: ProjectConfig,
    system: str,
    conversations: list[list[dict[str, str]]],
) -> list[str]:
    from opening_theory_engine.modelling.hf import (
        GenerationSettings,
        generate_texts,
        load_model,
        load_tokenizer,
    )

    adapter = _adapter_for_system(config, system)
    tokenizer = load_tokenizer(config.model.base_id)
    model = load_model(config.model.base_id, four_bit=config.model.four_bit, adapter_path=adapter)
    try:
        settings = GenerationSettings(
            max_new_tokens=config.evaluation.max_new_tokens, temperature=0.0
        )
        return generate_texts(
            model, tokenizer, conversations, settings, batch_size=config.generation.batch_size
        )
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _adapter_for_system(config: ProjectConfig, system: str) -> str | None:
    if system == "baseline":
        return None
    if system == "sft":
        adapter = config.sft.output_dir / "adapter"
    elif system == "dpo":
        adapter = config.dpo.output_dir / "adapter"
    else:
        raise ValueError(f"unknown system '{system}'")
    if not Path(adapter).is_dir():
        raise FileNotFoundError(
            f"adapter for system '{system}' not found at {adapter}; train it first"
        )
    return str(adapter)


def render_summary_table(summary: dict[str, Any]) -> str:
    """A small markdown table, ready to paste into the README results section."""
    systems = list(summary["systems"].keys())
    lines = [
        f"# Evaluation summary ({summary['profile']}, {summary['positions']} held-out positions)",
        "",
        "| Metric | " + " | ".join(systems) + " |",
        "| --- | " + " | ".join(["---"] * len(systems)) + " |",
    ]
    for metric in _SUMMARY_METRICS:
        cells = [str(summary["systems"][system].get(metric, "")) for system in systems]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    for fluency_key in ("mean_tokens", "distinct_2", "repetition_rate"):
        cells = [
            str(summary["systems"][system].get("fluency", {}).get(fluency_key, ""))
            for system in systems
        ]
        lines.append(f"| fluency: {fluency_key} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)
