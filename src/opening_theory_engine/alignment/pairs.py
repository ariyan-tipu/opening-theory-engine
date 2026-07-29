"""Build DPO preference pairs from scored candidates.

The chosen side must be factually clean (well formed, zero incorrect
claims), the rejected side is the worst-scoring candidate for the same
position, and a minimum score margin filters out pairs where the
preference would mostly be noise. No human labels anywhere: the
preference signal is entirely engine ground truth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_theory_engine.utils import get_logger, write_jsonl

_LOG = get_logger(__name__)

REQUIRED_COLUMNS = frozenset({"position_id", "prompt", "text", "score", "format_ok", "n_incorrect"})


class PairBuilder:
    """Selects one chosen/rejected pair per position where the data supports it."""

    def __init__(self, min_score_margin: float) -> None:
        self._min_margin = min_score_margin

    def build(self, candidates: pd.DataFrame) -> pd.DataFrame:
        missing = REQUIRED_COLUMNS - set(candidates.columns)
        if missing:
            raise ValueError(f"candidate frame is missing columns: {sorted(missing)}")

        pairs = []
        skipped_no_target = 0
        skipped_no_gap = 0
        for position_id, group in candidates.groupby("position_id", sort=True):
            # The chosen answer has to be well formed so it can serve as a
            # training target. Prefer a perfectly clean one when the position
            # produced any; otherwise take the best-scoring well-formed answer.
            usable = group[group["format_ok"]]
            if usable.empty:
                skipped_no_target += 1
                continue
            clean = usable[usable["n_incorrect"] == 0]
            pool = clean if not clean.empty else usable
            chosen = pool.loc[pool["score"].idxmax()]
            rejected = group.loc[group["score"].idxmin()]

            # Only keep the pair when the chosen answer is genuinely preferable:
            # it makes no more engine-flagged errors than the rejected one (a
            # rejected answer that fails the format counts as worse than any
            # well-formed one), and the score gap is more than noise.
            chosen_not_worse = (not bool(rejected["format_ok"])) or (
                int(chosen["n_incorrect"]) <= int(rejected["n_incorrect"])
            )
            margin = float(chosen["score"]) - float(rejected["score"])
            if (
                chosen["text"] == rejected["text"]
                or not chosen_not_worse
                or margin < self._min_margin
            ):
                skipped_no_gap += 1
                continue
            pairs.append(
                {
                    "position_id": position_id,
                    "prompt": chosen["prompt"],
                    "chosen": chosen["text"],
                    "rejected": rejected["text"],
                    "score_chosen": float(chosen["score"]),
                    "score_rejected": float(rejected["score"]),
                    "margin": margin,
                }
            )

        _LOG.info(
            "built %d pairs (%d positions had no well-formed answer, %d had no usable gap)",
            len(pairs),
            skipped_no_target,
            skipped_no_gap,
        )
        return pd.DataFrame(pairs)


def write_dpo_file(pairs: pd.DataFrame, path: Path) -> int:
    """Write pairs in the conversational preference format trl expects."""
    rows = (
        {
            "prompt": row["prompt"],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        }
        for row in pairs.to_dict("records")
    )
    return write_jsonl(path, rows)
