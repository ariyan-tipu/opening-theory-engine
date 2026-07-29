"""Direct preference optimisation on engine-verified pairs.

The SFT adapter is the starting point and stays trainable; trl handles the
reference policy by disabling the adapter for the reference forward pass,
so no second copy of the model is needed, which is what keeps this stage
inside a free T4's memory.
"""

from __future__ import annotations

from pathlib import Path

from opening_theory_engine.config import ProjectConfig
from opening_theory_engine.modelling.hf import load_model, load_tokenizer
from opening_theory_engine.training.common import (
    cast_trainable_to_fp32,
    precision_flags,
    resolve_report_to,
)
from opening_theory_engine.utils import get_logger, set_seed, write_json

_LOG = get_logger(__name__)


def run_dpo(config: ProjectConfig, sft_adapter_dir: Path | None = None) -> Path:
    """Train the DPO adapter on the preference pairs and return its directory."""
    import torch
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    set_seed(config.seed)
    pairs_path = config.paths.processed_dir / "dpo_pairs.jsonl"
    if not pairs_path.is_file():
        raise FileNotFoundError(f"{pairs_path} not found; run 'ote align pairs' first")
    dataset = load_dataset("json", data_files=str(pairs_path), split="train")
    _LOG.info("DPO dataset: %d preference pairs", len(dataset))

    adapter = str(sft_adapter_dir or config.sft.output_dir / "adapter")
    tokenizer = load_tokenizer(config.model.base_id)
    model = load_model(
        config.model.base_id,
        four_bit=config.model.four_bit,
        adapter_path=adapter,
        train_mode=True,
    )
    quantised = config.model.four_bit and torch.cuda.is_available()
    if quantised:
        model = cast_trainable_to_fp32(model)
        model.enable_input_require_grads()  # needed for checkpointing through a frozen 4-bit base
    # Same reasoning as SFT: skip the mixed-precision gradient scaler for QLoRA.
    precision = {"fp16": False, "bf16": False} if quantised else precision_flags()

    run_name = f"dpo-{config.profile}"
    args = DPOConfig(
        output_dir=str(config.dpo.output_dir),
        beta=config.dpo.beta,
        num_train_epochs=config.dpo.epochs,
        max_steps=config.dpo.max_steps,
        per_device_train_batch_size=config.dpo.per_device_batch,
        gradient_accumulation_steps=config.dpo.grad_accum,
        learning_rate=config.dpo.learning_rate,
        logging_steps=config.dpo.logging_steps,
        save_strategy="no",
        report_to=resolve_report_to(config.wandb, run_name),
        run_name=run_name,
        seed=config.seed,
        max_length=config.model.max_seq_len,
        gradient_checkpointing=torch.cuda.is_available(),
        **precision,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    result = trainer.train()

    adapter_dir = config.dpo.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_json(
        config.dpo.output_dir / "dpo_metrics.json",
        {
            "train_loss": result.training_loss,
            "steps": int(result.global_step),
            "pairs": len(dataset),
            "beta": config.dpo.beta,
            "base_model": config.model.base_id,
            "profile": config.profile,
        },
    )
    _LOG.info("DPO finished: loss %.4f after %d steps", result.training_loss, result.global_step)
    return adapter_dir
