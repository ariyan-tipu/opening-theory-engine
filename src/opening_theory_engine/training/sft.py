"""Supervised fine-tuning with QLoRA.

The dataset arrives as prompt and completion message lists, and
``completion_only_loss`` keeps the loss on the assistant's answer rather
than the prompt. On a CUDA machine with ``model.four_bit`` set, the base
model loads in 4-bit NF4 and only the LoRA adapters train; on CPU the same
code runs unquantised, which is how the laptop smoke profile and CI prove
the code path without a GPU.
"""

from __future__ import annotations

from pathlib import Path

from opening_theory_engine.config import ProjectConfig
from opening_theory_engine.modelling.hf import load_model, load_tokenizer
from opening_theory_engine.training.common import (
    build_lora_config,
    precision_flags,
    prepare_quantised_for_training,
    resolve_report_to,
)
from opening_theory_engine.utils import get_logger, set_seed, write_json

_LOG = get_logger(__name__)


def _last_checkpoint(output_dir: Path) -> str | None:
    """Return the newest resumable checkpoint under ``output_dir``, if any."""
    if not Path(output_dir).is_dir():
        return None
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(str(output_dir))


def run_sft(config: ProjectConfig) -> Path:
    """Train the SFT adapter and return the directory it was saved to."""
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    set_seed(config.seed)
    sft_path = config.paths.processed_dir / "sft_train.jsonl"
    if not sft_path.is_file():
        raise FileNotFoundError(f"{sft_path} not found; run 'ote data build' first")
    dataset = load_dataset("json", data_files=str(sft_path), split="train")
    _LOG.info("SFT dataset: %d examples", len(dataset))

    tokenizer = load_tokenizer(config.model.base_id)
    model = load_model(config.model.base_id, four_bit=config.model.four_bit, train_mode=True)
    quantised = config.model.four_bit and torch.cuda.is_available()
    model = prepare_quantised_for_training(model, config.model.four_bit)

    # Periodic checkpoints let a reclaimed free-tier session resume instead of
    # losing the whole run. save_steps <= 0 keeps the old save-only-at-the-end
    # behaviour used by the laptop smoke profile.
    checkpointing = config.sft.save_steps > 0
    # QLoRA trains without an fp16/bf16 gradient scaler. The frozen 4-bit base
    # already computes in its fast float16 compute dtype inside bitsandbytes, so
    # skipping mixed precision keeps the speed while avoiding the scaler, which
    # cannot unscale the adapter's non-float32 gradients on a T4.
    precision = {"fp16": False, "bf16": False} if quantised else precision_flags()
    run_name = f"sft-{config.profile}"
    args = SFTConfig(
        output_dir=str(config.sft.output_dir),
        num_train_epochs=config.sft.epochs,
        max_steps=config.sft.max_steps,
        per_device_train_batch_size=config.sft.per_device_batch,
        gradient_accumulation_steps=config.sft.grad_accum,
        learning_rate=config.sft.learning_rate,
        warmup_ratio=config.sft.warmup_ratio,
        logging_steps=config.sft.logging_steps,
        save_strategy="steps" if checkpointing else "no",
        save_steps=config.sft.save_steps if checkpointing else 500,
        save_total_limit=1,
        report_to=resolve_report_to(config.wandb, run_name),
        run_name=run_name,
        seed=config.seed,
        max_length=config.model.max_seq_len,
        packing=False,
        completion_only_loss=True,
        # prepare_quantised_for_training already enables checkpointing for the
        # 4-bit path, so only turn it on here for a plain (non-quantised) GPU.
        gradient_checkpointing=torch.cuda.is_available() and not quantised,
        dataset_num_proc=1,
        **precision,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(config.sft),
    )
    resume = _last_checkpoint(config.sft.output_dir) if checkpointing else None
    if resume:
        _LOG.info("resuming SFT from %s", resume)
    result = trainer.train(resume_from_checkpoint=resume)

    adapter_dir = config.sft.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_json(
        config.sft.output_dir / "sft_metrics.json",
        {
            "train_loss": result.training_loss,
            "steps": int(result.global_step),
            "examples": len(dataset),
            "base_model": config.model.base_id,
            "profile": config.profile,
        },
    )
    _LOG.info("SFT finished: loss %.4f after %d steps", result.training_loss, result.global_step)
    return adapter_dir
