"""Shared training plumbing: LoRA targets, precision flags, W&B wiring."""

from __future__ import annotations

import os
from typing import Any

from opening_theory_engine.config import SftConfig, WandbConfig
from opening_theory_engine.utils import get_logger

_LOG = get_logger(__name__)

# Attention and MLP projections shared by the Llama-style decoder families
# this project targets (Qwen2.5 and SmolLM2 both use these module names).
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def build_lora_config(config: SftConfig) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )


def prepare_quantised_for_training(model: Any, four_bit: bool) -> Any:
    """Ready a 4-bit model for float16 mixed-precision training.

    Casts the trainable (LoRA) parameters to float32 and re-enables gradient
    checkpointing. This is what lets the float16 gradient scaler work on a T4:
    it can unscale float32 gradients but not bfloat16 ones, while the frozen
    4-bit base still computes in the fast float16 compute dtype. A no-op off
    GPU or when four_bit is disabled, so the CPU smoke profile is unaffected.
    """
    import torch

    if not (four_bit and torch.cuda.is_available()):
        return model
    from peft import prepare_model_for_kbit_training

    return prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


def cast_trainable_to_fp32(model: Any) -> Any:
    """Upcast already-trainable parameters (an attached adapter) to float32.

    The DPO stage loads the SFT adapter as trainable rather than building a
    fresh one, so it cannot use ``prepare_quantised_for_training`` (that would
    refreeze the adapter). Casting the trainable params to float32 gives the
    float16 gradient scaler something it can unscale, same as for SFT.
    """
    import torch

    if not torch.cuda.is_available():
        return model
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(torch.float32)
    return model


def precision_flags() -> dict[str, bool]:
    """fp16 on pre-Ampere GPUs (the free T4 included), bf16 where native."""
    import torch

    from opening_theory_engine.modelling.hf import native_bf16_supported

    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}
    if native_bf16_supported():
        return {"fp16": False, "bf16": True}
    return {"fp16": True, "bf16": False}


def resolve_report_to(config: WandbConfig, run_name: str) -> str:
    """Decide whether this run reports to W&B, and set its environment up."""
    if config.mode == "disabled":
        return "none"
    mode = config.mode
    if mode == "auto":
        has_key = bool(os.environ.get("WANDB_API_KEY")) or _netrc_has_wandb()
        mode = "online" if has_key else "offline"
        _LOG.info("wandb mode resolved to %s", mode)
    os.environ.setdefault("WANDB_PROJECT", config.project)
    os.environ["WANDB_MODE"] = mode
    os.environ.setdefault("WANDB_RUN_NAME", run_name)
    return "wandb"


def _netrc_has_wandb() -> bool:
    import netrc

    try:
        credentials = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError):
        return False
    return credentials.authenticators("api.wandb.ai") is not None
