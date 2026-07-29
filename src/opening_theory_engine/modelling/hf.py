"""Hugging Face model loading and batched chat generation.

Imports of torch and transformers happen inside functions on purpose: the
base install (data pipeline, verifier, stub-backed API) works without the
``train`` extra, and only the code paths that genuinely need a model pay
for the import.

Quantisation policy: 4-bit NF4 via bitsandbytes when a CUDA device is
present and the config asks for it; float16 or bfloat16 on GPU otherwise;
float32 on CPU, where bitsandbytes is not available and small models run
fine unquantised.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from opening_theory_engine.utils import get_logger

_LOG = get_logger(__name__)

Conversation = list[dict[str, str]]


@dataclass(frozen=True)
class GenerationSettings:
    """Decoding parameters for one generation pass."""

    max_new_tokens: int = 200
    temperature: float = 0.0
    top_p: float = 0.95
    max_input_tokens: int = 1024

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0


def native_bf16_supported() -> bool:
    """True only when the GPU implements bfloat16 in hardware (Ampere, SM 8.0+).

    ``torch.cuda.is_bf16_supported()`` also returns True on older cards such as
    the Turing T4, where bfloat16 is emulated in software and runs markedly
    slower than float16. Gating on the real compute capability keeps the T4 on
    the fast float16 path, which is what the training profiles expect.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def load_tokenizer(base_id: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    base_id: str,
    *,
    four_bit: bool = False,
    adapter_path: str | None = None,
    train_mode: bool = False,
) -> Any:
    """Load the base model, optionally quantised, optionally with an adapter."""
    import torch
    from transformers import AutoModelForCausalLM

    cuda = torch.cuda.is_available()
    kwargs: dict[str, Any] = {}
    if four_bit and cuda:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if native_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["device_map"] = "auto"
        # Pin the non-quantised layers (norms, embeddings, LoRA adapters) to the
        # compute dtype. Without this they inherit the model's native dtype,
        # which is bfloat16 for Qwen, and float16 training then fails when the
        # gradient scaler tries to unscale bfloat16 gradients on a T4.
        kwargs["dtype"] = compute_dtype
        _LOG.info("loading %s in 4-bit NF4 (compute dtype %s)", base_id, compute_dtype)
    elif cuda:
        kwargs["dtype"] = torch.bfloat16 if native_bf16_supported() else torch.float16
        kwargs["device_map"] = "auto"
        _LOG.info("loading %s on GPU in %s", base_id, kwargs["dtype"])
    else:
        kwargs["dtype"] = torch.float32
        if four_bit:
            _LOG.info("no CUDA device: ignoring four_bit and loading %s in float32", base_id)
        else:
            _LOG.info("loading %s on CPU in float32", base_id)

    model = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
    if adapter_path:
        from peft import PeftModel

        _LOG.info("attaching adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=train_mode)
    if not train_mode:
        model.eval()
    return model


def generate_texts(
    model: Any,
    tokenizer: Any,
    conversations: Sequence[Conversation],
    settings: GenerationSettings,
    batch_size: int = 4,
) -> list[str]:
    """Apply the chat template and generate a completion for each conversation."""
    import torch

    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    for start in range(0, len(conversations), batch_size):
        batch = conversations[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            for conversation in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.max_input_tokens,
        ).to(device)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": settings.do_sample,
        }
        if settings.do_sample:
            generate_kwargs["temperature"] = settings.temperature
            generate_kwargs["top_p"] = settings.top_p
        with torch.no_grad():
            generated = model.generate(**encoded, **generate_kwargs)
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        outputs.extend(
            tokenizer.decode(tokens, skip_special_tokens=True).strip() for tokens in new_tokens
        )
    return outputs
