"""Small shared helpers: hashing, JSONL files, seeding and logging."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_LOGGING_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Project-wide logger with one plain, readable format."""
    global _LOGGING_CONFIGURED
    if not _LOGGING_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOGGING_CONFIGURED = True
    return logging.getLogger(name)


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a small JSON document, creating parent directories as needed."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def set_seed(seed: int) -> None:
    """Seed every random source that might be present."""
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
