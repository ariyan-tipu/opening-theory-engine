# Contributing

This is a personal research project, but issues and pull requests are
welcome. Here is how to get productive quickly.

## Development setup

```bash
git clone https://github.com/ariyan-tipu/opening-theory-engine.git
cd opening-theory-engine
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

The training extra (`pip install -e ".[dev,train]"`) is only needed for the
SFT, generation, DPO and evaluation stages; everything else, including the
stub-backed API, runs without torch installed.

## Checks

Everything CI runs, you can run locally:

```bash
ruff check src tests
ruff format --check src tests
pytest
```

Two opt-in test groups:

- `pytest -m engine` needs a Stockfish binary (`ote engine fetch`, then set
  `STOCKFISH_PATH`). CI installs one with apt.
- `OTE_RUN_TRAIN_TESTS=1 pytest -m train` downloads a small model and runs a
  few real optimiser steps on CPU. Expect several minutes.

## Conventions

- Formatting and linting are whatever `ruff` says; the configuration lives
  in `pyproject.toml`. No other style rules.
- Prose (docs, comments, docstrings) uses British English. Code identifiers
  follow the ecosystem's spelling where one exists, so a Hugging Face
  `tokenizer` stays a `tokenizer`.
- Type hints everywhere, dataclasses for records, classes only where there
  is state worth owning (an engine process, a cache, a loaded model).
- If you touch the bands, the claim lexicon or the templates, run the whole
  suite: the round-trip invariant in `test_templates.py` is what keeps the
  training data and the verifier telling the same story.

## Commits and pull requests

Small, focused commits with imperative subject lines. A pull request should
say what changed and why, and CI must be green before review.
