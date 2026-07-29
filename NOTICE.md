# Third-party notices

This project is licensed under the GNU General Public License v3.0 or later
(see [LICENSE](LICENSE)). It builds on the following third-party work.

## python-chess

[python-chess](https://github.com/niklasf/python-chess) (the `chess` package)
is licensed under GPL-3.0-or-later. Because this project imports it, the
project as a whole is distributed under a GPL-compatible licence. This is the
main reason the repository is GPL rather than MIT.

## Stockfish

[Stockfish](https://stockfishchess.org) is licensed under GPL-3.0. The engine
binary is **not** bundled in this repository or in any built artefact of it.
The `ote engine fetch` command downloads an official release binary from the
Stockfish GitHub releases page, pinned by version and checksum, onto your own
machine. Stockfish is developed by the Stockfish developers; see their
AUTHORS file.

## Lichess open database

Game data comes from the [Lichess open database](https://database.lichess.org),
which is released under the Creative Commons CC0 1.0 dedication (public
domain). A small fixture file of games from the January 2013 export is bundled
under `tests/fixtures/` so the test suite runs without network access.

## Base models

- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) is
  released under the **Qwen Research Licence Agreement**, which permits
  research and non-commercial use. The weights are not distributed with this
  repository; they are downloaded from Hugging Face when you run the training
  or serving stages. Any LoRA adapters trained from it inherit that licence's
  restrictions. This repository is a personal research and portfolio project,
  which is within the licence's scope. If you want a fully Apache-2.0 stack,
  set `model.base_id` to `Qwen/Qwen2.5-1.5B-Instruct` in your config.
- [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
  (used only for the laptop smoke profile and CI) is Apache-2.0.

## Everything else

Library dependencies (PyTorch, Hugging Face transformers, peft, trl, JAX,
pandas, FastAPI and friends) are used under their own permissive licences as
declared in `pyproject.toml`.

Copyright (c) 2026 Ariyan Mohammed Tipu.
