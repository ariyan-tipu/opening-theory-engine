# Changelog

## Unreleased

- `POST /explain` now bounds `fen` (120 characters) and `move` (10), and
  rejects empty values. `opening` and `eco` were already bounded; these two,
  the fields a caller actually controls, were not. python-chess rejected
  malformed input anyway, but only once the whole body had been parsed into
  memory. Requests above the limits return 422 rather than being parsed.

## 0.1.0

Initial release.

- Data pipeline: streamed Lichess filtering, position extraction, motif
  detection, resume-safe Stockfish annotation, Pandas dataset build with
  game-level splits and a generated dataset card.
- Training: QLoRA supervised fine-tuning and DPO via transformers, peft and
  trl, with Weights & Biases tracking wired into both stages.
- Alignment: deterministic claim parser and verifier, JAX jit/vmap batch
  scorer, preference pairs built from engine reference checks.
- Agent loop: explainer proposes, verifier checks, one bounded regeneration
  on failed verification.
- Serving: FastAPI app with stub and real backends, full verification
  evidence in every response, Dockerfile with a pinned engine.
- Evaluation: baseline versus SFT versus DPO on a held-out split, factual
  metrics plus fluency proxies.
- CI: lint, tests on two Python versions with a real engine, a CPU training
  smoke test, and a Docker build that boots the container and hits the API.
- Executed the full pipeline on a free Colab T4 and committed the measured
  results: an 8,993-position dataset, a 200-step QLoRA SFT run, 180
  engine-verified preference pairs, DPO and a held-out evaluation of all
  three systems (see `results/` and the README table).
- Hardened training for real hardware along the way: float16 on Turing GPUs,
  no mixed-precision scaler on the 4-bit path, a CPU-pinned JAX scorer,
  resume-safe checkpointing, and free-tier-sized profiles.
- Published the trained LoRA adapters as downloadable assets on the v0.1.0
  release.
