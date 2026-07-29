# Design notes

The decisions behind the project, with the reasoning I would give if you
asked me about them across a table. Written down mostly so the trade-offs
stay visible.

## Why Qwen2.5-3B-Instruct

The task is narrow and structured: four labelled lines, a controlled
vocabulary for verdicts and evaluations, short free text. That is a job for
a small model with decent instruction-following, not a frontier model.

The VRAM arithmetic on the free T4 (16 GB):

- 3B parameters in 4-bit NF4 is roughly 1.9 GB of weights.
- LoRA at rank 16 over the attention and MLP projections adds only tens of
  millions of trainable parameters, so adapter weights, gradients and Adam
  state stay small.
- The real variable is activations. At sequence length 768 and micro-batch 4
  with gradient checkpointing, peak sits comfortably under the T4's 16 GB
  with headroom for fragmentation.

A 7B model in 4-bit also fits, but roughly halves training and generation
throughput on this GPU, and generation is the bottleneck of the alignment
stage (thousands of sampled candidates). For a task this constrained, the
quality gap between 3B and 7B matters less than iteration speed on a free
tier, so 3B wins. The 1.5B variant is wired into the configs as a fallback;
it is also the option to pick if the Qwen Research Licence on the 3B model
is a problem, since 1.5B is Apache-2.0.

## Why fp16 rather than bf16

The T4 is a Turing card and has no bfloat16 units. The code asks PyTorch at
runtime (`torch.cuda.is_bf16_supported()`) and sets fp16 on the T4, bf16 on
Ampere or newer, and float32 on CPU. QLoRA's compute dtype follows the same
switch. Nothing is hard-coded, so the same config trains on an A100 without
edits.

## Why the SFT targets are templated from engine facts

The supervision problem: I need thousands of (position, explanation) pairs
where the explanation is factually right, and I am not paying a larger model
to write them or trusting it not to hallucinate.

The answer here is distant supervision that audits itself. Every
target is rendered from verified facts: the verdict comes from banding the
measured centipawn loss, the evaluation phrase from banding the engine
score, the quoted line from the engine's continuation, and each free-text
sentence is only emitted when its underlying flag (a motif, a settled
material swing) is actually true. The test suite enforces the invariant that
a rendered target, parsed back and verified, contains zero incorrect claims.

The cost is a ceiling on prose variety, and I accept it: SFT teaches
format and grounding, then the alignment stage optimises factual accuracy
under sampling, which is where the model's own failure modes get corrected.
Fluency is reported through transparent proxies, not overclaimed.

## Why DPO, and why not PPO or an LLM judge

The preference signal is deterministic: parse claims, check against engine
facts, score. Given that, DPO is the right tool. It is offline, stable, and
needs no reward model, because the "reward model" is a verifier that already
exists and is unit-tested. PPO would add an actor-critic loop, KL
controllers and a lot of variance to learn a signal I can compute exactly.

The same argument rules out an LLM judge for evaluation: a judge can
hallucinate about hallucinations. A deterministic verifier is free,
reproducible, and its coverage limits are documented rather than hidden.

## Guards against gaming the scorer

Optimising against any fixed scorer invites reward hacking, so the pair
builder is deliberately conservative:

- The chosen side must be well formed **and** carry zero incorrect claims.
  A candidate that says nothing checkable cannot outrank one that says true
  things, because correct claims add score.
- Pairs need a minimum score margin; near-ties are noise and are dropped.
- The scorer only feeds training through pairs. The reported results come
  from the held-out split, which no stage trains on, and game-level split
  assignment stops near-duplicate positions leaking across splits.
- The verifier is versioned in the repository; if its rules change, the
  numbers regenerate.

## Why JAX lives only in the scorer

The scoring step is pure array maths over a batch of candidates: band
distances, line-match ratios, claim counts, a weighted sum. That is exactly
the functional, jit-plus-vmap shape JAX is designed for, so the per-candidate
rule is written once as scalar code and vmap lifts it across the batch.

The caveat: at this project's scale NumPy would cope fine, and the
tests use a NumPy mirror as the oracle. JAX is here because the pattern is
right, the intent is clearer, and the identical function runs on an
accelerator when the batch grows. It is deliberately not forced into the
training loop, where the Hugging Face and PyTorch stack is the correct tool.

## The agentic loop, kept small on purpose

Two components with one job each. The explainer proposes; the deterministic
verifier checks every claim against the engine and either passes the
response, triggers one resample at a different temperature, or returns the
least-wrong attempt with the full evidence attached. There is no planner, no
supervisor, no tool-calling graph, because the task does not need one, and
every part of this loop can be explained and defended in detail.

## Serving choices

FastAPI wrapping plain Hugging Face `generate()`, one engine process guarded
by a lock, requests queueing rather than forking engines. At portfolio scale
this is the right amount of machinery. The considered next step for real
traffic is vLLM (paged attention, continuous batching) rather than threads
in front of `generate()`, and the response schema would not change.

## Where the compute came from

The training recipe targets one 16 GB T4. I ran it on Colab's free tier
because the budget for this project was zero; an AWS EC2 `g4dn.xlarge`
carries the same T4 and runs the same commands unchanged, which is the
point: nothing in the pipeline knows it is on Colab. The annotation cache
writing to Google Drive is what makes free-tier session recycling a
non-event.

## Engine settings and evaluation noise

Annotation runs at depth 14 with MultiPV 2 on the hosted profile. Deeper
search would tighten the ground truth but multiplies annotation time, and
opening positions are exactly where shallow-to-medium Stockfish is already
strong. Residual search noise is absorbed by working in bands: verdicts and
evaluations are checked at band level with a one-band tolerance, not at
centipawn precision, and exact-match rates are reported separately.

## Known limitations

- The claim lexicon is deliberately tight. Claims outside it (pawn
  structure judgements, long-term plans) are unverifiable and score
  neutrally rather than being checked.
- The static exchange evaluation plays out cheapest legal recaptures, which
  respects pins but ignores deeper x-ray effects; material claims lean on
  the settled swing along the engine's line instead.
- "Better was X" can only be confirmed against the engine's first choice.
  A legal alternative that is genuinely better than the played move but is
  not the engine's top move is recorded as unverifiable, not correct.
- Serving requests without an opening name give the model slightly
  different context than training, where names were always present. The
  API accepts optional opening and ECO fields for that reason.

## Versions this was verified against

transformers 5.14, trl 1.8, peft 0.19, datasets 5.0, accelerate 1.14,
torch 2.13, jax 0.10, pandas 3.0, python-chess 1.11, Stockfish 18, on
Python 3.13 (Windows) and 3.11/3.13 (CI, Ubuntu). The pyproject ranges pin
the majors these APIs belong to.
