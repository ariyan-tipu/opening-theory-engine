"""Opening theory engine.

A small fine-tuned language model that explains chess opening moves, with
every stage checked against a Stockfish reference: templated supervision
built from real engine analysis, QLoRA fine-tuning, preference alignment
(DPO) driven by a deterministic claim verifier, and a serving API that
verifies what the model says before returning it.
"""

__version__ = "0.1.0"
