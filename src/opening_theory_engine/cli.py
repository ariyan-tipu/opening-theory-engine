"""The ``ote`` command line interface.

One subcommand per pipeline stage, in the order you would run them:

    ote engine fetch                  download the pinned Stockfish build
    ote data build --config ...       games -> annotated positions -> datasets
    ote train sft --config ...        QLoRA supervised fine-tuning
    ote generate candidates ...       sample explanations for alignment
    ote align pairs --config ...      verify, score with JAX, build DPO pairs
    ote train dpo --config ...        preference optimisation
    ote eval run --config ...         held-out comparison of all systems
    ote serve                         run the API (settings via OTE_* env vars)

Training-stack imports happen inside handlers, so commands that do not
need torch work on a machine without the ``train`` extra installed.
"""

from __future__ import annotations

import argparse
import sys

from opening_theory_engine import __version__
from opening_theory_engine.config import load_config
from opening_theory_engine.utils import get_logger

_LOG = get_logger("ote")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ote",
        description="Chess opening explanations, checked against an engine reference.",
    )
    parser.add_argument("--version", action="version", version=f"ote {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    engine = commands.add_parser("engine", help="engine binary management")
    engine_sub = engine.add_subparsers(dest="engine_command", required=True)
    fetch = engine_sub.add_parser("fetch", help="download the pinned Stockfish release")
    fetch.add_argument("--dest", default=None, help="directory to install into")
    check = engine_sub.add_parser("check", help="confirm an engine binary is usable")
    check.add_argument("--config", default="configs/laptop.yaml")

    data = commands.add_parser("data", help="dataset construction")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    build = data_sub.add_parser("build", help="download, filter, annotate and write datasets")
    build.add_argument("--config", default="configs/laptop.yaml")
    build.add_argument("--stub-engine", action="store_true", help="use the test stub (no binary)")
    build.add_argument("--quiet", action="store_true", help="hide progress bars")

    train = commands.add_parser("train", help="training stages")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    sft = train_sub.add_parser("sft", help="QLoRA supervised fine-tuning")
    sft.add_argument("--config", default="configs/laptop.yaml")
    dpo = train_sub.add_parser("dpo", help="DPO on engine-verified preference pairs")
    dpo.add_argument("--config", default="configs/laptop.yaml")

    generate = commands.add_parser("generate", help="sampling stages")
    generate_sub = generate.add_subparsers(dest="generate_command", required=True)
    candidates = generate_sub.add_parser("candidates", help="sample explanations per position")
    candidates.add_argument("--config", default="configs/laptop.yaml")
    candidates.add_argument("--adapter", default=None, help="override the SFT adapter path")

    align = commands.add_parser("align", help="alignment stages")
    align_sub = align.add_subparsers(dest="align_command", required=True)
    pairs = align_sub.add_parser("pairs", help="verify, score and build DPO pairs")
    pairs.add_argument("--config", default="configs/laptop.yaml")

    evaluate = commands.add_parser("eval", help="evaluation")
    evaluate_sub = evaluate.add_subparsers(dest="eval_command", required=True)
    run = evaluate_sub.add_parser("run", help="compare systems on the held-out split")
    run.add_argument("--config", default="configs/laptop.yaml")
    run.add_argument(
        "--systems",
        nargs="+",
        default=None,
        choices=["baseline", "sft", "dpo"],
        help="subset of systems to evaluate",
    )

    serve = commands.add_parser("serve", help="run the API with uvicorn")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {
        "engine": _handle_engine,
        "data": _handle_data,
        "train": _handle_train,
        "generate": _handle_generate,
        "align": _handle_align,
        "eval": _handle_eval,
        "serve": _handle_serve,
    }[args.command]
    return handler(args)


def _handle_engine(args: argparse.Namespace) -> int:
    if args.engine_command == "fetch":
        from pathlib import Path

        from opening_theory_engine.engine.fetch import fetch

        path = fetch(Path(args.dest) if args.dest else None)
        print(f"Stockfish ready at: {path}")
        print("Set STOCKFISH_PATH to this path, or pass engine.path in your config.")
        return 0
    if args.engine_command == "check":
        import chess

        from opening_theory_engine.engine.analysis import StockfishAnalyser

        config = load_config(args.config)
        with StockfishAnalyser(config.engine) as analyser:
            line = analyser.top_lines(chess.Board())[0]
            print(f"Engine: {analyser.name}")
            print(
                f"Start position, depth {config.engine.depth}: "
                f"best {line.moves_san[0]} ({line.cp_white:+d} cp for White)"
            )
        return 0
    return 2


def _handle_data(args: argparse.Namespace) -> int:
    from opening_theory_engine.data.pipeline import run_data_build

    config = load_config(args.config)
    bundle = run_data_build(config, use_stub_engine=args.stub_engine, quiet=args.quiet)
    print(f"Dataset written: {bundle.split_counts} -> {bundle.positions_path.parent}")
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.train_command == "sft":
        from opening_theory_engine.training.sft import run_sft

        adapter = run_sft(config)
    else:
        from opening_theory_engine.training.dpo import run_dpo

        adapter = run_dpo(config)
    print(f"Adapter saved to: {adapter}")
    return 0


def _handle_generate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from opening_theory_engine.training.candidates import generate_candidates

    config = load_config(args.config)
    adapter = Path(args.adapter) if args.adapter else None
    path = generate_candidates(config, adapter_dir=adapter)
    print(f"Candidates written to: {path}")
    return 0


def _handle_align(args: argparse.Namespace) -> int:
    from opening_theory_engine.alignment.pipeline import build_preference_pairs

    config = load_config(args.config)
    path = build_preference_pairs(config)
    print(f"Preference pairs written to: {path}")
    return 0


def _handle_eval(args: argparse.Namespace) -> int:
    from opening_theory_engine.evaluation.run import render_summary_table, run_evaluation

    config = load_config(args.config)
    summary = run_evaluation(config, systems=args.systems)
    print(render_summary_table(summary))
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from opening_theory_engine.serving.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
