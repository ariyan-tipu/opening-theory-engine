"""CLI surface: parsing and the light commands."""

import pytest

from opening_theory_engine.cli import build_parser, main


def test_every_stage_parses():
    parser = build_parser()
    for argv in (
        ["engine", "fetch"],
        ["engine", "check", "--config", "x.yaml"],
        ["data", "build", "--config", "x.yaml", "--stub-engine", "--quiet"],
        ["train", "sft", "--config", "x.yaml"],
        ["train", "dpo", "--config", "x.yaml"],
        ["generate", "candidates", "--config", "x.yaml", "--adapter", "some/dir"],
        ["align", "pairs", "--config", "x.yaml"],
        ["eval", "run", "--config", "x.yaml", "--systems", "baseline", "sft"],
        ["serve", "--port", "9999"],
    ):
        args = parser.parse_args(argv)
        assert args.command == argv[0]


def test_version_flag():
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_missing_subcommand_errors():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train"])


def test_data_build_runs_offline(tmp_path, fixture_pgn, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    import yaml

    from conftest import make_test_config

    profile = make_test_config(tmp_path, pgn_url=str(fixture_pgn))
    config.write_text(yaml.safe_dump(profile.model_dump(mode="json")), encoding="utf-8")
    exit_code = main(["data", "build", "--config", str(config), "--stub-engine", "--quiet"])
    assert exit_code == 0
    assert (tmp_path / "data" / "processed" / "sft_train.jsonl").is_file()
