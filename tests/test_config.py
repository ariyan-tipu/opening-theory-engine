"""Config loading: the shipped profiles must always validate."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from opening_theory_engine.config import ProjectConfig, SplitsConfig, load_config

REPO_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("name", ["laptop.yaml", "colab-t4.yaml"])
def test_shipped_profiles_validate(name):
    config = load_config(REPO_ROOT / "configs" / name)
    assert isinstance(config, ProjectConfig)
    assert config.data.min_elo < config.data.max_elo
    assert len(config.generation.temperatures) >= 2


def test_laptop_profile_stays_light():
    config = load_config(REPO_ROOT / "configs" / "laptop.yaml")
    assert config.engine.threads == 1, "laptop profile must not grab extra cores"
    assert config.data.max_unique_positions <= 500
    assert not config.model.four_bit, "no CUDA on the laptop profile"


def test_splits_must_sum_to_one():
    with pytest.raises(ValidationError):
        SplitsConfig(train=0.5, align=0.4, eval=0.2)


def test_bad_elo_band_rejected(tmp_path):
    yaml_text = (REPO_ROOT / "configs" / "laptop.yaml").read_text(encoding="utf-8")
    broken = yaml_text.replace("min_elo: 1800", "min_elo: 2600")
    path = tmp_path / "broken.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)
