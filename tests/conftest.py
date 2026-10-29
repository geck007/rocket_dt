from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def root():
    return ROOT


@pytest.fixture
def baseline_cfg():
    from rocket_dt.config import load_config
    return load_config(ROOT / "configs/baseline.yaml")


@pytest.fixture
def ork():
    from rocket_dt.io.ork import load_ork, resolve_positions
    r = load_ork(ROOT / "data/rocket.ork.zip")
    resolve_positions(r)
    return r
