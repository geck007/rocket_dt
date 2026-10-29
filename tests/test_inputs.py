"""Parsing, mass properties and configuration (single source of truth)."""

import copy

import numpy as np
import pytest

from rocket_dt.builder import apply_overrides, build_rocket
from rocket_dt.config import ConfigError, load_config, parse_config
from rocket_dt.io.motor import cross_check, load_eng, load_openmotor_csv, load_ric
from rocket_dt.io.ork import OrkError
from rocket_dt.mass import component_mass

# Component masses computed by OpenRocket (CalcMass in the RockSim export of the same design)
OPENROCKET_MASSES_G = {
    "Nose Cone": 411.7367311629955,
    "recovery bay body tube": 527.8880967680014,
    "canards bay body": 246.39903881186427,
    "canards bay body/Trapezoidal Fin Set": 18.939904000000006,
    "avionics bay tube": 461.9020846720006,
    "Transition#1": 343.97780442240156,
    "Transition#2": 59.86130909105765,
}


@pytest.mark.parametrize("path,expected", OPENROCKET_MASSES_G.items())
def test_component_masses_match_openrocket(ork, path, expected):
    assert component_mass(ork.find(path)).mass * 1000 == pytest.approx(expected, rel=2e-4)


def test_positions_follow_openrocket_rules(ork):
    fins = ork.find("motor case transition/Trapezoidal Fin Set")
    tube = ork.find("motor case transition")
    assert fins.x_top + fins.length == pytest.approx(tube.x_top + tube.length - 0.01)
    assert ork.find("Transition#2").x_top == pytest.approx(1.90)


def test_ambiguous_and_unknown_paths_raise(ork):
    with pytest.raises(OrkError, match="ambiguous"):
        ork.find("Transition")
    with pytest.raises(OrkError, match="not found"):
        ork.find("no such part")


def test_motor_files_are_consistent(root):
    eng = load_eng(root / "data/pls work eng file.eng")
    ric = load_ric(root / "data/pls work.ric")
    burn = load_openmotor_csv(root / "data/csv files as asked.csv")
    cross_check(eng, ric, burn)
    assert np.trapezoid(burn.thrust, burn.time) == pytest.approx(4103.2, abs=1)


def test_motor_mismatch_is_detected(root):
    eng = load_eng(root / "data/pls work eng file.eng")
    ric = load_ric(root / "data/pls work.ric")
    burn = load_openmotor_csv(root / "data/csv files as asked.csv")
    eng.propellant_mass *= 1.1
    with pytest.raises(ValueError, match="propellant mass"):
        cross_check(eng, ric, burn)


def test_unknown_config_key_rejected(baseline_cfg):
    raw = copy.deepcopy(baseline_cfg.raw)
    raw["launch"]["rail_lenght"] = 2.0
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(raw, baseline_cfg.source.parent)


def test_override_requires_note_and_valid_field(baseline_cfg):
    raw = copy.deepcopy(baseline_cfg.raw)
    raw["rocket"]["overrides"].append({"component": "Nose Cone", "set": {"length": 0.4}})
    with pytest.raises(ConfigError, match="note"):
        parse_config(raw, baseline_cfg.source.parent)
    raw["rocket"]["overrides"][-1]["note"] = "test"
    raw["rocket"]["overrides"][-1]["set"] = {"lenght": 0.4}
    with pytest.raises(ConfigError, match="cannot set 'lenght'"):
        build_rocket(parse_config(raw, baseline_cfg.source.parent))


def test_override_shifts_downstream_components(baseline_cfg, ork):
    from rocket_dt.io.ork import load_ork, resolve_positions
    raw = copy.deepcopy(baseline_cfg.raw)
    raw["rocket"]["overrides"] = [{"component": "Nose Cone", "set": {"length": 0.46}, "note": "test"}]
    cfg = parse_config(raw, baseline_cfg.source.parent)
    r = load_ork(cfg.rocket.ork)
    apply_overrides(r, cfg, [])
    resolve_positions(r)
    assert r.find("avionics bay tube").x_top == pytest.approx(ork.find("avionics bay tube").x_top + 0.1)


def test_extends_merges_and_keeps_paths(root):
    cfg = load_config(root / "configs/fincan_disturbance_demo.yaml")
    assert cfg.rocket.ork == (root / "data/rocket.ork.zip").resolve()
    assert cfg.simulation.stop_at == "apogee"
    assert cfg.launch.rail_length == 1.0  # inherited
    assert cfg.disturbances.thrust_misalignment_deg == [0.1, 0.0]


def test_validation_design_reproduces_openrocket_mass(root):
    rk = build_rocket(load_config(root / "configs/validation_openrocket.yaml"))
    m0 = rk.mass.at(0.0)
    assert m0.mass == pytest.approx(17.150, abs=0.005)
    assert m0.cg_station == pytest.approx(1.24061, abs=0.002)
    assert any(n.startswith("OVERRIDE") for n in rk.notes)
