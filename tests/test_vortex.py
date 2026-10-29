"""Canard tip vortices acting on the fin can (per canard / fin pair)."""

import copy
import math

import numpy as np
import pytest

from rocket_dt.builder import build_rocket
from rocket_dt.config import ConfigError, load_config, parse_config

RHO, A_SOUND, NU = 1.225, 340.3, 1.46e-5
V = 100.0


def _rocket(root, **vortex):
    cfg = load_config(root / "configs/baseline.yaml")
    for k, v in vortex.items():
        setattr(cfg.aero.canard_vortex, k, v)
    return build_rocket(cfg)


@pytest.fixture(scope="module")
def rocket():
    from pathlib import Path
    return _rocket(Path(__file__).resolve().parents[1])


def _eval(rk, delta, fincan_angle=0.0, v_b=(V, 0.0, 0.0), vortex=True):
    return rk.aero.evaluate(np.asarray(v_b, float), np.zeros(3), 0.0, fincan_angle, np.radians(delta),
                            RHO, A_SOUND, NU, rk.mass.at(0.0).cg_station, diagnostics=True, vortex=vortex)


def _induced(rk, delta, fincan_angle=0.0, v_b=(V, 0.0, 0.0)):
    """Force and moment added by the vortices."""
    on, off = _eval(rk, delta, fincan_angle, v_b), _eval(rk, delta, fincan_angle, v_b, vortex=False)
    return on.force - off.force, on.moment - off.moment, on


def test_no_deflection_no_incidence_no_vortex_effect(rocket):
    df, dm, on = _induced(rocket, [0, 0, 0, 0], fincan_angle=0.3)
    assert np.allclose(df, 0, atol=1e-12) and np.allclose(dm, 0, atol=1e-12)
    assert on.diag["vortex_roll_moment"] == 0.0


def test_vortex_roll_opposes_canard_roll_and_depends_on_fincan_angle(rocket):
    ang = np.radians([0.0, 22.5, 45.0, 67.5, 90.0])
    m = rocket.aero.vortex_roll_map(0.3, ang, np.radians([1.0] * 4))
    ratio = m["induced"] / m["canard"]
    # roll reversal of slender-body size: strongest with the vortices between the fins
    assert -1.2 < ratio[2] < -0.8
    assert -1.0 < ratio[0] < -0.4 and ratio[0] > ratio[2]
    assert ratio[0] == pytest.approx(ratio[4], rel=1e-9)  # 4 fins: period 90 deg
    assert ratio[1] == pytest.approx(ratio[3], rel=1e-9)  # mirror symmetry about 45 deg


def test_vortex_moment_acts_on_fincan(rocket):
    on = _eval(rocket, [1, 1, 1, 1])
    off = _eval(rocket, [1, 1, 1, 1], vortex=False)
    assert on.fincan_roll_moment - off.fincan_roll_moment == pytest.approx(on.diag["vortex_roll_moment"])
    assert on.moment[0] - off.moment[0] == pytest.approx(on.diag["vortex_roll_moment"])


@pytest.mark.parametrize("fincan_deg", [0.0, 17.0, 45.0])
def test_single_canard_effect_rotates_with_the_canard(rocket, fincan_deg):
    """Canard k+1 alone gives the effect of canard k alone rotated by 90 deg (independent actuation)."""
    th = math.radians(fincan_deg)
    df0, dm0, on0 = _induced(rocket, [2, 0, 0, 0], th)
    df1, dm1, on1 = _induced(rocket, [0, 2, 0, 0], th)
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # +90 deg about x
    assert np.linalg.norm(df0[1:]) > 0.05  # a single canard's vortex also pushes the fins sideways
    assert np.allclose(df1, rot @ df0, atol=1e-9)
    assert np.allclose(dm1, rot @ dm0, atol=1e-9)
    assert on1.diag["vortex_roll_moment_each"][1] == pytest.approx(on0.diag["vortex_roll_moment_each"][0])
    assert on0.diag["vortex_roll_moment_each"][1:] == pytest.approx([0, 0, 0], abs=1e-12)


def test_per_canard_contributions_add_up(rocket):
    on = _eval(rocket, [1.0, -0.5, 0.3, 2.0], fincan_angle=0.2)
    each = on.diag["vortex_roll_moment_each"]
    assert np.all(each != 0)
    assert each.sum() == pytest.approx(on.diag["vortex_roll_moment"], rel=0.02)  # nearly linear


def test_strength_factor_scales_and_switches_off(root):
    base = _induced(_rocket(root), [1, 1, 1, 1], 0.4)[1][0]
    assert _induced(_rocket(root, strength_factor=0.5), [1, 1, 1, 1], 0.4)[1][0] == pytest.approx(0.5 * base, rel=0.02)
    assert _induced(_rocket(root, strength_factor=0.0), [1, 1, 1, 1], 0.4)[1][0] == pytest.approx(0.0, abs=1e-12)
    none = _rocket(root, model="none")
    assert _eval(none, [1, 1, 1, 1]).diag["vortex_roll_moment"] == 0.0


def test_downwash_reduces_tail_lift_and_static_margin(rocket, root):
    a = math.radians(2.0)
    df, dm, _ = _induced(rocket, [0, 0, 0, 0], v_b=(V * math.cos(a), 0.0, V * math.sin(a)))
    assert df[2] > 0  # less tail normal force (which acts along -z here)
    assert dm[1] > 0  # less restoring pitch moment
    cp_on = rocket.aero.static_cp(0.3)[1]
    cp_off = _rocket(root, model="none").aero.static_cp(0.3)[1]
    assert 0.0 < cp_off - cp_on < 0.05


def test_vortex_config_is_validated(root):
    from rocket_dt.config import load_raw
    raw = load_raw(root / "configs/baseline.yaml")
    bad = copy.deepcopy(raw)
    bad["aero"]["canard_vortex"]["model"] = "cfd"
    with pytest.raises(ConfigError, match="canard_vortex.model"):
        parse_config(bad, root / "configs")
