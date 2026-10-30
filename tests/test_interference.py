"""Slender-body interference of deflected and rolling fins (interference.py) and its use in aero.py."""

import math

import numpy as np
import pytest

from rocket_dt.builder import build_rocket
from rocket_dt.config import ConfigError, load_config, load_raw, parse_config
from rocket_dt.interference import k_wb_alpha, panel_interference


@pytest.mark.parametrize("tau", [0.2, 0.3, 0.484, 0.6])
def test_pitch_deflection_obeys_reciprocity(tau):
    """k_W(B) + k_B(W) = K_W(B) (closed form) in slender-body theory."""
    p = panel_interference(tau, 1.0 - tau, 4)
    k_wb = p.load_matrix[0, 0] - p.load_matrix[0, 2]  # a pitch pair: opposite panels deflected the same way
    assert k_wb + p.carryover == pytest.approx(k_wb_alpha(tau), rel=2e-3)


def test_no_body_limit_is_the_isolated_wing():
    p = panel_interference(1e-4, 1.0, 2)
    assert p.load_matrix[0, 0] - p.load_matrix[0, 1] == pytest.approx(1.0, rel=2e-3)
    assert p.carryover == pytest.approx(0.0, abs=1e-3)


def test_canard_values_match_the_conformal_map_solution():
    """Independent solution of the cruciform by conformal mapping (zeta -> zeta^2 -> Joukowski): tau = 0.484."""
    p = panel_interference(0.06, 0.064, 4)
    assert p.load_matrix[0, 0] - p.load_matrix[0, 2] == pytest.approx(0.9385, rel=2e-3)  # k_W(B)
    assert p.carryover == pytest.approx(0.4951, rel=2e-3)  # k_B(W)
    assert p.load_matrix[0].sum() == pytest.approx(0.6989, rel=3e-3)  # all four deflected (roll)
    assert p.roll_damping == pytest.approx(0.7143, rel=3e-3)
    assert np.allclose(p.load_matrix, p.load_matrix.T)


@pytest.fixture(scope="module")
def rocket():
    from pathlib import Path
    return build_rocket(load_config(Path(__file__).resolve().parents[1] / "configs/baseline.yaml"))


def _eval(rk, delta, omega=(0.0, 0.0, 0.0), V=100.0):
    return rk.aero.evaluate(np.array([V, 0.0, 0.0]), np.asarray(omega, float), 0.0, 0.0, np.radians(delta),
                            1.225, 340.3, 1.46e-5, rk.mass.at(0.0).cg_station, diagnostics=True, vortex=False)


def test_canard_roll_moment_uses_the_slender_body_roll_factor(rocket):
    c = next(f for f in rocket.aero.fins if f.is_canard)
    q, d = 0.5 * 1.225 * 100.0 ** 2, math.radians(0.5)
    # isolated-fin roll moment of four canards (linear part of the strip model)
    iso = 4 * q * rocket.aero.ref_area * c.cn_alpha_single(math.sqrt(1 - (100 / 340.3) ** 2)) * d \
        * float(c._w @ c._rho)
    roll = _eval(rocket, [0.5] * 4).diag["canard_roll_moment"]
    assert roll / iso == pytest.approx(c.deflection_matrix[0].sum(), rel=0.01)
    assert roll / iso < 0.75  # K_T (1.48) would double it


def test_body_carryover_adds_pitch_authority_but_no_roll(root):
    cfg = load_config(root / "configs/baseline.yaml")
    out = {}
    for share in (0.0, 1.0):
        cfg.aero.canard_body_carryover = share
        rk = build_rocket(cfg)
        r = _eval(rk, [1.0, 0.0, -1.0, 0.0])
        out[share] = (r.force[2], _eval(rk, [1.0] * 4).diag["canard_roll_moment"])
    c = next(f for f in rk.aero.fins if f.is_canard)
    k_wb = c.deflection_matrix[0, 0] - c.deflection_matrix[0, 2]
    assert out[1.0][0] / out[0.0][0] == pytest.approx((k_wb + 0.4951) / k_wb, rel=0.01)
    assert out[1.0][1] == pytest.approx(out[0.0][1])
    raw = load_raw(root / "configs/baseline.yaml")
    raw["aero"]["canard_body_carryover"] = 1.5
    with pytest.raises(ConfigError, match="canard_body_carryover"):
        parse_config(raw, root / "configs")
