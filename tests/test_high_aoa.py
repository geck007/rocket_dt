"""Aerodynamics at high angle of attack and in reverse flow (post-apogee)."""

import math

import numpy as np
import pytest

from rocket_dt.aero import _axial_multiplier

V, RHO = 30.0, 1.225


@pytest.fixture(scope="module")
def rocket():
    from pathlib import Path

    from rocket_dt.builder import build_rocket
    from rocket_dt.config import load_config
    return build_rocket(load_config(Path(__file__).resolve().parents[1] / "configs/baseline.yaml"))


def _eval(rk, alpha_deg, roll_deg=0.0, cg=1.2):
    a, ph = math.radians(alpha_deg), math.radians(roll_deg)
    v = V * np.array([math.cos(a), -math.sin(a) * math.sin(ph), math.sin(a) * math.cos(ph)])
    return rk.aero.evaluate(v, np.zeros(3), 0.0, 0.0, np.zeros(4), RHO, 340.3, 1.46e-5, cg, diagnostics=True,
                            vortex=False)


def test_broadside_force_is_body_crossflow_plus_flat_plate_fins(rocket):
    """At 90 deg: eta(L/D) C_dc on the body planform, C_dc,fin on the two broadside fins of each set at
    mid-chord, no attached-flow lift and no axial force."""
    a = rocket.aero
    assert a.crossflow_k == pytest.approx(1.2 * 0.722, rel=0.01)  # eta = 0.72 at L/D = 15.2
    qa = 0.5 * RHO * V * V * a.ref_area
    cg = 1.2
    r = _eval(rocket, 90.0, cg=cg)
    body = a.crossflow_k * a.slice_area / a.ref_area
    fins = [2 * 1.2 * f.planform.area / a.ref_area for f in a.fins]
    assert -r.force[2] / qa == pytest.approx(body.sum() + sum(fins), rel=1e-6)
    assert r.force[0] == pytest.approx(0.0, abs=1e-9)
    # pitch moment about the CG (body x = -station): M_y = sum (cg - x) F_z
    m = -(body @ (cg - a.slice_station)) - sum(c * (cg - f.station_mid) for c, f in zip(fins, a.fins))
    assert r.moment[1] / qa == pytest.approx(-m, rel=1e-6)


def test_spanwise_flow_does_not_load_the_fins(rocket):
    """Independence principle: fins at 45 deg to the crossflow carry cos^2(45 deg) of the broadside load."""
    a = rocket.aero
    body = a.crossflow_k * a.slice_area.sum() / a.ref_area
    qa = 0.5 * RHO * V * V * a.ref_area
    fins0 = np.linalg.norm(_eval(rocket, 90.0, 0.0).force[1:]) / qa - body
    fins45 = np.linalg.norm(_eval(rocket, 90.0, 45.0).force[1:]) / qa - body
    assert fins45 / fins0 == pytest.approx(1 / math.sqrt(2), rel=1e-6)


def test_normal_force_rises_smoothly_to_broadside(rocket):
    cn = [np.linalg.norm(_eval(rocket, a).force[1:]) for a in range(0, 91, 5)]
    assert np.all(np.diff(cn[:15]) > 0)  # no stall below 70 deg
    assert min(cn[12:]) > 0.85 * max(cn)  # and no collapse towards 90 deg


def test_tail_first_drag_has_the_base_facing_the_flow(rocket):
    a = rocket.aero
    fwd, rev = a.drag_breakdown(0.1, 4e6), a.drag_breakdown(0.1, 4e6, reverse=True)
    assert fwd["friction"] == rev["friction"]
    assert rev["base"] == 0.0  # the nose is a pointed afterbody
    base_area = a._press[False]["base"]
    assert rev["pressure"] - fwd["pressure"] == pytest.approx(0.85 * (1 + 0.01 / 4) * base_area, rel=0.05)
    qa = 0.5 * RHO * V * V * a.ref_area
    r = _eval(rocket, 175.0)
    assert r.force[0] / qa == pytest.approx(_axial_multiplier(math.radians(5.0)) * r.diag["cd"], rel=1e-9)
    assert r.diag["cd"] > 2 * _eval(rocket, 5.0).diag["cd"]
