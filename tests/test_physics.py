"""Aerodynamic sign conventions and rigid-body / rotor dynamics invariants."""

import math

import numpy as np
import pytest

from rocket_dt import quaternion as Q
from rocket_dt.aero import AeroResult
from rocket_dt.builder import build_rocket, build_simulation
from rocket_dt.config import load_config

RHO, A_SOUND, NU = 1.225, 340.3, 1.46e-5


@pytest.fixture(scope="module")
def rocket():
    from pathlib import Path
    return build_rocket(load_config(Path(__file__).resolve().parents[1] / "configs/baseline.yaml"))


def _eval(rk, v_b, omega=(0, 0, 0), fincan_rate=0.0, delta=None):
    return rk.aero.evaluate(np.asarray(v_b, float), np.asarray(omega, float), fincan_rate, 0.0, delta,
                            RHO, A_SOUND, NU, rk.mass.at(0.0).cg_station)


def test_small_alpha_normal_force_matches_barrowman(rocket):
    V, a = 100.0, math.radians(0.5)
    res = _eval(rocket, [V * math.cos(a), 0.0, V * math.sin(a)])
    q = 0.5 * RHO * V * V
    cn_alpha, cp = rocket.aero.static_cp(V / A_SOUND)
    assert -res.force[2] / (q * rocket.aero.ref_area * a) == pytest.approx(cn_alpha, rel=0.02)
    # restoring: crossflow +z must pitch the nose towards +z (negative moment about y)
    assert res.moment[1] < 0
    cg = rocket.mass.at(0.0).cg_station
    # M_y = -(x_cg - x_cp) F_z in body x (= -station)  =>  station_cp = station_cg + M_y / F_z
    assert cg + res.moment[1] / res.force[2] == pytest.approx(cp, abs=0.01)


def test_canard_deflection_gives_positive_roll(rocket):
    n = rocket.aero.n_canards
    res = _eval(rocket, [100.0, 0, 0], delta=np.full(n, math.radians(2.0)))
    assert res.moment[0] > 0
    assert abs(res.force[1]) < 1e-9 and abs(res.force[2]) < 1e-9  # pure roll couple


def test_roll_is_damped_by_body_and_fincan_fins(rocket):
    body = _eval(rocket, [100.0, 0, 0], omega=(5.0, 0, 0))
    assert body.moment[0] < 0
    can = _eval(rocket, [100.0, 0, 0], fincan_rate=5.0)
    assert can.fincan_roll_moment < 0 and can.moment[0] == pytest.approx(can.fincan_roll_moment)


def test_pitch_rate_is_damped(rocket):
    res = _eval(rocket, [100.0, 0, 0], omega=(0, 1.0, 0))
    assert res.moment[1] < 0


def _torque_free_sim(free: bool, bearing_viscous: float):
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/baseline.yaml")
    cfg.rocket.fin_can.enabled = free
    cfg.rocket.fin_can.bearing.coulomb_torque = 0.0
    cfg.rocket.fin_can.bearing.viscous_coeff = bearing_viscous
    sim = build_simulation(cfg)
    sim.rocket.aero.evaluate = lambda *a, **k: AeroResult(np.zeros(3), np.zeros(3), 0.0)
    return sim


def _angular_momentum(sim, t, y):
    ms = sim.rocket.mass.at(t)
    h_b = ms.inertia @ y[10:13] + ms.fincan_roll_inertia * y[14] * np.array([1.0, 0, 0])
    return Q.to_matrix(y[6:10]) @ h_b


def _spin(sim, y0, t0=10.0, dt=1e-3, n=3000):
    y, t = y0.copy(), t0
    for _ in range(n):
        y = sim._rk4(sim.deriv_flight, t, y, dt, np.zeros(sim.rocket.aero.n_canards))
        y[6:10] = Q.normalize(y[6:10])
        t += dt
    return y, t


def test_free_fincan_conserves_angular_momentum_and_energy():
    sim = _torque_free_sim(free=True, bearing_viscous=0.0)
    y0 = np.zeros(15)
    y0[6:10] = Q.from_axis_angle([0, 1, 0], 0.3)
    y0[10:13] = [1.0, 0.4, -0.3]
    y0[14] = 40.0
    t0 = 10.0  # after burnout: constant mass properties
    y, t = _spin(sim, y0, t0)
    assert np.allclose(_angular_momentum(sim, t, y), _angular_momentum(sim, t0, y0), rtol=1e-8, atol=1e-10)

    def energy(yy):
        ms = sim.rocket.mass.at(t0)
        w = yy[10:13]
        If = ms.fincan_roll_inertia
        # rotor: total = body-frame part with locked fin can + extra spin energy of the can
        return 0.5 * w @ ms.inertia @ w + If * w[0] * yy[14] + 0.5 * If * yy[14] ** 2
    assert energy(y) == pytest.approx(energy(y0), rel=1e-8)
    # frictionless bearing: fin can's inertial spin about the axis is constant (no axial torque)
    assert y[10] + y[14] == pytest.approx(y0[10] + y0[14], rel=1e-3)


def test_bearing_friction_brings_fincan_to_body_rate():
    sim = _torque_free_sim(free=True, bearing_viscous=0.05)
    y0 = np.zeros(15)
    y0[6] = 1.0
    y0[14] = 30.0
    y, t = _spin(sim, y0, n=5000)
    assert abs(y[14]) < 0.01 * 30.0
    # momentum about the axis is shared, not lost
    assert np.allclose(_angular_momentum(sim, t, y), _angular_momentum(sim, 10.0, y0), rtol=1e-8)


@pytest.mark.slow
def test_wind_direction_rotates_trajectory(root):
    """Rotating the wind by 90 deg must rotate the ascent by 90 deg (frame / symmetry check).

    With a vertical rail, a 4-fin rocket and no misalignments, the flight stays in the
    plane of the wind; the cross-wind coordinate is zero to round-off."""
    cfg = load_config(root / "configs/baseline.yaml")
    cfg.simulation.stop_at = "apogee"
    out = {}
    for frm in (90.0, 0.0):  # from east, from north
        cfg.environment.wind.direction_from_deg = frm
        d = build_simulation(cfg).run().data
        out[frm] = (d["x"][-1], d["y"][-1], d["z"][-1])
    (xe, ye, ze), (xn, yn, zn) = out[90.0], out[0.0]
    assert xe > 10 and abs(ye) < 1e-6  # weathercocks into the wind: east, nothing north
    assert yn == pytest.approx(xe, rel=1e-6) and abs(xn) < 1e-6
    assert zn == pytest.approx(ze, rel=1e-9)


@pytest.mark.slow
def test_validation_against_openrocket(root):
    from rocket_dt.analysis.openrocket import load_openrocket_csv
    cfg = load_config(root / "configs/validation_openrocket.yaml")
    cfg.simulation.stop_at = "apogee"
    sim = build_simulation(cfg)
    r = sim.run()
    orc, orev = load_openrocket_csv(root / "data/all.csv")
    assert r.data["z"].max() - r.data["z"][0] == pytest.approx(orc["Altitude"].max(), rel=0.01)
    assert r.events["apogee"] == pytest.approx(orev["apogee"], rel=0.01)
    assert r.data["speed"].max() == pytest.approx(orc["Total velocity"].max(), rel=0.01)
