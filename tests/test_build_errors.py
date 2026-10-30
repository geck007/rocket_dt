"""Build errors that put the CG or the CP off the axis (``disturbances`` in the config)."""

import math
from pathlib import Path

import numpy as np
import pytest

from rocket_dt.builder import build_rocket
from rocket_dt.config import load_config

ROOT = Path(__file__).resolve().parents[1]
V, RHO = 100.0, 1.225


def _rocket(**dist):
    cfg = load_config(ROOT / "configs/baseline.yaml")
    for k, v in dist.items():
        setattr(cfg.disturbances, k, v)
    return build_rocket(cfg)


@pytest.fixture(scope="module")
def perfect():
    return _rocket()


def _aero(rk, v_b, cg=1.2):
    return rk.aero.evaluate(np.asarray(v_b, float), np.zeros(3), 0.0, 0.0, np.zeros(4), RHO, 340.3, 1.46e-5, cg)


def test_cg_offset_is_the_liftoff_value_and_grows_as_the_propellant_burns(perfect):
    rk = _rocket(cg_offset_mm=[1.0, -0.5])
    m0, m1 = rk.mass.at(0.0), rk.mass.at(rk.motor.burn_time + 1.0)
    assert m0.cg[1:] * 1e3 == pytest.approx([1.0, -0.5], rel=1e-9)
    # the off-axis mass is in the airframe: the vehicle offset scales with 1 / m(t)
    assert m1.cg[1:] * 1e3 == pytest.approx(np.array([1.0, -0.5]) * m0.mass / m1.mass, rel=1e-9)
    # the axial CG and the mass are unchanged; products of inertia appear
    assert m0.cg_station == pytest.approx(perfect.mass.at(0.0).cg_station, abs=1e-12)
    assert abs(m0.inertia[0, 1]) > 0 and abs(perfect.mass.at(0.0).inertia[0, 1]) < 1e-15


def test_inertia_axis_tilt_rotates_the_airframe_inertia(perfect):
    rk = _rocket(inertia_axis_tilt_deg=[0.3, 0.2])
    I, I0 = rk.mass.at(0.0).inertia, perfect.mass.at(0.0).inertia
    assert np.trace(I) == pytest.approx(np.trace(I0), rel=1e-12)  # only the airframe part is rotated
    assert abs(I[0, 2]) > 1e-4 * I[0, 0] and abs(I[0, 1]) > 1e-4 * I[0, 0]
    assert rk.mass.at(0.0).cg == pytest.approx(perfect.mass.at(0.0).cg, abs=1e-15)


def test_misaligned_nose_lifts_at_zero_angle_of_attack(perfect):
    a = math.radians(0.3)  # nose axis rotated about +y: it points towards -z
    rk = _rocket(nose_misalignment_deg=[0.3, 0.0])
    am = rk.aero
    r = _aero(rk, [V, 0.0, 0.0])
    i = int(np.argmax(am.body_cna))  # the nose cone
    expect = 0.5 * RHO * V * V * am.ref_area * am.body_cna[i] * a
    assert r.force[2] == pytest.approx(-expect, rel=0.01)
    assert r.force[1] == pytest.approx(0.0, abs=1e-9)
    cg = 1.2
    assert r.moment[1] == pytest.approx(-(cg - am.body_cp_station[i]) * r.force[2], rel=0.01)
    # the same as the aligned rocket flying with the nose at that angle of attack, nose force only
    r_perfect = _aero(perfect, [V, 0.0, 0.0])
    assert np.allclose(r_perfect.force[1:], 0.0, atol=1e-9)


def test_fincan_imbalance_is_a_rotating_product_of_inertia(perfect):
    rk = _rocket(fincan_cg_offset_mm=[0.5, 0.0], fincan_axis_tilt_deg=[0.0, 0.4])
    ms = rk.mass.at(0.0)
    m, mf, st = ms.mass, rk.fincan_mass, ms.cg_station
    e = 0.5e-3
    Om = 60.0
    # static part: reaction of the bearing m_f e Omega^2 (reduced mass) at the fin can, lever arm about the CG
    arm = st - rk.fincan_station
    static = (1 - mf / m) * mf * e * Om ** 2 * arm
    couple = (rk.fincan_imbalance_couple[1]) * Om ** 2
    shift, M, a_frame = rk.fincan_imbalance(0.0, Om, m, st)
    assert M[0] == 0.0
    assert M[1] == pytest.approx(-couple, rel=1e-9)
    assert M[2] == pytest.approx(static, rel=1e-9)
    assert shift == pytest.approx([mf * e / m, 0.0], rel=1e-9)
    assert a_frame[1:] == pytest.approx([Om ** 2 * mf * e / m, 0.0], rel=1e-9)
    # it turns with the fin can and scales with Omega^2
    shift90, M90, _ = rk.fincan_imbalance(math.pi / 2, Om, m, st)
    assert shift90 == pytest.approx([0.0, mf * e / m], abs=1e-12)
    assert np.linalg.norm(M90) == pytest.approx(np.linalg.norm(M), rel=1e-9)
    assert np.linalg.norm(rk.fincan_imbalance(0.0, 2 * Om, m, st)[1]) == pytest.approx(4 * np.linalg.norm(M))
    # a balanced fin can does nothing
    assert not perfect.fincan_imbalance(0.3, Om, m, st)[1].any()


def test_build_errors_change_the_flight_and_zero_errors_change_nothing():
    from rocket_dt.sim import Simulation

    def apogee_tilt(**dist):
        cfg = load_config(ROOT / "configs/baseline.yaml")
        cfg.simulation.t_max = 6.0
        for k, v in dist.items():
            setattr(cfg.disturbances, k, v)
        d = Simulation(build_rocket(cfg), cfg).run().data
        return d["tilt_east"][-1], d["tilt_north"][-1]

    base = apogee_tilt()
    assert apogee_tilt(cg_offset_mm=[0.0, 0.0], nose_misalignment_deg=[0.0, 0.0]) == base
    # pitch errors (about body y) and yaw errors move the trajectory; each must have an effect
    for kw in ({"cg_offset_mm": [2.0, 0.0]}, {"nose_misalignment_deg": [0.3, 0.0]}):
        tilt = apogee_tilt(**kw)
        assert math.hypot(tilt[0] - base[0], tilt[1] - base[1]) > math.radians(0.5), kw
