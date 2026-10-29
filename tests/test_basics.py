import math

import numpy as np
import pytest

from rocket_dt import quaternion as Q
from rocket_dt.atmosphere import ISAAtmosphere, gravity
from rocket_dt.geometry import fin_planform


def test_quaternion_constant_rate_matches_axis_angle():
    w = np.array([0.3, -1.2, 2.0])
    q = np.array([1.0, 0, 0, 0])
    h, n = 1e-3, 2000
    for _ in range(n):  # RK4
        k1 = Q.derivative(q, w)
        k2 = Q.derivative(q + h / 2 * k1, w)
        k3 = Q.derivative(q + h / 2 * k2, w)
        k4 = Q.derivative(q + h * k3, w)
        q = Q.normalize(q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    exact = Q.from_axis_angle(w, np.linalg.norm(w) * h * n)
    assert np.allclose(Q.to_matrix(q), Q.to_matrix(exact), atol=1e-9)


def test_quaternion_vertical_attitude_is_regular():
    """Pointing straight up (Euler singularity) is an ordinary attitude for quaternions."""
    q = Q.from_two_vectors(np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))
    R = Q.to_matrix(q)
    assert np.allclose(R @ [1, 0, 0], [0, 0, 1])
    assert np.allclose(R @ R.T, np.eye(3))


def test_isa_sea_level_and_1000m():
    atm = ISAAtmosphere()
    s0 = atm.at(0.0)
    assert s0.density == pytest.approx(1.2250, abs=1e-4)
    assert s0.speed_of_sound == pytest.approx(340.29, abs=0.05)
    assert s0.kinematic_viscosity == pytest.approx(1.46e-5, rel=0.01)
    assert atm.at(1000.0).density == pytest.approx(1.1117, abs=1e-3)


def test_gravity_matches_openrocket_value():
    assert gravity(28.61, 0.0) == pytest.approx(9.792, abs=5e-4)


def test_trapezoid_fin_mac_equals_barrowman_cp():
    cr, ct, sweep, s = 0.16, 0.14, 0.1133987, 0.16
    pf = fin_planform([(0, 0), (sweep, s), (sweep + ct, s), (cr, 0)])
    barrowman = sweep / 3 * (cr + 2 * ct) / (cr + ct) + (cr + ct - cr * ct / (cr + ct)) / 6
    assert pf.mac_lead + 0.25 * pf.mac_length == pytest.approx(barrowman, rel=1e-4)
    assert pf.area == pytest.approx((cr + ct) / 2 * s, rel=1e-6)
    assert pf.mac_span == pytest.approx(s / 3 * (cr + 2 * ct) / (cr + ct), rel=1e-4)
    assert math.tan(pf.midchord_sweep) == pytest.approx((sweep + ct / 2 - cr / 2) / s, rel=1e-6)
