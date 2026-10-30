"""Canard servo model and aerodynamic hinge moments."""

import math

import numpy as np
import pytest

from rocket_dt.actuators import IdealActuator, ServoActuator, make_actuator
from rocket_dt.builder import build_rocket
from rocket_dt.config import CanardActuatorConfig, load_config

DT = 0.002


def _servo(**kw):
    return ServoActuator(4, CanardActuatorConfig(model="servo", **kw))


def _run(act, cmd_deg, t_end, hinge=None):
    cmd = np.radians(np.broadcast_to(cmd_deg, (4,)).astype(float))
    out = []
    for _ in range(int(round(t_end / DT))):
        out.append(act.step(cmd, DT, hinge).copy())
    return np.degrees(np.array(out))


def test_small_step_is_second_order():
    zeta, fn = 0.7, 15.0
    y = _run(_servo(natural_frequency_hz=fn, damping_ratio=zeta), 0.5, 0.5)[:, 0]
    overshoot = y.max() / 0.5 - 1.0
    assert overshoot == pytest.approx(math.exp(-zeta * math.pi / math.sqrt(1 - zeta ** 2)), abs=0.01)
    t_peak = (np.argmax(y) + 1) * DT
    assert t_peak == pytest.approx(math.pi / (2 * math.pi * fn * math.sqrt(1 - zeta ** 2)), abs=2 * DT)
    assert y[-1] == pytest.approx(0.5, abs=1e-4)


def test_large_step_is_rate_limited():
    act = _servo(max_rate_dps=300.0)
    y = _run(act, 12.0, 0.4)[:, 0]
    rate = np.diff(y) / DT
    assert rate.max() <= 300.0 + 1e-6
    assert rate.max() > 200.0  # it does saturate on the torque-speed line
    assert y[-1] == pytest.approx(12.0, abs=0.01)


def test_hinge_moment_deflects_servo_by_load_over_stiffness():
    K, H = 40.0, 0.2
    y = _run(_servo(stiffness=K), 2.0, 1.0, hinge=np.full(4, H))[:, 0]
    assert y[-1] - 2.0 == pytest.approx(math.degrees(H / K), rel=1e-3)


def test_stall_torque_limits_holding():
    # a hinge moment above stall torque drives the canard to the stop
    y = _run(_servo(stall_torque=0.5), 0.0, 1.0, hinge=np.full(4, 0.8))[:, 0]
    assert y[-1] == pytest.approx(15.0)


def test_deadband_resolution_backlash_and_limits():
    assert np.all(_run(_servo(deadband_deg=0.2), 0.09, 0.3) == 0.0)  # inside half the deadband
    act = _servo(resolution_deg=0.09)
    act.step(np.radians([0.05, 0.04, -0.05, 0.2]), DT)
    assert np.degrees(act.setpoint) == pytest.approx([0.09, 0.0, -0.09, 0.18])
    # backlash: loaded canard rests against the side the load pushes it to
    act = _servo(backlash_deg=0.2)
    y = _run(act, 0.0, 0.5, hinge=np.array([0.01, -0.01, 0.0, 0.0]))
    servo = np.degrees(act.theta)
    assert y[-1, 0] - servo[0] == pytest.approx(0.1) and y[-1, 1] - servo[1] == pytest.approx(-0.1)
    assert y[-1, 2] == pytest.approx(0.0, abs=1e-9)
    assert np.abs(_run(_servo(), 40.0, 0.5)).max() <= 15.0 + 1e-9


def test_ideal_and_factory():
    assert isinstance(make_actuator(CanardActuatorConfig(), 4), IdealActuator)
    assert isinstance(make_actuator(CanardActuatorConfig(model="servo"), 4), ServoActuator)
    a = IdealActuator(4)
    assert np.degrees(a.step(np.radians([1, 20, -20, 0]), DT)) == pytest.approx([1, 15, -15, 0])


@pytest.fixture(scope="module")
def rocket():
    from pathlib import Path
    return build_rocket(load_config(Path(__file__).resolve().parents[1] / "configs/baseline.yaml"))


def _hinge(rk, delta_deg, v_b=(100.0, 0.0, 0.0)):
    return rk.aero.evaluate(np.asarray(v_b, float), np.zeros(3), 0.0, 0.0, np.radians(delta_deg), 1.225, 340.3,
                            1.46e-5, rk.mass.at(0.0).cg_station).hinge_moment


def test_hinge_at_centroid_is_aft_of_cp_so_load_increases_deflection(rocket):
    h = _hinge(rocket, [1.0, 0.0, 0.0, 0.0])
    assert h[0] > 0
    # the neighbours of a deflected canard are loaded the other way (fin-fin interference), much less
    assert h[1] == pytest.approx(h[3]) and -0.15 * h[0] < h[1] < 0 and -0.15 * h[0] < h[2] < 0
    assert _hinge(rocket, [2.0, 0.0, 0.0, 0.0])[0] == pytest.approx(2 * h[0], rel=0.03)  # nearly linear
    assert _hinge(rocket, [-1.0, 0.0, 0.0, 0.0])[0] == pytest.approx(-h[0])
    assert rocket.aero.canard_hinge == pytest.approx(rocket.aero.fins[0].planform.centroid_x)


def test_hinge_ahead_of_cp_restores(root):
    cfg = load_config(root / "configs/baseline.yaml")
    cfg.actuators.canards.hinge_from_root_le = 0.0  # root leading edge
    assert _hinge(build_rocket(cfg), [1.0] * 4)[0] < 0
