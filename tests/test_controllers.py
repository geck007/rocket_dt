"""Controller registry, pad phase, strapdown estimator and the vertical_hold example."""

import copy
import math

import numpy as np
import pytest

from rocket_dt import quaternion as Q
from rocket_dt.atmosphere import G0
from rocket_dt.builder import build_simulation
from rocket_dt.config import ConfigError, ImuConfig, load_config, load_raw, parse_config
from rocket_dt.controllers import CONTROLLERS, make_controller
from rocket_dt.controllers.estimation import StrapdownEstimator
from rocket_dt.controllers.vertical_hold import VerticalHold
from rocket_dt.sensors import ImuSample

DT = 0.002


def test_controller_config_is_strict(root):
    raw = load_raw(root / "configs/vertical_hold.yaml")
    cfg = parse_config(copy.deepcopy(raw), root / "configs")
    from rocket_dt.builder import build_rocket
    rk = build_rocket(cfg)
    assert isinstance(make_controller(cfg, rk), VerticalHold)
    bad = copy.deepcopy(raw)
    bad["controller"]["params"]["kpp"] = 1.0
    with pytest.raises(ConfigError, match="unknown key"):
        make_controller(parse_config(bad, root / "configs"), rk)
    bad["controller"] = {"type": "autopilot"}
    with pytest.raises(ConfigError, match="not registered"):
        make_controller(parse_config(bad, root / "configs"), rk)


def _feed(est, n, gyro, accel, t0=0.0):
    for k in range(n):
        est.update_imu(ImuSample(t=t0 + k * DT, accel=np.asarray(accel, float), gyro=np.asarray(gyro, float)))
    return t0 + n * DT


def test_estimator_calibrates_on_the_pad_and_detects_launch():
    est = StrapdownEstimator(ImuConfig(name="imu0", station=1.0))
    tilt = math.radians(5.0)
    f_pad = G0 * np.array([math.cos(tilt), 0.0, math.sin(tilt)])  # rail leaning 5 deg: gravity partly along body z
    bias = np.array([0.01, -0.02, 0.005])
    t = _feed(est, 1000, bias, f_pad, t0=-2.0)
    assert not est.launched
    _feed(est, 10, bias, [60.0, 0.0, 0.0], t0=t)
    assert est.launched and est.calibrated
    assert est.gyro_bias == pytest.approx(bias, abs=1e-12)
    axis = est.R[:, 0]
    assert math.degrees(math.acos(axis[2])) == pytest.approx(5.0, abs=1e-6)


def test_estimator_integrates_rotation():
    est = StrapdownEstimator(ImuConfig(name="imu0", station=1.0))
    t = _feed(est, 500, [0, 0, 0], [G0, 0, 0], t0=-1.0)
    t = _feed(est, 10, [0, 0, 0], [60.0, 0, 0], t0=t)
    _feed(est, 500, [0.0, math.radians(10.0), 0.0], [60.0, 0, 0], t0=t)  # 10 deg/s pitch for 1 s
    assert math.degrees(math.acos(est.R[2, 0])) == pytest.approx(10.0, abs=0.1)


@pytest.fixture(scope="module")
def vh_sim():
    from pathlib import Path
    return build_simulation(load_config(Path(__file__).resolve().parents[1] / "configs/vertical_hold.yaml"))


def test_vertical_hold_command_turns_the_rocket_back_upright(vh_sim):
    """Signs end to end: for a tilted rocket the commanded canards produce a moment that reduces the tilt."""
    rk, ctrl = vh_sim.rocket, vh_sim.controller
    ctrl.reset()
    est = ctrl.est
    est.launched, est.velocity = True, np.array([0.0, 0.0, 100.0])
    for axis in ([0, 1, 0], [0, 0, 1], [0, 1, 1]):
        est.q = Q.multiply(Q.from_two_vectors(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
                           Q.from_axis_angle(axis, math.radians(3.0)))  # nose tipped 3 deg off vertical
        ctrl.integral[:] = 0.0
        from rocket_dt.control import ControllerInput
        from rocket_dt.sensors import SensorReadings
        cmd = ctrl.update(ControllerInput(1.0, SensorReadings(imu={"imu0": []}), None))
        assert ctrl.active and np.abs(cmd).max() > 0
        r = rk.aero.evaluate(np.array([100.0, 0, 0]), np.zeros(3), 0.0, 0.0, cmd, 1.225, 340.3, 1.46e-5,
                             rk.mass.at(1.0).cg_station)
        r0 = rk.aero.evaluate(np.array([100.0, 0, 0]), np.zeros(3), 0.0, 0.0, None, 1.225, 340.3, 1.46e-5,
                              rk.mass.at(1.0).cg_station)
        m_world = est.R @ (r.moment - r0.moment)
        wanted = np.cross(est.R[:, 0], [0, 0, 1.0])  # rotation axis that brings the nose up
        assert m_world @ wanted > 0.9 * np.linalg.norm(m_world) * np.linalg.norm(wanted)


def test_pad_phase_runs_sensors_before_ignition(root):
    cfg = load_config(root / "configs/baseline.yaml")
    cfg.simulation.pad_time_s = 1.0
    cfg.simulation.t_max = 0.5
    sim = build_simulation(cfg)
    res = sim.run()
    imu = res.sensors["imu0"]
    assert imu["t"][0] == pytest.approx(-1.0)
    pad = imu["t"] < 0
    assert np.allclose(imu["true_ax"][pad], sim.g, atol=1e-9)  # at rest on a vertical rail: +1 g along the axis
    assert res.data["t"][0] == 0.0 and res.events["liftoff"] < 0.01


@pytest.mark.slow
def test_vertical_hold_keeps_the_rocket_upright(root):
    out = {}
    for kind in ("none", "vertical_hold"):
        cfg = load_config(root / "configs/vertical_hold.yaml")
        cfg.controller.type = kind
        if kind == "none":
            cfg.controller.params = {}
        r = build_simulation(cfg).run()
        d = r.data
        asc = d["t"] >= r.events["rail_exit"]
        out[kind] = (math.degrees(d["zenith_angle"][asc].max()), d)
    assert out["none"][0] > 45.0  # 5 deg rail + 4 m/s wind: weathercocks far over without control
    assert out["vertical_hold"][0] < 10.0
    d = out["vertical_hold"][1]
    assert d["ctrl_active"].max() == 1.0 and d["ctrl_calibrated"][-1] == 1.0
    assert np.degrees(np.abs(d["p"]).max()) < 30.0  # roll damped


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CONTROLLERS))
def test_every_controller_flies_the_first_seconds(root, name):
    """Smoke test for every controller module in rocket_dt/controllers/ (default params, baseline rocket)."""
    cfg = load_config(root / "configs/baseline.yaml")
    cfg.controller.type, cfg.controller.params = name, {}
    cfg.simulation.t_max = 3.0
    sim = build_simulation(cfg)
    n = sim.rocket.aero.n_canards
    update = sim.controller.update

    def checked(inp):
        cmd = np.asarray(update(inp), dtype=float)
        assert cmd.shape == (n,), f"{name} must return one deflection (rad) per canard"
        assert np.all(np.isfinite(cmd)), f"{name} returned a non-finite command"
        return cmd

    sim.controller.update = checked
    d = sim.run().data
    assert d["t"][-1] > 2.9
    assert all(np.all(np.isfinite(v)) for k, v in d.items() if k.startswith("ctrl_"))
