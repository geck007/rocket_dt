"""Sensor error models, latency, toggles and closed-loop plumbing."""

import copy
import math

import numpy as np
import pytest

from rocket_dt.atmosphere import G0, ISAAtmosphere
from rocket_dt.builder import build_simulation
from rocket_dt.config import (AccelConfig, BaroConfig, ConfigError, GnssConfig, GyroConfig, ImuConfig, SensorsConfig,
                              load_config, load_raw, parse_config)
from rocket_dt.control import Controller
from rocket_dt.sensors import SensorSuite, SensorTruth, pressure_altitude

DT = 0.002
ATM = ISAAtmosphere()


def _perfect_imu(**kw):
    accel = AccelConfig(bits=32, noise_density_ug_rthz=0, bias_sigma_mg=0, scale_factor_sigma_pct=0,
                        misalignment_sigma_deg=0)
    gyro = GyroConfig(bits=32, noise_density_mdps_rthz=0, bias_sigma_dps=0, scale_factor_sigma_pct=0,
                      misalignment_sigma_deg=0)
    return ImuConfig(**{"name": "imu0", "station": 1.0, "rate_hz": 500, "accel": accel, "gyro": gyro, **kw})


def _truth(t=0.0, R=None, omega=(0, 0, 0), accel=(0, 0, 0), pos=(0, 0, 0), vel=(0, 0, 0), q=0.0):
    return SensorTruth(t=t, position=np.array(pos, float), velocity=np.array(vel, float),
                       R=np.eye(3) if R is None else R, omega=np.array(omega, float), cg_station=1.0, g=G0,
                       dynamic_pressure=q, air_at=lambda h: ATM.at(h), accel=np.array(accel, float),
                       omega_dot=np.zeros(3))


def _one(suite, truth, step=0):
    suite.step(step, truth)
    return suite.read(truth.t)


def test_perfect_imu_on_the_pad_reads_plus_one_g_along_the_axis():
    vertical = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)  # body x -> up
    r = _one(SensorSuite(SensorsConfig(imu=[_perfect_imu()]), DT), _truth(R=vertical)).imu["imu0"][0]
    assert r.accel == pytest.approx([G0, 0, 0], abs=1e-6)
    assert r.gyro == pytest.approx([0, 0, 0], abs=1e-9)


def test_lever_arm_centripetal_acceleration():
    # 10 rad/s roll, IMU 2 cm off the axis in +y: centripetal -w^2 r along y; gravity along body z (R = I)
    imu = _perfect_imu(offset_yz_m=[0.02, 0.0])
    r = _one(SensorSuite(SensorsConfig(imu=[imu]), DT), _truth(omega=(10.0, 0, 0))).imu["imu0"][0]
    assert r.accel == pytest.approx([0.0, -100 * 0.02, G0], abs=1e-6)
    assert r.gyro == pytest.approx([10.0, 0, 0], abs=1e-9)


def test_mount_rotation_is_applied():
    imu = _perfect_imu(orientation_deg=[0.0, 0.0, 90.0])  # sensor x along body y
    r = _one(SensorSuite(SensorsConfig(imu=[imu]), DT), _truth(omega=(0, 1.0, 0))).imu["imu0"][0]
    assert r.gyro == pytest.approx([1.0, 0, 0], abs=1e-7)


def test_noise_level_saturation_and_quantisation():
    imu = ImuConfig(name="imu0", station=1.0, rate_hz=500,
                    accel=AccelConfig(noise_density_ug_rthz=100, bias_sigma_mg=0, scale_factor_sigma_pct=0,
                                      misalignment_sigma_deg=0),
                    gyro=GyroConfig(range_dps=2000, noise_density_mdps_rthz=5, bias_sigma_dps=0,
                                    scale_factor_sigma_pct=0, misalignment_sigma_deg=0))
    suite = SensorSuite(SensorsConfig(imu=[imu]), DT)
    acc = np.array([_one(suite, _truth(t=k * DT), k).imu["imu0"][0].accel for k in range(4000)])
    assert acc[:, 0].std() == pytest.approx(100e-6 * G0 * math.sqrt(250), rel=0.08)
    lsb = 2 * 16 * G0 / 2 ** 16
    assert np.allclose(acc / lsb, np.round(acc / lsb))
    fast = _one(suite, _truth(t=1.0, omega=(50.0, 0, 0)), 500).imu["imu0"][0]
    assert fast.gyro[0] == pytest.approx(math.radians(2000))  # 50 rad/s is beyond the 2000 deg/s range


def test_rates_and_latency():
    cfg = SensorsConfig(imu=[_perfect_imu(rate_hz=100, latency_s=0.01)], baro=BaroConfig(station=1.0, rate_hz=25))
    suite = SensorSuite(cfg, DT)
    got = {"imu": [], "baro": []}
    for k in range(250):  # 0.5 s
        tr = _truth(t=k * DT)
        suite.step(k, tr)
        r = suite.read(tr.t)
        got["imu"] += [(tr.t, s.t) for s in r.imu["imu0"]]
        got["baro"] += [(tr.t, s.t) for s in r.baro]
    assert len(got["imu"]) == 49 and len(got["baro"]) == 13  # 100 Hz minus the last 10 ms, 25 Hz
    assert all(t_read - t_s == pytest.approx(0.01) for t_read, t_s in got["imu"])
    assert all(t_read == t_s for t_read, t_s in got["baro"])


def test_barometer_and_pressure_altitude():
    b = BaroConfig(station=1.0, noise_pa=0, bias_sigma_pa=0, resolution_pa=0, lag_s=0, temperature_bias_sigma_k=0,
                   temperature_noise_k=0)
    r = _one(SensorSuite(SensorsConfig(baro=b), DT), _truth(pos=(0, 0, 1000.0))).baro[0]
    assert r.pressure == pytest.approx(ATM.at(1000.0).pressure)
    assert pressure_altitude(r.pressure, ATM.at(0.0).pressure) == pytest.approx(1000.0, abs=0.01)
    b.static_error_coeff = 0.01
    r = _one(SensorSuite(SensorsConfig(baro=b), DT), _truth(q=20000.0)).baro[0]
    assert r.pressure - ATM.at(0.0).pressure == pytest.approx(200.0)


def test_gnss_loses_lock_under_high_acceleration_and_reacquires():
    g = GnssConfig(station=1.0, rate_hz=10, latency_s=0.0, max_accel_g=4.0, reacquire_s=1.0)
    suite = SensorSuite(SensorsConfig(gnss=g), DT)
    valid = []
    for k in range(0, 2500, 50):  # every 0.1 s for 5 s
        t = k * DT
        a = (0, 0, 100.0) if 1.0 <= t < 2.0 else (0, 0, 0)
        valid.append(_one(suite, _truth(t=t, accel=a), k).gnss[0].valid)
    valid = np.array(valid)
    assert valid[:10].all() and not valid[10:20].any() and not valid[20:29].any() and valid[30:].all()


def test_each_sensor_has_its_own_random_stream():
    imu = ImuConfig(name="imu0", station=1.0)
    with_gnss = SensorSuite(SensorsConfig(seed=3, imu=[imu], gnss=GnssConfig(station=1.0)), DT)
    without = SensorSuite(SensorsConfig(seed=3, imu=[imu]), DT)
    other_seed = SensorSuite(SensorsConfig(seed=4, imu=[imu]), DT)
    tr = _truth()
    a, b, c = (_one(s, tr).imu["imu0"][0].accel for s in (with_gnss, without, other_seed))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_sensor_config_validation(root):
    raw = load_raw(root / "configs/baseline.yaml")
    bad = copy.deepcopy(raw)
    bad["sensors"]["imu"][0]["rate_hz"] = 1000  # above 1 / dt = 500 Hz
    with pytest.raises(ConfigError, match="rate"):
        parse_config(bad, root / "configs")
    bad = copy.deepcopy(raw)
    bad["sensors"]["imu"][1]["name"] = "imu0"
    with pytest.raises(ConfigError, match="unique"):
        parse_config(bad, root / "configs")
    bad = copy.deepcopy(raw)
    bad["sensors"]["gnss"]["antenna"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(bad, root / "configs")


class _Recorder(Controller):
    rate_hz = 100.0

    def __init__(self, n):
        self.n = n

    def reset(self):
        self.imu = {}
        self.baro, self.gnss, self.calls = [], [], 0

    def update(self, inp):
        self.calls += 1
        for k, v in inp.sensors.imu.items():
            self.imu.setdefault(k, []).extend(v)
        self.baro += inp.sensors.baro
        self.gnss += inp.sensors.gnss
        return np.zeros(self.n)


@pytest.mark.slow
def test_closed_loop_plumbing_with_all_sensors(root):
    cfg = load_config(root / "configs/baseline.yaml")
    cfg.simulation.stop_at = "apogee"
    cfg.sensors.gnss.enabled = True
    cfg.sensors.imu[1].enabled = False
    sim = build_simulation(cfg)
    ctrl = sim.controller = _Recorder(sim.rocket.aero.n_canards)
    res = sim.run()
    T = res.events["apogee"] + cfg.simulation.pad_time_s  # sensors and controller also run on the pad
    assert set(ctrl.imu) == {"imu0"} and set(res.sensors) == {"imu0", "baro", "gnss"}
    assert len(ctrl.imu["imu0"]) == pytest.approx(500 * T, abs=3)
    assert len(ctrl.baro) == pytest.approx(50 * T, abs=2)
    assert ctrl.calls == pytest.approx(100 * T, abs=2)
    # GNSS: lock lost during the boost (> 4 g), valid again later; latency 0.1 s
    gv = np.array([s.valid for s in ctrl.gnss])
    gt = np.array([s.t for s in ctrl.gnss])
    assert gv[gt < 0.1].all() and not gv[(gt > 0.3) & (gt < 2.0)].any() and gv[gt > 4.0].all()
    # measurements track the truth
    g = res.sensors["imu0"]
    boost = (g["t"] > 0.5) & (g["t"] < 1.5)
    assert np.median(g["ax"][boost] / g["true_ax"][boost]) == pytest.approx(1.0, abs=0.03)
    b = res.sensors["baro"]
    alt = pressure_altitude(b["pressure"], b["pressure"][0])
    assert np.abs(alt - (b["true_altitude"] - b["true_altitude"][0])).max() < 25.0
    n = res.sensors["gnss"]
    ok = n["valid"] > 0
    assert np.abs(n["u"][ok] - n["true_u"][ok]).max() < 15.0
