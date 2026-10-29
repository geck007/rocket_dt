"""Sensor models: IMUs (accelerometer + gyroscope), barometer and GNSS receiver.

Every sensor samples the true state at its own rate (a whole number of simulation
steps) and its sample becomes available to the flight software ``latency_s`` later.
Sensors can be switched off individually with ``enabled: false``.

Error models (all random quantities derive from ``sensors.seed``; each sensor has its
own stream, so switching one sensor off does not change the others' errors):

IMU (body-mounted at ``station`` / ``offset_yz_m``, nominal mount rotation
``orientation_deg`` known to the flight software), per axis, in the sensor frame:

    accel = sat(quant((I + S + M) f + b + G-M bias + noise))
    gyro  = sat(quant((I + S + M) w + b + G-M bias + g_sens f + noise))

    f   specific force at the sensor point (includes the lever-arm terms
        w' x r + w x (w x r), which matter at high roll rates); on the pad f = +g along +x
    S   scale-factor errors, M axis misalignment (off-diagonal), b turn-on bias,
        G-M a first-order Gauss-Markov bias drift, noise white with
        sigma = noise_density * sqrt(rate / 2)
    sat saturation at the range, quant quantisation to range / 2^(bits - 1)

Barometer: pressure at the sensor altitude (ISA, with the config's temperature
offset) + static-port error ``coeff * q`` through a first-order lag, plus bias,
white noise and quantisation. Temperature: true air temperature + bias + noise.

GNSS: antenna position and velocity in the launch ENU frame (east, north, up, metres
from the pad), with a slowly varying Gauss-Markov position error plus white noise.
Lock is lost while the acceleration exceeds ``max_accel_g`` and fixes are valid again
``reacquire_s`` after it drops below; invalid samples carry ``valid = False`` and NaNs.

Outputs are SI: m/s^2, rad/s, Pa, K, m, m/s.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from rocket_dt.atmosphere import G0, R_AIR, AirState
from rocket_dt.config import BaroConfig, GnssConfig, ImuConfig, SensorsConfig


# ----------------------------------------------------------------------------- samples (what the controller gets)


@dataclass
class ImuSample:
    t: float  # sample time, s
    accel: np.ndarray  # specific force, sensor frame, m/s^2
    gyro: np.ndarray  # angular rate, sensor frame, rad/s


@dataclass
class BaroSample:
    t: float
    pressure: float  # Pa
    temperature: float  # K


@dataclass
class GnssSample:
    t: float
    valid: bool
    position: np.ndarray  # ENU from the pad, m (NaN when not valid)
    velocity: np.ndarray  # ENU, m/s (NaN when not valid)


def pressure_altitude(pressure, reference_pressure, reference_temperature=288.15):
    """ISA altitude of ``pressure`` above the level where the pressure is ``reference_pressure``."""
    return reference_temperature / 0.0065 * (1.0 - (np.asarray(pressure) / reference_pressure)
                                             ** (0.0065 * R_AIR / G0))


# ----------------------------------------------------------------------------- truth seen by the sensors


@dataclass
class SensorTruth:
    t: float
    position: np.ndarray  # CG, ENU, m
    velocity: np.ndarray  # CG, ENU, m/s
    R: np.ndarray  # body -> ENU
    omega: np.ndarray  # body rates, rad/s
    cg_station: float
    g: float
    dynamic_pressure: float
    air_at: Callable[[float], AirState]  # height above the pad -> air
    accel: np.ndarray | None = None  # CG acceleration, ENU, m/s^2 (only when an IMU or GNSS samples)
    omega_dot: np.ndarray | None = None


def _cross(a, b):
    """3-vector cross product (np.cross has large per-call overhead)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _rot(roll, pitch, yaw):
    """Rotation matrix (sensor -> body) for a mount rotated by yaw, pitch, roll (degrees)."""
    r, p, y = np.radians([roll, pitch, yaw])
    cx, sx, cy, sy, cz, sz = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _error_matrix(rng, scale_sigma_pct, mis_sigma_deg):
    E = np.eye(3) + np.diag(rng.normal(0.0, scale_sigma_pct / 100.0, 3))
    off = rng.normal(0.0, math.radians(mis_sigma_deg), (3, 3))
    np.fill_diagonal(off, 0.0)
    return E + off


class _GaussMarkov:
    def __init__(self, rng, sigma, tau):
        self.rng, self.sigma, self.tau = rng, np.asarray(sigma, dtype=float), tau
        self.x = rng.normal(0.0, 1.0, self.sigma.shape) * self.sigma

    def step(self, dt):
        if not np.any(self.sigma):
            return self.x
        a = math.exp(-dt / self.tau)
        self.x = a * self.x + self.sigma * math.sqrt(1.0 - a * a) * self.rng.standard_normal(self.sigma.shape)
        return self.x


class _Sensor:
    name: str
    needs_accel = False

    def __init__(self, cfg, sim_dt: float, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.every = max(1, int(round(1.0 / (cfg.rate_hz * sim_dt))))
        self.period = self.every * sim_dt
        self.log: list[dict] = []

    def point(self, station, yz, cg_station):
        return np.array([cg_station - station, *yz])

    def advance(self, dt: float, truth: SensorTruth) -> None:
        """Continuous-time sensor dynamics, called every simulation step."""

    def sample(self, truth: SensorTruth):
        raise NotImplementedError

    def describe(self) -> str:
        return f"SENSOR {self.name}: {1 / self.period:.0f} Hz, latency {self.cfg.latency_s * 1e3:.0f} ms"


class Imu(_Sensor):
    needs_accel = True

    def __init__(self, cfg: ImuConfig, sim_dt, rng):
        super().__init__(cfg, sim_dt, rng)
        self.name = cfg.name
        a, g = cfg.accel, cfg.gyro
        self.C = _rot(*cfg.orientation_deg).T  # body -> sensor
        self.a_range = a.range_g * G0
        self.a_lsb = 2 * self.a_range / 2 ** a.bits
        self.a_sigma = a.noise_density_ug_rthz * 1e-6 * G0 * math.sqrt(cfg.rate_hz / 2)
        self.a_E = _error_matrix(rng, a.scale_factor_sigma_pct, a.misalignment_sigma_deg)
        self.a_b = rng.normal(0.0, a.bias_sigma_mg * 1e-3 * G0, 3)
        self.a_gm = _GaussMarkov(rng, np.full(3, a.bias_instability_mg * 1e-3 * G0), a.bias_correlation_s)
        self.g_range = math.radians(g.range_dps)
        self.g_lsb = 2 * self.g_range / 2 ** g.bits
        self.g_sigma = math.radians(g.noise_density_mdps_rthz * 1e-3) * math.sqrt(cfg.rate_hz / 2)
        self.g_E = _error_matrix(rng, g.scale_factor_sigma_pct, g.misalignment_sigma_deg)
        self.g_b = rng.normal(0.0, math.radians(g.bias_sigma_dps), 3)
        self.g_gm = _GaussMarkov(rng, np.full(3, math.radians(g.bias_instability_dps)), g.bias_correlation_s)
        self.g_sens = rng.normal(0.0, math.radians(g.g_sensitivity_dps_per_g) / G0, (3, 3))

    def sample(self, truth: SensorTruth) -> ImuSample:
        cfg = self.cfg
        r = self.point(cfg.station, cfg.offset_yz_m, truth.cg_station)
        w, wd = truth.omega, truth.omega_dot
        a_point = truth.accel + truth.R @ (_cross(wd, r) + _cross(w, _cross(w, r)))
        f_s = self.C @ (truth.R.T @ (a_point + np.array([0.0, 0.0, truth.g])))
        w_s = self.C @ w
        acc = self.a_E @ f_s + self.a_b + self.a_gm.step(self.period) + self.rng.normal(0.0, self.a_sigma, 3)
        gyr = (self.g_E @ w_s + self.g_b + self.g_gm.step(self.period) + self.g_sens @ f_s
               + self.rng.normal(0.0, self.g_sigma, 3))
        acc = np.clip(self.a_lsb * np.round(acc / self.a_lsb), -self.a_range, self.a_range)
        gyr = np.clip(self.g_lsb * np.round(gyr / self.g_lsb), -self.g_range, self.g_range)
        self.log.append({"t": truth.t, "ax": acc[0], "ay": acc[1], "az": acc[2],
                         "gx": gyr[0], "gy": gyr[1], "gz": gyr[2],
                         "true_ax": f_s[0], "true_ay": f_s[1], "true_az": f_s[2],
                         "true_gx": w_s[0], "true_gy": w_s[1], "true_gz": w_s[2]})
        return ImuSample(truth.t, acc, gyr)

    def describe(self) -> str:
        a, g = self.cfg.accel, self.cfg.gyro
        return (super().describe() + f", station {self.cfg.station:.3f} m; accel +-{a.range_g:g} g, "
                f"{a.noise_density_ug_rthz:g} ug/rtHz; gyro +-{g.range_dps:g} deg/s, "
                f"{g.noise_density_mdps_rthz:g} mdps/rtHz")


class Barometer(_Sensor):
    name = "baro"

    def __init__(self, cfg: BaroConfig, sim_dt, rng):
        super().__init__(cfg, sim_dt, rng)
        self.bias = rng.normal(0.0, cfg.bias_sigma_pa)
        self.t_bias = rng.normal(0.0, cfg.temperature_bias_sigma_k)
        self.p_lag = None

    def _true(self, truth: SensorTruth):
        r = self.point(self.cfg.station, (0.0, 0.0), truth.cg_station)
        h = truth.position[2] + (truth.R @ r)[2]
        air = truth.air_at(h)
        return air, h, air.pressure + self.cfg.static_error_coeff * truth.dynamic_pressure

    def advance(self, dt, truth):
        _, _, p = self._true(truth)
        if self.p_lag is None or self.cfg.lag_s <= 0:
            self.p_lag = p
        else:
            self.p_lag += (p - self.p_lag) * (1.0 - math.exp(-dt / self.cfg.lag_s))

    def sample(self, truth: SensorTruth) -> BaroSample:
        c = self.cfg
        air, h, _ = self._true(truth)
        if self.p_lag is None:
            self.advance(0.0, truth)
        p = self.p_lag + self.bias + self.rng.normal(0.0, c.noise_pa)
        if c.resolution_pa > 0:
            p = c.resolution_pa * round(p / c.resolution_pa)
        temp = air.temperature + self.t_bias + self.rng.normal(0.0, c.temperature_noise_k)
        self.log.append({"t": truth.t, "pressure": p, "temperature": temp,
                         "true_pressure": air.pressure, "true_altitude": h})
        return BaroSample(truth.t, p, temp)


class Gnss(_Sensor):
    name = "gnss"
    needs_accel = True

    def __init__(self, cfg: GnssConfig, sim_dt, rng):
        super().__init__(cfg, sim_dt, rng)
        self.err = _GaussMarkov(rng, [cfg.horizontal_sigma_m, cfg.horizontal_sigma_m, cfg.vertical_sigma_m],
                                cfg.correlation_time_s)
        self.locked = True  # warm on the pad
        self.t_below = math.inf

    def sample(self, truth: SensorTruth) -> GnssSample:
        c = self.cfg
        r = truth.R @ self.point(c.station, (0.0, 0.0), truth.cg_station)
        pos = truth.position + r
        vel = truth.velocity + truth.R @ _cross(truth.omega, self.point(c.station, (0.0, 0.0), truth.cg_station))
        if np.linalg.norm(truth.accel) > c.max_accel_g * G0:
            self.locked, self.t_below = False, 0.0
        elif not self.locked:
            self.t_below += self.period
            self.locked = self.t_below >= c.reacquire_s
        e = self.err.step(self.period)
        if self.locked:
            p = pos + e + self.rng.normal(0.0, c.white_noise_m, 3)
            v = vel + self.rng.normal(0.0, c.velocity_sigma_mps, 3)
        else:
            p = v = np.full(3, np.nan)
        self.log.append({"t": truth.t, "valid": float(self.locked), "e": p[0], "n": p[1], "u": p[2],
                         "ve": v[0], "vn": v[1], "vu": v[2], "true_e": pos[0], "true_n": pos[1], "true_u": pos[2],
                         "true_ve": vel[0], "true_vn": vel[1], "true_vu": vel[2]})
        return GnssSample(truth.t, self.locked, p, v)


# ----------------------------------------------------------------------------- suite


@dataclass
class SensorReadings:
    """New samples since the previous controller update, oldest first."""
    imu: dict[str, list[ImuSample]] = field(default_factory=dict)
    baro: list[BaroSample] = field(default_factory=list)
    gnss: list[GnssSample] = field(default_factory=list)


class SensorSuite:
    def __init__(self, cfg: SensorsConfig, sim_dt: float):
        self.cfg = cfg
        self.sim_dt = sim_dt
        self.reset()

    def reset(self) -> None:
        cfg, dt = self.cfg, self.sim_dt
        rng = lambda k: np.random.default_rng([cfg.seed, k])  # noqa: E731  one stream per sensor slot
        self.sensors: list[_Sensor] = [Imu(c, dt, rng(i)) for i, c in enumerate(cfg.imu) if c.enabled]
        if cfg.baro is not None and cfg.baro.enabled:
            self.sensors.append(Barometer(cfg.baro, dt, rng(1000)))
        if cfg.gnss is not None and cfg.gnss.enabled:
            self.sensors.append(Gnss(cfg.gnss, dt, rng(1001)))
        self._queue: deque = deque()
        self._pending = SensorReadings(imu={s.name: [] for s in self.sensors if isinstance(s, Imu)})

    @property
    def active(self) -> bool:
        return bool(self.sensors)

    def needs_accel(self, step: int) -> bool:
        return any(s.needs_accel and step % s.every == 0 for s in self.sensors)

    def step(self, step: int, truth: SensorTruth) -> None:
        for s in self.sensors:
            s.advance(self.sim_dt, truth)
            if step % s.every == 0:
                self._queue.append((truth.t + s.cfg.latency_s, s, s.sample(truth)))

    def read(self, t: float) -> SensorReadings:
        """Samples that have become available by time t (moved out of the latency queue)."""
        keep = deque()
        p = self._pending
        for item in sorted(self._queue, key=lambda x: x[0]):
            t_avail, s, smp = item
            if t_avail > t + 1e-12:
                keep.append(item)
            elif isinstance(s, Imu):
                p.imu[s.name].append(smp)
            elif isinstance(s, Barometer):
                p.baro.append(smp)
            else:
                p.gnss.append(smp)
        self._queue = keep
        self._pending = SensorReadings(imu={k: [] for k in p.imu})
        return p

    def logs(self) -> dict[str, dict[str, np.ndarray]]:
        return {s.name: {k: np.array([row[k] for row in s.log]) for k in s.log[0]}
                for s in self.sensors if s.log}

    def describe(self) -> list[str]:
        lines = [s.describe() for s in self.sensors]
        off = [c.name for c in self.cfg.imu if not c.enabled]
        off += [n for n, c in (("baro", self.cfg.baro), ("gnss", self.cfg.gnss)) if c is not None and not c.enabled]
        if off:
            lines.append(f"SENSORS disabled: {', '.join(off)}")
        return lines or ["SENSORS none configured"]
