"""Example controller: keep the rocket over the pad (minimise its distance from the pad's vertical).

It is vertical_hold with a guidance loop on top: instead of pointing the axis straight up,
it points it at a target direction that steers the horizontal position back to zero.
The attitude loop (canard mixing, rate and roll damping, integral) is vertical_hold's.

Guidance, with r and v the estimated horizontal position and velocity (pad = origin):

    v_des = -r / tau                          horizontal velocity that returns to the axis in ~tau
    a     = (v_des + k_v (v_des - v)) / V     horizontal tilt of the target, rad (small angle)
    t     = normalize(a_e, a_n, 1)            limited to max_tilt_deg from vertical

Pointing the axis along the desired velocity, plus an overshoot proportional to the velocity
error, turns the flight path towards it: in the boost the thrust follows the axis, and in
the coast the aerodynamic lift acts towards the side the nose points. With r = v = 0 the
target is vertical and this is vertical_hold.

Navigation: the guidance is only as good as the position estimate. The IMU alone drifts by
tens of metres by apogee (gyro g-sensitivity in the boost, see estimation.py), which is
as large as the distance being removed, so this controller aids the estimator with GNSS
(``sensors.gnss.enabled: true``, as in configs/pad_hold.yaml). The receiver loses lock in
the boost and reacquires about a second after burnout; until then the estimate is inertial.
GNSS needs a true heading, which is taken from the rail: its tilt and azimuth as measured
at the pad (``rail_tilt_deg``, ``rail_azimuth_deg``; by default the config's ``launch``
values) and the lean the accelerometer measures on the pad. The heading error is
~15 deg rms on a 5 deg rail, which costs little.

Without a usable position (GNSS disabled, or a rail tilted less than ``min_align_tilt_deg``
so the heading can't be aligned) the target is vertical: steering on the inertial position
alone roughly doubles the tilt and barely reduces the distance. A magnetometer, or a
heading state in a Kalman filter, would remove the need for a tilted rail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rocket_dt.control import ControllerInput
from rocket_dt.controllers import vertical_hold
from rocket_dt.controllers.estimation import StrapdownEstimator


@dataclass
class Params(vertical_hold.Params):
    tau: float = 2.0  # s, time to return to the axis
    k_v: float = 3.0  # extra tilt per unit of velocity error (dimensionless, per V)
    max_tilt_deg: float = 10.0  # limit of the target's tilt from vertical
    gnss_gain_pos: float = 0.3  # fraction of each GNSS position difference applied
    gnss_gain_vel: float = 0.5
    min_align_tilt_deg: float = 4.0  # rail tilt below which the heading can't be aligned
    rail_tilt_deg: float | None = None  # rail as measured at the pad; None = launch.angle_from_vertical_deg
    rail_azimuth_deg: float | None = None  # None = launch.azimuth_deg


class PadHold(vertical_hold.VerticalHold):
    def __init__(self, rocket, cfg, params: Params):
        super().__init__(rocket, cfg, params)
        tilt = cfg.launch.angle_from_vertical_deg if params.rail_tilt_deg is None else params.rail_tilt_deg
        azimuth = cfg.launch.azimuth_deg if params.rail_azimuth_deg is None else params.rail_azimuth_deg
        self.has_gnss = cfg.sensors.gnss is not None and cfg.sensors.gnss.enabled
        self.est = StrapdownEstimator(self.imu, rail_tilt_deg=tilt, rail_azimuth_deg=azimuth,
                                      min_align_tilt_deg=params.min_align_tilt_deg,
                                      gnss_gain_pos=params.gnss_gain_pos, gnss_gain_vel=params.gnss_gain_vel)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.target_tilt = np.zeros(2)

    def update(self, inp: ControllerInput) -> np.ndarray:
        # fixes arrive after the receiver's latency, so the IMU history already covers their time
        for s in inp.sensors.gnss:
            self.est.update_gnss(s)
        return super().update(inp)

    @property
    def guiding(self) -> bool:
        """True if the horizontal position estimate is GNSS-aided, so steering on it pays."""
        return self.has_gnss and self.est.heading_aligned

    def target_direction(self) -> np.ndarray:
        p, est = self.p, self.est
        if not self.guiding:
            self.target_tilt = np.zeros(2)
            return self.target  # vertical_hold's target
        r, v = est.position[:2], est.velocity[:2]
        v_des = -r / p.tau
        a = (v_des + p.k_v * (v_des - v)) / max(est.speed, p.min_speed)
        lim = math.tan(math.radians(p.max_tilt_deg))
        n = float(np.linalg.norm(a))
        if n > lim:
            a *= lim / n
        self.target_tilt = a
        t = np.array([a[0], a[1], 1.0])
        return t / np.linalg.norm(t)

    def telemetry(self) -> dict:
        r = self.est.position[:2]
        return super().telemetry() | {
            "dist_est": float(np.linalg.norm(r)),
            "target_tilt_deg": math.degrees(math.atan(float(np.linalg.norm(self.target_tilt)))),
            "gnss_fixes": float(self.est.gnss_fixes), "guiding": float(self.guiding)}


CONTROLLER = PadHold
