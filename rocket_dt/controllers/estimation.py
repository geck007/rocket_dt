"""A simple strapdown state estimator built only from sensor samples.

On the pad (before launch is detected) it averages the IMU: the mean gyro reading is
the gyro bias, and the mean specific force points up, which gives the initial tilt.
The heading about the vertical can't be observed this way and is set arbitrarily; it
doesn't matter for tilt. After launch it integrates the gyro (attitude quaternion) and
the accelerometer (velocity, position), and uses the barometer for air density.

Without aiding the errors grow: accelerometer bias tilts the initial attitude by about
bias/g (20 mg -> ~1 deg), gyro g-sensitivity adds attitude error during the boost (a few
degrees at 0.1 deg/s/g), and that error tilts the integrated thrust, so the velocity is
off by several m/s at burnout and the position by tens of metres at apogee.

Optional GNSS aiding (``update_gnss``, used by pad_hold.py):

* Heading: GNSS is in true ENU, so the estimated heading has to be true too. Given the
  rail's tilt and azimuth (``rail_tilt_deg``, ``rail_azimuth_deg``: set up at the pad), the
  heading is set at launch so that the lean measured on the pad points along the rail's
  azimuth. The error is about atan(lean error / rail tilt): the lean error is ~bias/g
  plus the accelerometer misalignment, ~1-2 deg, so ~10-20 deg on a 5 deg rail. On a rail
  tilted less than ``min_align_tilt_deg`` the lean is mostly sensor error, the heading
  stays arbitrary and only the vertical GNSS channel is used.
* Origin: the mean GNSS position on the pad is the origin, which removes the receiver's
  slowly varying offset.
* Correction: each fix is compared with the inertial state at the fix's time (the receiver
  has latency), and a fraction (``gnss_gain_pos``, ``gnss_gain_vel``) of the difference is
  added to the current position and velocity. The attitude is not corrected.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from rocket_dt import quaternion as Q
from rocket_dt.atmosphere import G0, R_AIR
from rocket_dt.config import ImuConfig
from rocket_dt.sensors import BaroSample, GnssSample, ImuSample, mount_rotation

UP = np.array([0.0, 0.0, 1.0])


class StrapdownEstimator:
    def __init__(self, imu: ImuConfig, launch_accel_g: float = 3.0, launch_samples: int = 5,
                 rail_tilt_deg: float = 0.0, rail_azimuth_deg: float | None = None, min_align_tilt_deg: float = 4.0,
                 gnss_gain_pos: float = 0.3, gnss_gain_vel: float = 0.5):
        self.C_sb = mount_rotation(*imu.orientation_deg)  # sensor -> body
        self.launch_accel = launch_accel_g * G0
        self.launch_samples = launch_samples
        self.rail_azimuth = None if rail_azimuth_deg is None else math.radians(rail_azimuth_deg)
        self.align = self.rail_azimuth is not None and rail_tilt_deg >= min_align_tilt_deg
        self.gain_pos, self.gain_vel = gnss_gain_pos, gnss_gain_vel
        self.reset()

    def reset(self) -> None:
        self.launched = False
        self.t_launch = math.nan
        self.calibrated = False  # True if there were pad samples to calibrate from
        self.q = Q.from_two_vectors(np.array([1.0, 0.0, 0.0]), UP)  # until calibrated: assume vertical
        self.gyro_bias = np.zeros(3)
        self.omega = np.zeros(3)  # body rates, bias removed
        self.velocity = np.zeros(3)  # ENU (heading arbitrary), m/s
        self.position = np.zeros(3)
        self.air_density = 1.225
        self._t_last = None
        self._sum_g, self._sum_f, self._n = np.zeros(3), np.zeros(3), 0
        self._recent: deque = deque(maxlen=self.launch_samples)  # pad samples that may already be the launch
        self.heading_aligned = False  # True once the heading is true (GNSS horizontal channels usable)
        self.gnss_fixes = 0  # fixes used after launch
        self._gnss_sum, self._gnss_n = np.zeros(3), 0
        self._history: deque = deque(maxlen=1000)  # (t, position, velocity) after launch, for delayed fixes

    @property
    def R(self) -> np.ndarray:
        """Estimated rotation body -> ENU."""
        return Q.to_matrix(self.q)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def dynamic_pressure(self) -> float:
        return 0.5 * self.air_density * self.speed ** 2

    def update_imu(self, s: ImuSample) -> None:
        g_b, f_b = self.C_sb @ s.gyro, self.C_sb @ s.accel
        dt = 0.0 if self._t_last is None else s.t - self._t_last
        self._t_last = s.t
        if not self.launched:
            self._recent.append((g_b, f_b))
            above = sum(f[0] > self.launch_accel for _, f in self._recent)
            if above == self.launch_samples:
                self._launch(s.t - (self.launch_samples - 1) * dt)
            elif len(self._recent) == self.launch_samples:
                g0, f0 = self._recent[0]
                self._sum_g += g0
                self._sum_f += f0
                self._n += 1
            if not self.launched:
                return
        self.omega = g_b - self.gyro_bias
        w = float(np.linalg.norm(self.omega))
        if w > 0 and dt > 0:
            self.q = Q.normalize(Q.multiply(self.q, Q.from_axis_angle(self.omega, w * dt)))
        accel = self.R @ f_b - G0 * UP
        self.position += self.velocity * dt + 0.5 * accel * dt * dt
        self.velocity += accel * dt
        self._history.append((s.t, self.position.copy(), self.velocity.copy()))

    def update_baro(self, s: BaroSample) -> None:
        self.air_density = s.pressure / (R_AIR * s.temperature)

    def update_gnss(self, s: GnssSample) -> None:
        """Pad: average the fixes (origin). Flight: correct position and velocity towards the fix."""
        if not s.valid:
            return
        if not self.launched:
            self._gnss_sum += s.position
            self._gnss_n += 1
            return
        if self._gnss_n == 0 or not self._history or s.t < self._history[0][0]:
            return  # no pad origin, or the fix is older than the stored history
        t_h, p_h, v_h = next(h for h in reversed(self._history) if h[0] <= s.t)
        dp = (s.position - self._gnss_sum / self._gnss_n) - p_h
        dv = s.velocity - v_h
        if not self.heading_aligned:
            dp[:2] = dv[:2] = 0.0  # the horizontal axes don't match ENU
        cp, cv = self.gain_pos * dp, self.gain_vel * dv
        self.position += cp
        self.velocity += cv
        for h in self._history:  # so the next (delayed) fix is compared with corrected states
            h[1][:] += cp
            h[2][:] += cv
        self.gnss_fixes += 1

    def _launch(self, t: float) -> None:
        self.launched, self.t_launch = True, t
        if self._n > 0:
            self.calibrated = True
            self.gyro_bias = self._sum_g / self._n
            up_b = self._sum_f / np.linalg.norm(self._sum_f)
            self.q = Q.from_two_vectors(up_b, UP)  # rotates the measured "up" onto ENU up
            lean = self.R[:2, 0]  # horizontal part of the body axis, estimated frame
            if self.align and np.linalg.norm(lean) > 0.0:
                want = math.atan2(math.cos(self.rail_azimuth), math.sin(self.rail_azimuth))  # azimuth from north
                yaw = want - math.atan2(lean[1], lean[0])
                self.q = Q.normalize(Q.multiply(Q.from_axis_angle(UP, yaw), self.q))
                self.heading_aligned = True
