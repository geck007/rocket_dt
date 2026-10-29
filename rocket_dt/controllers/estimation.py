"""A simple strapdown state estimator built only from sensor samples.

On the pad (before launch is detected) it averages the IMU: the mean gyro reading is
the gyro bias, and the mean specific force points up, which gives the initial tilt.
The heading about the vertical can't be observed this way and is set arbitrarily; it
doesn't matter for tilt. After launch it integrates the gyro (attitude quaternion) and
the accelerometer (velocity, position), and uses the barometer for air density.

No aiding after launch, so errors grow: accelerometer bias tilts the initial attitude
by about bias/g (20 mg -> ~1 deg), gyro g-sensitivity adds attitude error during the
boost, and velocity drifts with the accelerometer bias. A good next step is to aid it
with the barometer (altitude, vertical speed) and GNSS when available.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from rocket_dt import quaternion as Q
from rocket_dt.atmosphere import G0, R_AIR
from rocket_dt.config import ImuConfig
from rocket_dt.sensors import BaroSample, ImuSample, mount_rotation

UP = np.array([0.0, 0.0, 1.0])


class StrapdownEstimator:
    def __init__(self, imu: ImuConfig, launch_accel_g: float = 3.0, launch_samples: int = 5):
        self.C_sb = mount_rotation(*imu.orientation_deg)  # sensor -> body
        self.launch_accel = launch_accel_g * G0
        self.launch_samples = launch_samples
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

    def update_baro(self, s: BaroSample) -> None:
        self.air_density = s.pressure / (R_AIR * s.temperature)

    def _launch(self, t: float) -> None:
        self.launched, self.t_launch = True, t
        if self._n > 0:
            self.calibrated = True
            self.gyro_bias = self._sum_g / self._n
            up_b = self._sum_f / np.linalg.norm(self._sum_f)
            self.q = Q.from_two_vectors(up_b, UP)  # rotates the measured "up" onto ENU up
