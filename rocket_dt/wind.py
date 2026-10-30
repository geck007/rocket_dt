"""Wind: mean profile plus optional turbulence.

Frames: inertial ENU (x = east, y = north, z = up) with origin at the launch pad.
"""

from __future__ import annotations

import math

import numpy as np

from rocket_dt.config import WindConfig


class Wind:
    def __init__(self, cfg: WindConfig):
        self.cfg = cfg
        az = math.radians(cfg.direction_from_deg)
        # Wind blows FROM azimuth az, i.e. towards az + 180 deg.
        self._dir = -np.array([math.sin(az), math.cos(az), 0.0])
        self._rng = np.random.default_rng(cfg.turbulence.seed)
        self.gust = np.zeros(3)

    def mean_speed(self, h_agl: float) -> float:
        c = self.cfg
        if c.profile == "constant":
            return c.speed
        return c.speed * (max(h_agl, 0.5) / c.reference_height) ** c.exponent

    def mean(self, h_agl: float) -> np.ndarray:
        return self.mean_speed(h_agl) * self._dir

    def at(self, h_agl: float) -> np.ndarray:
        return self.mean(h_agl) + self.gust

    def update_turbulence(self, dt: float, h_agl: float, airspeed: float) -> None:
        """Advance the gust state by dt (held constant within an integration step).

        Dryden-type first-order shaping filters per axis, driven by the distance flown
        through the air. MIL-F-8785C scale lengths: the low-altitude forms below 1000 ft,
        1750 ft above 2000 ft, linearly interpolated in between.
        """
        t = self.cfg.turbulence
        if t.model == "none":
            return
        u_mean = self.mean_speed(h_agl)
        sigma_w = t.intensity * u_mean
        h_ft = max(h_agl, 3.0) / 0.3048
        k = 0.177 + 0.000823 * min(h_ft, 1000.0)
        sigma_u = sigma_w / k ** 0.4  # = sigma_w from 1000 ft up
        L_u, L_w = min(h_ft, 1000.0) / k ** 1.2, min(h_ft, 1000.0)
        if h_ft > 1000.0:
            f = min((h_ft - 1000.0) / 1000.0, 1.0)
            L_u, L_w = L_u + f * (1750.0 - L_u), L_w + f * (1750.0 - L_w)
        L_u, L_w = L_u * 0.3048, L_w * 0.3048
        sig = np.array([sigma_u, sigma_u, sigma_w])
        L = np.array([L_u, L_u, L_w])
        a = np.exp(-max(airspeed, 1.0) * dt / L)
        self.gust = a * self.gust + sig * np.sqrt(1.0 - a * a) * self._rng.standard_normal(3)
