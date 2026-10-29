"""Canard control interfaces.

The simulation calls ``controller.update`` at ``controller.rate_hz`` and holds the
command between calls (zero-order hold), exactly like flight software running in a
fixed-rate loop. The actuator model turns commands into actual deflections.

Today the controller receives the true state; a sensor layer will later sit between
truth and controller without changing this interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TruthSnapshot:
    t: float
    position: np.ndarray  # ENU, m
    velocity: np.ndarray  # ENU, m/s
    quaternion: np.ndarray  # body -> ENU
    omega: np.ndarray  # body rates, rad/s
    fincan_rate: float  # fin-can roll rate relative to body, rad/s
    airspeed: float
    mach: float
    dynamic_pressure: float


class Controller:
    rate_hz: float = 100.0

    def reset(self) -> None:
        pass

    def update(self, snap: TruthSnapshot) -> np.ndarray:
        raise NotImplementedError


class NoControl(Controller):
    """Canards held at zero."""

    def __init__(self, n_canards: int):
        self.n = n_canards

    def update(self, snap: TruthSnapshot) -> np.ndarray:
        return np.zeros(self.n)


class RollDeflectionSchedule(Controller):
    """Open-loop test: all canards deflected by the same (roll) angle versus time.

    ``schedule`` is a list of (t, delta_deg) points, linearly interpolated.
    """

    def __init__(self, n_canards: int, schedule: list[tuple[float, float]], rate_hz: float = 100.0):
        self.n = n_canards
        self.t = np.array([p[0] for p in schedule], dtype=float)
        self.d = np.radians([p[1] for p in schedule])
        self.rate_hz = rate_hz

    def update(self, snap: TruthSnapshot) -> np.ndarray:
        return np.full(self.n, float(np.interp(snap.t, self.t, self.d)))


class IdealActuator:
    """Deflection follows the command instantly, clipped to +-limit."""

    def __init__(self, n: int, limit_deg: float = 15.0):
        self.limit = np.radians(limit_deg)
        self.deflection = np.zeros(n)

    def reset(self) -> None:
        self.deflection[:] = 0.0

    def step(self, command: np.ndarray, dt: float) -> np.ndarray:
        self.deflection = np.clip(command, -self.limit, self.limit)
        return self.deflection
