"""Controller interface.

The simulation calls ``controller.update(inp)`` at ``controller.rate_hz`` and holds the
returned command (one deflection per canard, rad) until the next call (zero-order
hold), like flight software running in a fixed-rate loop. The actuator model
(actuators.py) turns commands into actual canard deflections.

``inp.sensors`` holds every sensor sample that has become available since the previous
call (after each sensor's latency), oldest first. Flight software must only use these.
``inp.truth`` is the true state, for debugging and for developing a controller with
perfect state knowledge before adding an estimator; a controller meant to fly must
not use it.

Optional ``telemetry()``: a dict of floats (the same keys on every call) that is
logged with the flight as ``ctrl_<key>`` columns.

Controllers that can be chosen from a config are registered in
``rocket_dt.controllers``; see ``rocket_dt/controllers/vertical_hold.py`` for an example.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rocket_dt.sensors import SensorReadings


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


@dataclass
class ControllerInput:
    t: float  # s; negative while on the pad before ignition (simulation.pad_time_s)
    sensors: SensorReadings  # new samples since the previous update
    truth: TruthSnapshot  # debugging / ideal-sensor studies only


class Controller:
    rate_hz: float = 100.0

    def reset(self) -> None:
        """Called at the start of every run."""

    def update(self, inp: ControllerInput) -> np.ndarray:
        raise NotImplementedError


class NoControl(Controller):
    """Canards held at zero."""

    def __init__(self, n_canards: int):
        self.n = n_canards

    def update(self, inp: ControllerInput) -> np.ndarray:
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

    def update(self, inp: ControllerInput) -> np.ndarray:
        return np.full(self.n, float(np.interp(inp.t, self.t, self.d)))
