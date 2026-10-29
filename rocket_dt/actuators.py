"""Canard actuators: one servo per canard, each canard hinged on a radial line
(``actuators.canards.hinge_from_root_le``, default the planform area centroid).

``IdealActuator``: the deflection equals the clipped command.

``ServoActuator``: a position-controlled servo modelled as a second-order system
with the nonlinearities that matter for roll control at tiny deflections:

    J theta'' = clip(K e - C theta', torque available at this speed) + H_aero

    e         command minus output angle, minus the deadband (no torque while |e| < deadband/2)
    K, C      stiffness and damping: K = holding stiffness, C = 2 zeta K / wn, J = K / wn^2,
              so the unloaded small-signal response has natural frequency wn and damping zeta
    torque    limited by a linear torque-speed line from stall torque (at rest) to zero
              (at the no-load speed), which sets the rate limit
    H_aero    aerodynamic hinge moment from aero.py (positive = increases the deflection);
              with the hinge aft of the canard's centre of pressure it pushes the canard
              away from neutral and the servo must hold against it

The command is clipped to the stops and quantised to ``resolution_deg``. The canard sits on
the output shaft with ``backlash_deg`` of play: under load it rests against the side the
hinge moment pushes it to, otherwise it stays where it was inside the play.

Canard zero offsets (installation errors) are not modelled here: they are
``disturbances.fin_misalignment_deg`` of the canard fin set.
"""

from __future__ import annotations

import math

import numpy as np

from rocket_dt.config import CanardActuatorConfig


class IdealActuator:
    """Deflection follows the command instantly, clipped to +-limit."""

    def __init__(self, n: int, limit_deg: float = 15.0):
        self.limit = np.radians(limit_deg)
        self.deflection = np.zeros(n)

    def reset(self) -> None:
        self.deflection[:] = 0.0

    def step(self, command: np.ndarray, dt: float, hinge_moment: np.ndarray | None = None) -> np.ndarray:
        self.deflection = np.clip(command, -self.limit, self.limit)
        return self.deflection

    def describe(self) -> str:
        return f"ACTUATOR ideal: deflection = command, limit +-{math.degrees(self.limit):.1f} deg"


class ServoActuator:
    def __init__(self, n: int, cfg: CanardActuatorConfig):
        self.n = n
        self.cfg = cfg
        self.limit = math.radians(cfg.limit_deg)
        self.wn = 2.0 * math.pi * cfg.natural_frequency_hz
        self.K = cfg.stiffness
        self.C = 2.0 * cfg.damping_ratio * cfg.stiffness / self.wn
        self.J = cfg.stiffness / self.wn ** 2
        self.tau_max = cfg.stall_torque
        self.w_max = math.radians(cfg.max_rate_dps)
        self.half_db = 0.5 * math.radians(cfg.deadband_deg)
        self.res = math.radians(cfg.resolution_deg)
        self.half_bl = 0.5 * math.radians(cfg.backlash_deg)
        self.reset()

    def reset(self) -> None:
        self.theta = np.zeros(self.n)  # servo output shaft angle
        self.omega = np.zeros(self.n)
        self.canard = np.zeros(self.n)  # canard angle (output + backlash)
        self.deflection = np.zeros(self.n)
        self.setpoint = np.zeros(self.n)  # command after clipping and quantisation

    def step(self, command: np.ndarray, dt: float, hinge_moment: np.ndarray | None = None) -> np.ndarray:
        cmd = np.clip(np.asarray(command, dtype=float), -self.limit, self.limit)
        if self.res > 0:
            cmd = self.res * np.round(cmd / self.res)
        self.setpoint = cmd
        H = np.zeros(self.n) if hinge_moment is None else np.asarray(hinge_moment, dtype=float)
        n_sub = max(1, math.ceil(dt * self.wn / 0.1))
        h = dt / n_sub
        th, w = self.theta, self.omega
        for _ in range(n_sub):  # semi-implicit Euler
            e = cmd - th
            if self.half_db > 0:
                e = e - np.clip(e, -self.half_db, self.half_db)
            tau = self.K * e - self.C * w
            motoring = tau * w > 0
            avail = np.where(motoring, self.tau_max * np.clip(1.0 - np.abs(w) / self.w_max, 0.0, 1.0), self.tau_max)
            tau = np.clip(tau, -avail, avail)
            w = w + h * (tau + H) / self.J
            th = th + h * w
            stop = np.abs(th) > self.limit
            if stop.any():
                th = np.where(stop, np.sign(th) * self.limit, th)
                w = np.where(stop & (th * w > 0), 0.0, w)
        self.theta, self.omega = th, w
        if self.half_bl > 0:
            c = np.clip(self.canard, th - self.half_bl, th + self.half_bl)
            loaded = np.abs(H) > 1e-4
            self.canard = np.where(loaded, th + np.sign(H) * self.half_bl, c)
        else:
            self.canard = th
        self.deflection = self.canard.copy()
        return self.deflection

    def describe(self) -> str:
        c = self.cfg
        return (f"ACTUATOR servo x{self.n}: {c.natural_frequency_hz:g} Hz, zeta {c.damping_ratio:g}, "
                f"stiffness {c.stiffness:g} N m/rad (J {self.J:.2e} kg m^2), stall {c.stall_torque:g} N m, "
                f"{c.max_rate_dps:g} deg/s, limit +-{c.limit_deg:g} deg, deadband {c.deadband_deg:g}, "
                f"resolution {c.resolution_deg:g}, backlash {c.backlash_deg:g} deg")


def make_actuator(cfg: CanardActuatorConfig, n: int):
    return ServoActuator(n, cfg) if cfg.model == "servo" else IdealActuator(n, cfg.limit_deg)
