"""Example controller: keep the rocket's axis vertical, and damp body roll.

It uses only sensor samples (StrapdownEstimator) and runs only during powered and
coasting ascent: canards stay at zero on the pad, on the rail (until the estimated
speed exceeds ``min_speed``) and after apogee.

Control law, with u the lateral command of each canard pair (rad of deflection):

    e    = b x t        tilt error: b = estimated body axis, t = target direction (ENU);
                        |e| = sin(tilt), e is the axis to rotate about to remove it
    u    = R^T (kp e + ki integral(e) dt) - (kd / V) w_lat      (body frame, y and z)
    d_r  = -k_roll p / V                                        (roll damping)
    d_k  = d_r - (u . e_r,k)                                    canard k at roll angle phi_k

e_r,k = (cos phi_k, sin phi_k) is canard k's radial direction in body (y, z). A
positive deflection of canard k pitches the nose about -e_r,k, so the minus sign turns
u into a moment along +u. Canard lift and the rocket's aerodynamic restoring moment
both scale with dynamic pressure, so a deflection-per-tilt gain kp has the same relative
effect at any speed. The rotational frequency grows with speed, so the rate damping is
divided by the speed V. The integral is kept in the ENU frame so the body's roll does
not mix its axes, and it is limited to max_deflection.

Tuning notes (current .ork, Mach 0.3, liftoff CG): one canard gives about 0.0080 q N m/rad
of pitch moment, including its downwash on the tail and half the slender-body body
carry-over (``aero.canard_body_carryover`` 0.5; 0.0063 with none, 0.0097 with all of it).
The aerodynamic restoring moment is about 0.048 q N m per rad of angle of attack, so
kp = 5 makes the canards about 1.7 times stiffer than the rocket's own weathercocking. All
four canards give about 0.29 N m of roll per degree at 100 m/s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rocket_dt.control import Controller, ControllerInput
from rocket_dt.controllers.estimation import StrapdownEstimator


@dataclass
class Params:
    imu: str = "imu0"  # preferred IMU; the first enabled IMU is used if this one is disabled
    target_direction: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])  # ENU; default straight up
    kp: float = 5.0  # rad of pair deflection per rad of tilt
    ki: float = 2.5  # per rad s of tilt
    kd: float = 75.0  # m: rate damping = kd / V rad per rad/s
    k_roll: float = 0.76  # m: roll damping = k_roll / V rad per rad/s
    max_deflection_deg: float = 10.0
    min_speed: float = 15.0  # m/s, estimated speed above which control is active (off the rail)


class VerticalHold(Controller):
    def __init__(self, rocket, cfg, params: Params):
        self.p = params
        imus = [i for i in cfg.sensors.imu if i.enabled]
        if not imus:
            raise ValueError("vertical_hold needs at least one enabled IMU")
        self.imu = next((i for i in imus if i.name == params.imu), imus[0])
        self.est = StrapdownEstimator(self.imu)
        canards = next(f for f in rocket.aero.fins if f.is_canard)
        self.e_r = np.column_stack([np.cos(canards.angles), np.sin(canards.angles)])  # (n, 2)
        t = np.asarray(params.target_direction, dtype=float)
        self.target = t / np.linalg.norm(t)
        self.max_d = math.radians(params.max_deflection_deg)
        self.reset()

    def reset(self) -> None:
        self.est.reset()
        self.integral = np.zeros(3)
        self.active = False
        self.tilt = 0.0
        self.command = np.zeros(len(self.e_r))

    def update(self, inp: ControllerInput) -> np.ndarray:
        for s in inp.sensors.imu.get(self.imu.name, []):
            self.est.update_imu(s)
        for s in inp.sensors.baro:
            self.est.update_baro(s)
        est, p = self.est, self.p
        R = est.R
        e = np.cross(R[:, 0], self.target)
        self.tilt = math.atan2(float(np.linalg.norm(e)), float(R[:, 0] @ self.target))
        V = est.speed
        self.active = est.launched and V > p.min_speed and est.velocity[2] > 0.0
        if not self.active:
            self.command = np.zeros(len(self.e_r))
            return self.command
        if p.ki > 0:
            self.integral += e / self.rate_hz
            lim = self.max_d / p.ki
            n = float(np.linalg.norm(self.integral))
            if n > lim:
                self.integral *= lim / n
        w = est.omega
        u = R.T @ (p.kp * e + p.ki * self.integral) - (p.kd / V) * np.array([0.0, w[1], w[2]])
        d_roll = -p.k_roll * w[0] / V
        self.command = np.clip(d_roll - self.e_r @ u[1:], -self.max_d, self.max_d)
        return self.command

    def telemetry(self) -> dict:
        return {"active": float(self.active), "tilt_est_deg": math.degrees(self.tilt),
                "speed_est": self.est.speed, "q_est": self.est.dynamic_pressure,
                "integral_deg": math.degrees(float(np.linalg.norm(self.integral)) * self.p.ki),
                "calibrated": float(self.est.calibrated)}


CONTROLLER = VerticalHold
