"""6-DOF flight simulation with a free-spinning fin can.

State vector (inertial frame ENU, body frame +x forward, origin at the nose tip):

    y[0:3]   position of the CG, m
    y[3:6]   velocity of the CG, m/s
    y[6:10]  attitude quaternion (body -> ENU), scalar first   (no gimbal lock)
    y[10:13] body angular velocity omega, rad/s
    y[13]    fin-can roll angle relative to the body, rad
    y[14]    fin-can roll rate relative to the body (Omega), rad/s

Rotational dynamics, rotor formulation (the fin can is an axisymmetric rotor
spinning about body x on a bearing; I includes the fin can as if locked):

    I w' + I_f Omega' e_x = M_ext - I' w - M_jet - w x (I w + I_f Omega e_x)
    I_f (w_x' + Omega')   = M_fincan_aero - T_bearing(Omega)

M_ext is the sum of all aerodynamic and thrust moments (the bearing torque is
internal and cancels). With the fin can locked, Omega = 0 and the second row drops.

Integration: fixed-step RK4. Inputs that come from discrete-time processes
(canard deflections, turbulence) are held constant within a step.

Each step: sensors sample the state (sensors.py), the controller runs at its own
rate on the samples that have become available (control.py), and the actuator
model advances the canards over the step under the latest hinge moments
(actuators.py). With ``simulation.pad_time_s`` > 0 the sensors, controller and
servos first run for that long with the rocket at rest on the pad (t < 0), so
flight software can calibrate; ignition is at t = 0.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field

import numpy as np

from rocket_dt import quaternion as Q
from rocket_dt.aero import AeroModel
from rocket_dt.atmosphere import ISAAtmosphere, gravity
from rocket_dt.actuators import make_actuator
from rocket_dt.config import BearingConfig, Config
from rocket_dt.control import Controller, ControllerInput, NoControl, TruthSnapshot
from rocket_dt.mass import MassModel
from rocket_dt.motor import SolidMotor
from rocket_dt.sensors import SensorSuite, SensorTruth
from rocket_dt.wind import Wind

E_X = np.array([1.0, 0.0, 0.0])


def _cross(a, b):
    """3-vector cross product (np.cross has large per-call overhead)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


@dataclass
class RocketModel:
    """Everything that defines the vehicle, built once from the config (see builder.py)."""
    name: str
    mass: MassModel
    aero: AeroModel
    motor: SolidMotor
    fincan_free: bool
    bearing: BearingConfig | None
    thrust_direction: np.ndarray  # unit vector, body frame
    thrust_offset: np.ndarray  # lateral nozzle offset (0, y, z), body frame
    length: float
    notes: list[str] = field(default_factory=list)

    def bearing_torque(self, omega_rel: float) -> float:
        """Friction torque acting on the fin can (opposes relative rotation)."""
        b = self.bearing
        return -(b.viscous_coeff * omega_rel + b.coulomb_torque * math.tanh(omega_rel / b.smoothing_rate))


@dataclass
class FlightResult:
    data: dict[str, np.ndarray]
    events: dict[str, float]
    rocket_name: str
    wall_time: float
    notes: list[str]
    sensors: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # full-rate sensor logs

    def __getitem__(self, k: str) -> np.ndarray:
        return self.data[k]

    def summary(self) -> str:
        d, ev = self.data, self.events
        i_apo = int(np.argmax(d["z"]))
        fast = d["airspeed"][:i_apo + 1] > 20.0  # AoA is meaningless near zero airspeed
        after_rail = d["t"][:i_apo + 1] >= ev.get("rail_exit", 0.0)
        lines = [f"Flight of '{self.rocket_name}' ({self.wall_time:.1f} s wall time)"]
        for k in ("liftoff", "rail_exit", "burnout", "apogee", "ground_hit"):
            if k in ev:
                lines.append(f"  {k:12s} t = {ev[k]:8.3f} s")
        lines += [
            f"  apogee           {d['z'][i_apo]:9.1f} m AGL ({d['z'][i_apo] / 0.3048:.0f} ft)",
            f"  max speed        {d['speed'].max():9.1f} m/s (Mach {d['mach'].max():.3f})",
            f"  rail exit speed  {ev.get('rail_exit_speed', float('nan')):9.2f} m/s",
            f"  max accel        {d['accel_axial'].max():9.1f} m/s^2 ({d['accel_axial'].max() / 9.81:.1f} g)",
            f"  max AoA (ascent, airspeed > 20 m/s) {np.degrees(d['aoa'][:i_apo + 1][fast].max()):6.2f} deg",
            f"  max tilt from vertical (rail exit to apogee) {np.degrees(d['zenith_angle'][:i_apo + 1][after_rail].max()):6.2f} deg",
            f"  min stab. margin {np.nanmin(d['stability_cal'][:i_apo + 1]):9.2f} cal (ascent, after rail)",
            f"  max |body roll|  {np.degrees(np.abs(d['p']).max()):9.1f} deg/s",
            f"  max |fin-can roll rate (inertial)| {np.degrees(np.abs(d['fincan_rate_abs']).max()):9.1f} deg/s",
            f"  landing point    E {d['x'][-1]:8.1f} m, N {d['y'][-1]:8.1f} m",
        ]
        return "\n".join(lines)

    def save_csv(self, path) -> None:
        keys = list(self.data)
        arr = np.column_stack([self.data[k] for k in keys])
        hdr = ",".join(keys)
        ev = "; ".join(f"{k}={v:.4f}" for k, v in self.events.items())
        np.savetxt(path, arr, delimiter=",", header=f"events: {ev}\n{hdr}", comments="# ")

    def save_sensor_csvs(self, out_dir) -> list:
        """One CSV per sensor (measurements and the matching true values)."""
        from pathlib import Path
        paths = []
        for name, cols in self.sensors.items():
            p = Path(out_dir) / f"sensor_{name}.csv"
            np.savetxt(p, np.column_stack(list(cols.values())), delimiter=",", header=",".join(cols), comments="")
            paths.append(p)
        return paths


class Simulation:
    def __init__(self, rocket: RocketModel, cfg: Config, controller: Controller | None = None,
                 actuator=None):
        self.rocket = rocket
        self.cfg = cfg
        self.atm = ISAAtmosphere(cfg.environment.atmosphere.temperature_offset_K,
                                 cfg.environment.atmosphere.sea_level_pressure)
        self.wind = Wind(cfg.environment.wind)
        self.launch_alt = cfg.launch.altitude_msl
        self.g = gravity(cfg.launch.latitude_deg, cfg.launch.altitude_msl)
        n_c = rocket.aero.n_canards
        self.controller = controller or NoControl(n_c)
        self.actuator = actuator or make_actuator(cfg.actuators.canards, n_c)
        self.sensors = SensorSuite(cfg.sensors, cfg.simulation.dt)
        self._hinge = np.zeros(n_c)
        desc = getattr(self.actuator, "describe", None)
        self.notes = rocket.notes + ([desc()] if desc else []) + self.sensors.describe()
        th = math.radians(cfg.launch.angle_from_vertical_deg)
        az = math.radians(cfg.launch.azimuth_deg)
        self.rail_dir = np.array([math.sin(th) * math.sin(az), math.sin(th) * math.cos(az), math.cos(th)])

    # ------------------------------------------------------------------ dynamics

    def _air(self, z: float):
        return self.atm.at(self.launch_alt + z)

    def _forces(self, t, y, delta, diagnostics=False):
        rk = self.rocket
        q = y[6:10]
        R = Q.to_matrix(q)
        ms = rk.mass.at(t)
        air = self._air(y[2])
        v_air = y[3:6] - self.wind.at(y[2])
        v_b = R.T @ v_air
        omega = y[10:13]
        cg_st = ms.cg_station
        aero = rk.aero.evaluate(v_b, omega, y[14], y[13], delta, air.density, air.speed_of_sound,
                                air.kinematic_viscosity, cg_st, diagnostics)
        self._hinge = aero.hinge_moment
        thrust = rk.motor.thrust(t)
        f_thrust = thrust * rk.thrust_direction
        r_noz = np.array([cg_st - rk.motor.nozzle_station, 0.0, 0.0]) + rk.thrust_offset - np.array([0.0, *ms.cg[1:]])
        cg_lat = np.array([0.0, *ms.cg[1:]])
        moment = aero.moment - _cross(cg_lat, aero.force) + _cross(r_noz, f_thrust)
        return R, ms, air, v_b, aero, thrust, aero.force + f_thrust, moment

    def deriv_flight(self, t, y, delta):
        rk = self.rocket
        R, ms, air, v_b, aero, thrust, F_b, M_b = self._forces(t, y, delta)
        omega = y[10:13]
        m = ms.mass
        acc = R @ F_b / m
        acc[2] -= self.g

        I = ms.inertia
        mflow = -ms.mass_dot
        if mflow > 0:
            le = ms.cg_station - rk.motor.nozzle_station
            M_b = M_b - mflow * np.array([0.5 * rk.motor.exit_radius ** 2 * omega[0], le * le * omega[1],
                                          le * le * omega[2]])
            M_b = M_b - ms.inertia_dot @ omega

        dy = np.empty(15)
        dy[0:3] = y[3:6]
        dy[3:6] = acc
        dy[6:10] = Q.derivative(y[6:10], omega)
        if rk.fincan_free:
            If = ms.fincan_roll_inertia
            Om = y[14]
            H = I @ omega + If * Om * E_X
            rhs1 = M_b - _cross(omega, H)
            rhs2 = aero.fincan_roll_moment + rk.bearing_torque(Om)
            A = np.zeros((4, 4))
            A[:3, :3] = I
            A[:3, 3] = If * E_X
            A[3, 0] = If
            A[3, 3] = If
            sol = np.linalg.solve(A, np.append(rhs1, rhs2))
            dy[10:13] = sol[:3]
            dy[13] = Om
            dy[14] = sol[3]
        else:
            dy[10:13] = np.linalg.solve(I, M_b - _cross(omega, I @ omega))
            dy[13] = 0.0
            dy[14] = 0.0
        return dy

    def deriv_rail(self, t, y, delta):
        """Motion constrained along the rail: no rotation, lateral loads taken by the rail."""
        rk = self.rocket
        ms = rk.mass.at(t)
        air = self._air(y[2])
        R = Q.to_matrix(y[6:10])
        v_b = R.T @ (y[3:6] - self.wind.at(y[2]))
        aero = rk.aero.evaluate(v_b, np.zeros(3), 0.0, 0.0, delta, air.density, air.speed_of_sound,
                                air.kinematic_viscosity, ms.cg_station)
        self._hinge = aero.hinge_moment
        f_along = rk.motor.thrust(t) + aero.force[0] - ms.mass * self.g * self.rail_dir[2]
        s = float(np.dot(y[0:3] - self._r0, self.rail_dir))
        a = f_along / ms.mass
        if s <= 1e-9 and a < 0 and np.dot(y[3:6], self.rail_dir) <= 0:
            a = 0.0
        dy = np.zeros(15)
        dy[0:3] = y[3:6]
        dy[3:6] = a * self.rail_dir
        return dy

    @staticmethod
    def _rk4(f, t, y, h, u):
        k1 = f(t, y, u)
        k2 = f(t + h / 2, y + h / 2 * k1, u)
        k3 = f(t + h / 2, y + h / 2 * k2, u)
        k4 = f(t + h, y + h * k3, u)
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # ------------------------------------------------------------------ run

    def initial_state(self) -> np.ndarray:
        ms = self.rocket.mass.at(0.0)
        q0 = Q.from_two_vectors(E_X, self.rail_dir)
        # CG starts one CG-distance up the rail from a pad at the origin (aft end on the pad)
        r0 = self.rail_dir * (self.rocket.length - ms.cg_station)
        y = np.zeros(15)
        y[0:3] = r0
        y[6:10] = q0
        self._r0 = r0.copy()
        return y

    def run(self) -> FlightResult:
        wall0 = _time.perf_counter()
        cfg = self.cfg.simulation
        dt = cfg.dt
        log_every = max(1, int(round(cfg.log_dt / dt)))
        ctrl_every = max(1, int(round(1.0 / (self.controller.rate_hz * dt))))
        self.controller.reset()
        self.actuator.reset()
        self.sensors.reset()
        self._hinge = np.zeros(self.rocket.aero.n_canards)
        y = self.initial_state()
        t = 0.0
        phase = "rail"
        events: dict[str, float] = {}
        log: dict[str, list] = {}
        delta = self.actuator.deflection.copy()
        command = delta.copy()
        self._command = command
        for step in range(-int(round(cfg.pad_time_s / dt)), 0):  # on the pad before ignition, nothing moves
            tp = step * dt
            if self.sensors.active:
                truth = self._sensor_truth(tp, y, delta, phase, False)
                truth.accel, truth.omega_dot = np.zeros(3), np.zeros(3)
                self.sensors.step(step, truth)
            if step % ctrl_every == 0:
                command = np.asarray(self.controller.update(
                    ControllerInput(tp, self.sensors.read(tp), self._snapshot(tp, y))), dtype=float)
                self._command = command
            delta = self.actuator.step(command, dt, np.zeros_like(command))
        step = 0
        max_steps = int(cfg.t_max / dt) + 1

        while step < max_steps:
            # discrete-time updates (zero-order hold over the step)
            if self.sensors.active:
                self.sensors.step(step, self._sensor_truth(t, y, delta, phase, self.sensors.needs_accel(step)))
            if step % ctrl_every == 0:
                command = np.asarray(self.controller.update(
                    ControllerInput(t, self.sensors.read(t), self._snapshot(t, y))), dtype=float)
                self._command = command
            delta = self.actuator.step(command, dt, self._hinge)
            airspeed = float(np.linalg.norm(y[3:6] - self.wind.at(y[2])))
            self.wind.update_turbulence(dt, max(y[2], 0.0), airspeed)

            if step % log_every == 0:
                self._log(log, t, y, delta, phase)

            f = self.deriv_rail if phase == "rail" else self.deriv_flight
            y_new = self._rk4(f, t, y, dt, delta)
            t_new = t + dt
            y_new[6:10] = Q.normalize(y_new[6:10])

            if phase == "rail":
                if "liftoff" not in events and np.dot(y_new[3:6], self.rail_dir) > 0:
                    events["liftoff"] = t
                s = float(np.dot(y_new[0:3] - self._r0, self.rail_dir))
                if s >= self.cfg.launch.rail_length:
                    s_old = float(np.dot(y[0:3] - self._r0, self.rail_dir))
                    events["rail_exit"] = t + dt * (self.cfg.launch.rail_length - s_old) / (s - s_old)
                    events["rail_exit_speed"] = float(np.linalg.norm(y_new[3:6]))
                    phase = "flight"
            else:
                if "burnout" not in events and t_new >= self.rocket.motor.burn_time:
                    events["burnout"] = self.rocket.motor.burn_time
                if "apogee" not in events and y[5] > 0 >= y_new[5]:
                    events["apogee"] = t + dt * y[5] / (y[5] - y_new[5])
                    if cfg.stop_at == "apogee":
                        y, t = y_new, t_new
                        self._log(log, t, y, delta, phase)
                        break
                if y_new[2] < 0 and y_new[5] < 0:
                    events["ground_hit"] = t + dt * y[2] / (y[2] - y_new[2])
                    y, t = y_new, t_new
                    self._log(log, t, y, delta, phase)
                    break
            y, t = y_new, t_new
            step += 1
            if not np.all(np.isfinite(y)):
                raise FloatingPointError(f"State became non-finite at t = {t:.3f} s")

        data = {k: np.asarray(v) for k, v in log.items()}
        return FlightResult(data, events, self.rocket.name, _time.perf_counter() - wall0, self.notes,
                            self.sensors.logs())

    # ------------------------------------------------------------------ helpers

    def _sensor_truth(self, t, y, delta, phase, need_accel: bool) -> SensorTruth:
        R = Q.to_matrix(y[6:10])
        air = self._air(y[2])
        va = float(np.linalg.norm(y[3:6] - self.wind.at(y[2])))
        truth = SensorTruth(t=t, position=y[0:3].copy(), velocity=y[3:6].copy(), R=R, omega=y[10:13].copy(),
                            cg_station=self.rocket.mass.at(t).cg_station, g=self.g,
                            dynamic_pressure=0.5 * air.density * va * va, air_at=self._air)
        if need_accel:
            dy = (self.deriv_rail if phase == "rail" else self.deriv_flight)(t, y, delta)
            truth.accel, truth.omega_dot = dy[3:6], dy[10:13]
        return truth

    def _snapshot(self, t, y) -> TruthSnapshot:
        air = self._air(y[2])
        va = float(np.linalg.norm(y[3:6] - self.wind.at(y[2])))
        return TruthSnapshot(t=t, position=y[0:3].copy(), velocity=y[3:6].copy(), quaternion=y[6:10].copy(),
                             omega=y[10:13].copy(), fincan_rate=float(y[14]), airspeed=va,
                             mach=va / air.speed_of_sound, dynamic_pressure=0.5 * air.density * va * va)

    def _log(self, log, t, y, delta, phase):
        rk = self.rocket
        if phase == "rail":
            R = Q.to_matrix(y[6:10])
            ms = rk.mass.at(t)
            air = self._air(y[2])
            v_b = R.T @ (y[3:6] - self.wind.at(y[2]))
            aero = rk.aero.evaluate(v_b, np.zeros(3), 0.0, 0.0, delta, air.density, air.speed_of_sound,
                                    air.kinematic_viscosity, ms.cg_station, diagnostics=True)
            thrust = rk.motor.thrust(t)
            F_b = aero.force + thrust * E_X
            acc_axial = (F_b[0] - ms.mass * self.g * self.rail_dir[2]) / ms.mass
        else:
            R, ms, air, v_b, aero, thrust, F_b, _ = self._forces(t, y, delta, diagnostics=True)
            acc_axial = F_b[0] / ms.mass
        d = aero.diag or {}
        body_x = R[:, 0]
        cp = d.get("cp_station", float("nan"))
        stab = (cp - ms.cg_station) / rk.aero.ref_length if phase == "flight" else float("nan")
        # roll angle about the body axis, measured from the vertical plane (defined when not vertical)
        z_b = R.T @ np.array([0.0, 0.0, 1.0])
        roll_angle = math.atan2(-z_b[1], z_b[2]) if math.hypot(z_b[1], z_b[2]) > 1e-6 else float("nan")
        row = {
            "t": t, "x": y[0], "y": y[1], "z": y[2], "vx": y[3], "vy": y[4], "vz": y[5],
            "speed": float(np.linalg.norm(y[3:6])), "airspeed": float(np.linalg.norm(v_b)),
            "qw": y[6], "qx": y[7], "qy": y[8], "qz": y[9],
            "p": y[10], "q": y[11], "r": y[12],
            "fincan_angle_rel": y[13], "fincan_rate_rel": y[14], "fincan_rate_abs": y[10] + y[14],
            "zenith_angle": math.acos(max(-1.0, min(1.0, body_x[2]))),
            # lean of the axis from vertical in the East-Up and North-Up planes (+ = nose towards East / North)
            "tilt_east": math.atan2(body_x[0], body_x[2]), "tilt_north": math.atan2(body_x[1], body_x[2]),
            "heading": math.atan2(body_x[0], body_x[1]),
            "roll_angle": roll_angle,
            "mass": ms.mass, "cg_station": ms.cg_station, "Ixx": ms.inertia[0, 0], "Iyy": ms.inertia[1, 1],
            "thrust": thrust, "accel_axial": acc_axial,
            "mach": d.get("mach", 0.0), "aoa": d.get("aoa", 0.0), "q_dyn": d.get("q", 0.0),
            "reynolds": d.get("reynolds", 0.0),
            "cd": d.get("cd", float("nan")), "cd_friction": d.get("cd_friction", float("nan")),
            "cd_pressure": d.get("cd_pressure", float("nan")), "cd_base": d.get("cd_base", float("nan")),
            "ca": d.get("ca", float("nan")), "cn_alpha": d.get("cn_alpha", float("nan")),
            "cp_station": cp, "stability_cal": stab,
            "drag_force": -aero.force[0], "fincan_aero_roll_moment": aero.fincan_roll_moment,
            "aero_fx": aero.force[0], "aero_fy": aero.force[1], "aero_fz": aero.force[2],
            "normal_force": math.hypot(aero.force[1], aero.force[2]),
            "aero_mx": aero.moment[0], "aero_my": aero.moment[1], "aero_mz": aero.moment[2],
            "canard_roll_moment": d.get("canard_roll_moment", 0.0),
            "vortex_roll_moment": d.get("vortex_roll_moment", 0.0),
            "bearing_torque": rk.bearing_torque(y[14]) if rk.fincan_free else 0.0,
            "wind_e": self.wind.at(y[2])[0], "wind_n": self.wind.at(y[2])[1],
            "air_density": air.density, "air_temperature": air.temperature,
        }
        each = d.get("vortex_roll_moment_each", np.zeros(len(delta)))
        hinge = aero.hinge_moment if aero.hinge_moment is not None else np.zeros(len(delta))
        for i, dlt in enumerate(delta):
            row[f"canard_{i}"] = dlt
            row[f"canard_cmd_{i}"] = self._command[i]
            row[f"hinge_moment_{i}"] = hinge[i]
            row[f"vortex_roll_{i}"] = each[i]
        telemetry = getattr(self.controller, "telemetry", None)
        if telemetry is not None:
            row.update({f"ctrl_{k}": v for k, v in telemetry().items()})
        for k, v in row.items():
            log.setdefault(k, []).append(float(v))
