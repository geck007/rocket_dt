"""Build the vehicle and simulation from a config: the single place where the .ork,
the motor files and the YAML are combined. Every override and cross-check is
recorded in ``RocketModel.notes``."""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from rocket_dt.aero import AeroModel
from rocket_dt.config import Config, ConfigError
from rocket_dt.control import Controller
from rocket_dt.io.motor import cross_check, load_eng, load_openmotor_csv, load_ric
from rocket_dt.io.ork import Component, MassComponent, OrkRocket, load_ork, resolve_positions, rocket_length
from rocket_dt.mass import MassModel, combine, component_mass
from rocket_dt.motor import SolidMotor
from rocket_dt.sim import RocketModel, Simulation

_PROTECTED = {"kind", "name", "id", "children", "parent", "x_top", "planform", "material"}


def apply_overrides(ork: OrkRocket, cfg: Config, notes: list[str]) -> None:
    for ov in cfg.rocket.overrides:
        c = ork.find(ov.component)
        for key, val in ov.set.items():
            if key == "mass" and not isinstance(c, MassComponent):
                old, c.override_mass = c.override_mass, float(val)
            elif key == "cg":
                old, c.override_cg = c.override_cg, float(val)
            elif key == "density":
                if c.material is None:
                    raise ConfigError(f"override {c.path}: component has no material")
                old, c.material.density = c.material.density, float(val)
            elif key in _PROTECTED or not hasattr(c, key):
                allowed = sorted({f.name for f in dataclasses.fields(c)} - _PROTECTED | {"mass", "cg", "density"})
                raise ConfigError(f"override {c.path}: cannot set '{key}'. Allowed: {allowed}")
            else:
                old = getattr(c, key)
                if isinstance(old, float):
                    val = float(val)
                elif type(old) is not type(val):
                    raise ConfigError(f"override {c.path}.{key}: expected {type(old).__name__}, got {val!r}")
                setattr(c, key, val)
            notes.append(f"OVERRIDE {c.path}.{key}: {old} -> {val}   [{ov.note}]")


def _motor_mount_aft_station(ork: OrkRocket) -> float:
    mounts = [c for c in ork.walk() if getattr(c, "motor_overhang", None) is not None]
    if len(mounts) != 1:
        raise ConfigError(f"Expected exactly one motor mount in the .ork, found {len(mounts)}")
    m = mounts[0]
    return m.x_top + m.length + m.motor_overhang


def _subtree(c: Component) -> list[Component]:
    return list(c.walk())


def build_rocket(cfg: Config) -> RocketModel:
    notes: list[str] = []
    ork = load_ork(cfg.rocket.ork)
    notes.append(f"ORK {cfg.rocket.ork.name}: '{ork.name}' ({ork.creator}, format {ork.ork_version})")
    notes.append("ORK: embedded simulation settings and motor configurations are ignored; "
                 "launch/environment come from the YAML config, the motor from the motor files")
    apply_overrides(ork, cfg, notes)
    resolve_positions(ork)

    # --- motor (one source per quantity; overlaps cross-checked)
    eng, ric, burn = load_eng(cfg.motor.eng), load_ric(cfg.motor.grains), load_openmotor_csv(cfg.motor.burn_data)
    notes += [f"MOTOR check: {n}" for n in cross_check(eng, ric, burn)]
    motor = SolidMotor(eng, ric, burn, aft_station=_motor_mount_aft_station(ork),
                       grain_stack_offset=cfg.motor.grain_stack_offset, grain_spacing=cfg.motor.grain_spacing,
                       hardware_cg_offset=cfg.motor.hardware_cg_offset, thrust_scale=cfg.disturbances.thrust_scale)
    notes.append(f"MOTOR {motor.designation}: {motor.total_impulse():.0f} Ns, burn {motor.burn_time:.2f} s, "
                 f"propellant {motor.propellant_mass:.3f} kg, hardware {motor.hardware_mass:.3f} kg, "
                 f"stations {motor.fore_station:.3f}-{motor.aft_station:.3f} m")

    # --- split airframe into body and fin can
    fc = cfg.rocket.fin_can
    fincan_roots = [ork.find(p) for p in fc.components] if fc else []
    fincan_ids = {id(c) for r in fincan_roots for c in _subtree(r)}
    all_comps = list(ork.walk())
    body = combine([component_mass(c) for c in all_comps if id(c) not in fincan_ids], "body")
    fincan = None
    if fincan_roots:
        fincan = combine([component_mass(c) for c in all_comps if id(c) in fincan_ids], "fin can")
        if np.linalg.norm(fincan.cg[1:]) > 1e-6:
            raise ConfigError("Fin-can CG is off the rocket axis; it cannot be modelled as a balanced rotor")
        geo_ixx = fincan.inertia[0, 0]
        if fc.roll_inertia is not None:
            fincan.inertia[0, 0] = fc.roll_inertia
            notes.append(f"OVERRIDE fin-can roll inertia: {geo_ixx:.5f} (geometry) -> {fc.roll_inertia:.5f} kg m^2")
        notes.append(f"FIN CAN ({'free on bearing' if fc.enabled else 'LOCKED'}): mass {fincan.mass:.3f} kg, "
                     f"CG station {-fincan.cg[0]:.3f} m, Ixx {fincan.inertia[0, 0]:.5f} kg m^2; "
                     f"components: {', '.join(r.path for r in fincan_roots)}")

    dd = cfg.disturbances
    # cg_offset_mm is the offset of the assembled vehicle at liftoff (what a balance measures); it belongs to
    # the airframe, so the airframe's own offset is larger by m_liftoff / m_airframe
    m_liftoff = body.mass + (fincan.mass if fincan else 0.0) + sum(e.mass for e in motor.mass_elements(0.0))
    mass = MassModel(body, fincan, motor,
                     body_cg_offset=tuple(1e-3 * v * m_liftoff / body.mass for v in dd.cg_offset_mm),
                     body_axis_tilt=tuple(math.radians(v) for v in dd.inertia_axis_tilt_deg))
    m0 = mass.at(0.0)
    notes.append(f"MASS dry airframe {body.mass + (fincan.mass if fincan else 0):.3f} kg, liftoff {m0.mass:.3f} kg, "
                 f"CG station {m0.cg_station:.4f} m, Ixx {m0.inertia[0, 0]:.4f}, Iyy {m0.inertia[1, 1]:.4f} kg m^2")
    if any(dd.cg_offset_mm):
        m1 = mass.at(motor.burn_time + 1.0)
        notes.append(f"BUILD vehicle CG offset {1e3 * np.linalg.norm(m0.cg[1:]):.2f} mm at liftoff, "
                     f"{1e3 * np.linalg.norm(m1.cg[1:]):.2f} mm after burnout (airframe without motor and fin can: "
                     f"{1e3 * np.linalg.norm(m0.cg[1:]) * m_liftoff / body.mass:.2f} mm)")

    # fin-can imbalance, in the fin-can frame (it turns with the fin can)
    imb_static = np.zeros(2)
    imb_couple = np.zeros(2)
    if fincan is not None:
        imb_static = fincan.mass * 1e-3 * np.array(dd.fincan_cg_offset_mm)  # kg m
        imb_couple = (fincan.inertia[1, 1] - fincan.inertia[0, 0]) * np.radians(dd.fincan_axis_tilt_deg)  # kg m^2
        if imb_static.any() or imb_couple.any():
            notes.append(f"BUILD fin-can imbalance: static {1e6 * np.linalg.norm(imb_static):.0f} g mm, "
                         f"couple {1e6 * np.linalg.norm(imb_couple):.0f} g mm^2")
    elif any(dd.fincan_cg_offset_mm) or any(dd.fincan_axis_tilt_deg):
        raise ConfigError("disturbances.fincan_*: the rocket has no fin can")

    aero = AeroModel(ork, {r.path for r in fincan_roots}, cfg.rocket.canards,
                     cfg.disturbances.fin_misalignment_deg, cfg.disturbances.drag_scale,
                     cfg.aero.canard_vortex, cfg.actuators.canards.hinge_from_root_le,
                     cfg.aero.canard_body_carryover, cfg.aero.body_crossflow_cd,
                     tuple(math.radians(v) for v in cfg.disturbances.nose_misalignment_deg))
    cna, cp = aero.static_cp(0.3)
    notes.append(f"AERO ref diameter {aero.ref_length:.4f} m, length {aero.length:.4f} m; at M 0.3: "
                 f"CN_alpha {cna:.2f}/rad, CP station {cp:.4f} m, static margin "
                 f"{(cp - m0.cg_station) / aero.ref_length:.2f} cal at liftoff")
    for f in aero.fins:
        notes.append(f"AERO fin set '{f.path}': {f.count} x {f.cross_section}, span {f.planform.span:.4f} m, "
                     f"area {f.planform.area * 1e4:.2f} cm^2, CP station {f.station_cp:.4f} m"
                     f"{', CANARDS' if f.is_canard else ''}{', on fin can' if f.on_fincan else ''}")
    if (cp - m0.cg_station) <= 0:
        notes.append("WARNING: rocket is statically UNSTABLE at liftoff (CP ahead of CG)")

    a, b = (math.radians(v) for v in cfg.disturbances.thrust_misalignment_deg)
    thrust_dir = np.array([math.cos(a) * math.cos(b), math.cos(a) * math.sin(b), -math.sin(a)])
    thrust_off = np.array([0.0, *cfg.disturbances.thrust_offset_m])

    return RocketModel(name=ork.name, mass=mass, aero=aero, motor=motor,
                       fincan_free=bool(fincan is not None and fc.enabled),
                       bearing=fc.bearing if fc else None, thrust_direction=thrust_dir, thrust_offset=thrust_off,
                       length=rocket_length(ork), notes=notes,
                       fincan_mass=fincan.mass if fincan is not None else 0.0,
                       fincan_station=-float(fincan.cg[0]) if fincan is not None else 0.0,
                       fincan_imbalance_static=imb_static, fincan_imbalance_couple=imb_couple)


def build_simulation(cfg: Config, controller: Controller | None = None, actuator=None) -> Simulation:
    """The rocket plus the controller named in ``cfg.controller`` (unless one is given)."""
    from rocket_dt.controllers import make_controller
    rocket = build_rocket(cfg)
    return Simulation(rocket, cfg, controller or make_controller(cfg, rocket), actuator)
