"""Airframe constants for controller design, derived from the rocket's design files.

A controller's ``Params`` that inherits ``AirframeParams`` gets these fields; any left
unset in the config (None) are filled by ``make_controller`` before the controller is
built, so ``params.CNa_tot`` etc. are always numbers and follow the .ork (and motor)
files when those change.

They are what the team knows on the ground: the *nominal* rocket, i.e. the .ork, the
motor files and the config's ``rocket`` overrides, without ``disturbances`` (CG offset,
misalignments, fin-can imbalance, thrust and drag scale; Monte Carlo disperses those).
They come from the simulation's own model, so they are still better than a real
rocket's estimates (OpenRocket, wind tunnel; off by 10-20 %), and they are simplified:

* one Mach number (``model_mach``) for the lift slopes and the drag;
* linear lift only: no body crossflow lift, no canard downwash on the fins, no
  roll damping;
* the arms X are taken from the dry (burnout) CG; in the boost the CG is further aft,
  so the arms are shorter (``S_cg_wet`` is the liftoff CG).

Stations S are measured aft from the nose tip; arms X = S_cp - S_cg are positive aft of
the CG. Normal-force slopes are per rad, referenced to Sref. A set of N fins counts as
N/2 fins in its plane with body interference (Barrowman), which for four fins is the
two fins that lift in a pitch or yaw plane. Inertias are about the CG in body axes
(x along the rocket), with the fin can included as if locked to the body.

With these, the linearised pitch (or yaw) dynamics about the CG are

    I_yy theta'' = -qbar Sref sum_CNaX alpha - qbar Sref (sum_CNaX2 / V) theta' + canard moment

(restoring moment from the angle of attack, damping from the pitch rate: a component at
arm X sees an extra angle of attack theta' X / V).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from rocket_dt.config import ConfigError, DisturbanceConfig

SOUND_SPEED = 340.3  # m/s, sea level standard (for the drag's Reynolds number)
NU_SEA_LEVEL = 1.46e-5  # m^2/s


@dataclass
class AirframeParams:
    model_mach: float = 0.3  # Mach number for the lift slopes and CD0 below
    # geometry
    Sref: float | None = None  # m^2, reference area (body cross-section)
    d_ref: float | None = None  # m, reference diameter
    length: float | None = None  # m, nose tip to tail
    # aerodynamics (all components except the canards, unless named _can)
    CNa_tot: float | None = None  # sum of CNa_i (nose, transitions, tail fins)
    sum_CNaX: float | None = None  # m, sum of CNa_i X_i, X_i = S_cp_i - S_cg
    sum_CNaX2: float | None = None  # m^2, sum of CNa_i X_i^2 (pitch damping)
    X_cp: float | None = None  # m, S_cp - S_cg (= sum_CNaX / CNa_tot)
    CNa_can: float | None = None  # canard set alone
    X_cp_can: float | None = None  # m, S_cp - S_cg of the canards (negative: ahead of the CG)
    Y_cp_can: float | None = None  # m, radial arm of a canard's CP (MAC) from the rocket's axis
    canard_angles_deg: list[float] | None = None  # roll position of each canard, from body +y towards +z
    CD0: float | None = None  # zero-lift drag coefficient (coast, no thrust), on Sref
    # mass: dry = after burnout, wet = at liftoff
    mass: float | None = None  # kg, dry
    S_cg: float | None = None  # m, dry CG station
    inertia: list[list[float]] | None = None  # kg m^2, dry, 3x3
    mass_wet: float | None = None  # kg
    S_cg_wet: float | None = None  # m
    inertia_wet: list[list[float]] | None = None  # kg m^2, 3x3
    inertia_fincan_roll: float | None = None  # kg m^2, roll inertia of the fin can alone (0 without one)
    burn_time: float | None = None  # s


def nominal_rocket(cfg, rocket=None):
    """The rocket as designed: ``cfg`` without its disturbances (``rocket`` is reused if there are none)."""
    if rocket is not None and cfg.disturbances == DisturbanceConfig():
        return rocket
    from rocket_dt.builder import build_rocket
    return build_rocket(dataclasses.replace(cfg, disturbances=DisturbanceConfig()))


def airframe_constants(rocket, mach: float = 0.3) -> dict:
    """The AirframeParams values (except model_mach) of a built rocket (sim.RocketModel)."""
    aero = rocket.aero
    beta = math.sqrt(max(1.0 - mach * mach, 0.05))
    dry, wet = rocket.mass.at(rocket.motor.burn_time + 1.0), rocket.mass.at(0.0)
    s_cg = dry.cg_station
    cna = np.array(list(aero.body_cna) + [f.cn_alpha_set(beta) for f in aero.fins if not f.is_canard])
    x = np.array(list(aero.body_cp_station) + [f.station_cp for f in aero.fins if not f.is_canard]) - s_cg
    V = mach * SOUND_SPEED
    out = {"Sref": aero.ref_area, "d_ref": aero.ref_length, "length": aero.length,
           "CNa_tot": float(cna.sum()), "sum_CNaX": float(cna @ x), "sum_CNaX2": float(cna @ x ** 2),
           "X_cp": float(cna @ x / cna.sum()),
           "CNa_can": None, "X_cp_can": None, "Y_cp_can": None, "canard_angles_deg": None,
           "CD0": aero.drag_breakdown(mach, V * aero.length / NU_SEA_LEVEL)["total"],
           "mass": float(dry.mass), "S_cg": s_cg, "inertia": dry.inertia.copy(),
           "mass_wet": float(wet.mass), "S_cg_wet": wet.cg_station, "inertia_wet": wet.inertia.copy(),
           "inertia_fincan_roll": float(dry.fincan_roll_inertia), "burn_time": rocket.motor.burn_time}
    can = next((f for f in aero.fins if f.is_canard), None)
    if can is not None:
        out |= {"CNa_can": can.cn_alpha_set(beta), "X_cp_can": can.station_cp - s_cg,
                "Y_cp_can": can.root_radius + can.planform.mac_span,
                "canard_angles_deg": [math.degrees(a) for a in can.angles]}
    return out


def fill_airframe(params: AirframeParams, cfg, rocket=None) -> None:
    """Set the fields left at None from the nominal rocket; the inertias become 3x3 arrays."""
    missing = [f.name for f in dataclasses.fields(AirframeParams) if getattr(params, f.name) is None]
    if missing:
        values = airframe_constants(nominal_rocket(cfg, rocket), params.model_mach)
        for k in missing:
            setattr(params, k, values[k])
    for k in ("inertia", "inertia_wet"):
        if getattr(params, k) is not None:
            a = np.asarray(getattr(params, k), dtype=float)
            if a.shape != (3, 3):
                raise ConfigError(f"controller.params.{k} must be 3x3, got shape {a.shape}")
            setattr(params, k, a)
