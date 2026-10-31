"""Airframe constants for controller design (linear Barrowman model, dry rocket).

A controller's ``Params`` that inherits ``AirframeParams`` gets these fields; any left
unset in the config (None) are filled from the simulated rocket by ``make_controller``
before the controller is built, so ``params.CNa_tot`` etc. are always numbers.

They come from the same model the simulation flies, so a controller using them knows
the airframe better than real flight software would: on a real rocket they would be
estimates (OpenRocket, wind tunnel), off by 10-20 %. They are also simplified:

* one Mach number (``model_mach``) for the lift slopes;
* the dry (burnout) mass, CG and inertia; in the boost the CG is further aft, so the
  real arms X are shorter;
* linear lift only: no body crossflow lift, no canard downwash on the fins, no
  roll damping.

Stations are measured aft from the nose tip; arms X = S_cp - S_cg are positive aft of
the CG. Normal-force slopes are per rad, referenced to Sref. A set of N fins counts as
N/2 fins in its plane with body interference (Barrowman), which for four fins is the
two fins that lift in a pitch or yaw plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rocket_dt.config import ConfigError


@dataclass
class AirframeParams:
    model_mach: float = 0.3  # Mach number for the lift slopes below
    Sref: float | None = None  # m^2, reference area (body cross-section)
    CNa_tot: float | None = None  # sum of CNa of all components except the canards (nose, transitions, fins)
    sum_CNaX: float | None = None  # m, sum of CNa_i X_i over the same components, X_i = S_cp_i - S_cg
    CNa_can: float | None = None  # canard set alone
    Y_cp_can: float | None = None  # m, radial arm of a canard's CP (MAC) from the rocket's axis
    X_cp: float | None = None  # m, S_cp - S_cg of the rocket without canards (= sum_CNaX / CNa_tot)
    X_cp_can: float | None = None  # m, S_cp - S_cg of the canards (negative: ahead of the CG)
    mass: float | None = None  # kg, dry
    inertia: list[list[float]] | None = None  # kg m^2, dry, about the CG, body axes (x along the rocket)


def airframe_constants(rocket, mach: float = 0.3) -> dict:
    """The AirframeParams values (except model_mach) of a built rocket (sim.RocketModel)."""
    aero = rocket.aero
    beta = math.sqrt(max(1.0 - mach * mach, 0.05))
    dry = rocket.mass.at(rocket.motor.burn_time + 1.0)
    s_cg = dry.cg_station
    cna = list(aero.body_cna) + [f.cn_alpha_set(beta) for f in aero.fins if not f.is_canard]
    x = list(aero.body_cp_station - s_cg) + [f.station_cp - s_cg for f in aero.fins if not f.is_canard]
    cna_tot = float(sum(cna))
    sum_cnax = float(np.dot(cna, x))
    out = {"Sref": aero.ref_area, "CNa_tot": cna_tot, "sum_CNaX": sum_cnax, "X_cp": sum_cnax / cna_tot,
           "CNa_can": None, "Y_cp_can": None, "X_cp_can": None,
           "mass": dry.mass, "inertia": dry.inertia.copy()}
    can = next((f for f in aero.fins if f.is_canard), None)
    if can is not None:
        out |= {"CNa_can": can.cn_alpha_set(beta), "Y_cp_can": can.root_radius + can.planform.mac_span,
                "X_cp_can": can.station_cp - s_cg}
    return out


def fill_airframe(params: AirframeParams, rocket) -> None:
    """Set the fields left at None from the rocket; inertia becomes a 3x3 array."""
    for k, v in airframe_constants(rocket, params.model_mach).items():
        if getattr(params, k) is None:
            setattr(params, k, v)
    if params.inertia is not None:
        params.inertia = np.asarray(params.inertia, dtype=float)
        if params.inertia.shape != (3, 3):
            raise ConfigError(f"controller.params.inertia must be 3x3, got shape {params.inertia.shape}")
