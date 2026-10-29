"""Motor input files: RASP .eng, openMotor .ric and openMotor CSV export.

Each physical quantity is taken from exactly one file:

  * thrust(t) and per-grain propellant mass(t) -> openMotor CSV
    (same simulation, so thrust and mass flow are mutually consistent)
  * grain geometry and order (fore -> aft)     -> openMotor .ric
  * loaded mass, case diameter and length      -> .eng header

Where files overlap (eng thrust curve, eng propellant mass, ric grain volume) the
values are only *cross-checked*; a mismatch raises instead of silently picking one.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


class MotorDataError(ValueError):
    pass


@dataclass
class EngFile:
    designation: str
    diameter: float  # m
    length: float  # m
    propellant_mass: float  # kg
    total_mass: float  # kg
    time: np.ndarray
    thrust: np.ndarray


def load_eng(path: str | Path) -> EngFile:
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(";")]
    head = lines[0].split()
    t, f = [], []
    for ln in lines[1:]:
        a, b = ln.split()[:2]
        t.append(float(a))
        f.append(float(b))
    return EngFile(designation=head[0], diameter=float(head[1]) / 1000.0, length=float(head[2]) / 1000.0,
                   propellant_mass=float(head[4]), total_mass=float(head[5]),
                   time=np.array(t), thrust=np.array(f))


@dataclass
class Grain:
    type: str
    outer_diameter: float
    core_diameter: float  # initial (BATES/Finocyl core)
    length: float
    density: float
    raw: dict

    def initial_volume(self) -> float:
        v = math.pi / 4 * (self.outer_diameter ** 2 - self.core_diameter ** 2) * self.length
        if self.type == "Finocyl":
            p = self.raw
            v -= p["numFins"] * p["finLength"] * p["finWidth"] * self.length
        return v


@dataclass
class RicFile:
    grains: list[Grain]  # fore -> aft
    throat_diameter: float
    exit_diameter: float
    propellant_name: str


class _RicLoader(yaml.SafeLoader):
    pass


# openMotor tags its files with python-object YAML tags; read them as plain data.
_RicLoader.add_multi_constructor("tag:yaml.org,2002:python/",
                                 lambda loader, suffix, node: loader.construct_sequence(node)
                                 if isinstance(node, yaml.SequenceNode) else None)


def load_ric(path: str | Path) -> RicFile:
    d = yaml.load(Path(path).read_text(), Loader=_RicLoader)["data"]
    rho = float(d["propellant"]["density"])
    grains = []
    for g in d["grains"]:
        p = g["properties"]
        if g["type"] not in ("BATES", "Finocyl"):
            raise MotorDataError(f"Grain type {g['type']} not supported (BATES, Finocyl)")
        grains.append(Grain(type=g["type"], outer_diameter=float(p["diameter"]),
                            core_diameter=float(p["coreDiameter"]), length=float(p["length"]),
                            density=rho, raw=p))
    nz = d["nozzle"]
    return RicFile(grains=grains, throat_diameter=float(nz["throat"]), exit_diameter=float(nz["exit"]),
                   propellant_name=d["propellant"]["name"])


@dataclass
class BurnData:
    time: np.ndarray
    thrust: np.ndarray
    grain_mass: np.ndarray  # shape (n_times, n_grains), fore -> aft


def load_openmotor_csv(path: str | Path) -> BurnData:
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    hdr = rows[0]
    data = np.array([[float(v) for v in r] for r in rows[1:] if r])
    try:
        it = hdr.index("Time(s)")
        iF = hdr.index("Thrust(N)")
    except ValueError as e:
        raise MotorDataError(f"{path}: not an openMotor CSV export ({e})")
    gcols = [i for i, h in enumerate(hdr) if h.startswith("Propellant Mass(G")]
    if not gcols:
        raise MotorDataError(f"{path}: no 'Propellant Mass(G..)' columns")
    return BurnData(time=data[:, it], thrust=data[:, iF], grain_mass=data[:, gcols])


def cross_check(eng: EngFile, ric: RicFile, burn: BurnData, rel_tol: float = 0.01) -> list[str]:
    """Verify the overlapping data in the three files agree; raise on disagreement.

    Returns human-readable notes describing what was checked.
    """
    notes = []
    m0 = burn.grain_mass[0]
    if len(m0) != len(ric.grains):
        raise MotorDataError(f"openMotor CSV has {len(m0)} grains but .ric has {len(ric.grains)}")
    for i, (g, m) in enumerate(zip(ric.grains, m0)):
        mv = g.initial_volume() * g.density
        if abs(mv - m) > rel_tol * m:
            raise MotorDataError(f"Grain {i + 1}: .ric geometry gives {mv:.4f} kg but CSV says {m:.4f} kg")
    notes.append(f"grain masses: .ric geometry matches CSV within {rel_tol:.0%}")

    if abs(eng.propellant_mass - m0.sum()) > rel_tol * m0.sum():
        raise MotorDataError(f".eng propellant mass {eng.propellant_mass} kg != CSV {m0.sum():.4f} kg")
    notes.append(".eng propellant mass matches CSV")

    f_eng = np.interp(burn.time, eng.time, eng.thrust, left=0.0, right=0.0)
    imp_eng = np.trapezoid(eng.thrust, eng.time)
    imp_csv = np.trapezoid(burn.thrust, burn.time)
    if abs(imp_eng - imp_csv) > rel_tol * imp_csv:
        raise MotorDataError(f".eng impulse {imp_eng:.1f} Ns != CSV impulse {imp_csv:.1f} Ns")
    dev = np.max(np.abs(f_eng - burn.thrust)[1:-1]) / np.max(burn.thrust)
    if dev > 0.02:
        raise MotorDataError(f".eng thrust curve deviates from CSV by {dev:.1%} of peak thrust")
    notes.append(f".eng thrust curve matches CSV (impulse {imp_csv:.1f} Ns)")

    max_od = max(g.outer_diameter for g in ric.grains)
    if max_od > eng.diameter:
        raise MotorDataError(f"Grain OD {max_od} m exceeds case diameter {eng.diameter} m")
    stack = sum(g.length for g in ric.grains)
    if stack > eng.length:
        raise MotorDataError(f"Grain stack {stack} m longer than motor {eng.length} m")
    return notes
