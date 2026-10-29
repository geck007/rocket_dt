"""Runtime solid-motor model: thrust, burning grains and case hardware."""

from __future__ import annotations

import math

import numpy as np

from rocket_dt.io.motor import BurnData, EngFile, RicFile
from rocket_dt.mass import MassElement, _cylinder


class SolidMotor:
    """Motor placed in the airframe.

    The case hardware (eng loaded mass minus propellant) is a thick-walled cylinder
    between the grain OD and the case OD. Each grain is a hollow cylinder whose core
    radius grows so that its mass follows the openMotor per-grain mass history
    (end-burning is folded into the equivalent core radius).
    """

    def __init__(self, eng: EngFile, ric: RicFile, burn: BurnData, aft_station: float,
                 grain_stack_offset: float = 0.0, grain_spacing: float = 0.0, hardware_cg_offset: float = 0.0,
                 thrust_scale: float = 1.0):
        self.designation = eng.designation
        self.length = eng.length
        self.case_radius = eng.diameter / 2.0
        self.aft_station = aft_station
        self.fore_station = aft_station - eng.length
        self.nozzle_station = aft_station
        self.exit_radius = ric.exit_diameter / 2.0
        self.grains = ric.grains
        self._t = burn.time
        self._F = burn.thrust * thrust_scale
        self._mg = burn.grain_mass
        self.propellant_mass = float(burn.grain_mass[0].sum())
        self.hardware_mass = eng.total_mass - self.propellant_mass
        if self.hardware_mass <= 0:
            raise ValueError("Motor hardware mass (eng total - propellant) is not positive")
        nz = np.nonzero(self._F > 0)[0]
        self.burn_time = float(self._t[nz[-1] + 1]) if nz[-1] + 1 < len(self._t) else float(self._t[-1])

        stack = sum(g.length for g in self.grains) + grain_spacing * (len(self.grains) - 1)
        centre = self.fore_station + eng.length / 2.0 + grain_stack_offset
        x = centre - stack / 2.0
        self.grain_stations = []  # (top, length)
        for g in self.grains:
            self.grain_stations.append((x, g.length))
            x += g.length + grain_spacing
        if self.grain_stations[0][0] < self.fore_station - 1e-9 or x - grain_spacing > aft_station + 1e-9:
            raise ValueError("Grain stack does not fit in the motor case with the configured offset")
        r_in_hw = max(g.outer_diameter for g in self.grains) / 2.0
        self._hardware = _cylinder("motor hardware", self.hardware_mass,
                                   self.fore_station + eng.length / 2.0 + hardware_cg_offset,
                                   eng.length, self.case_radius, r_in_hw)

    def time_grid(self) -> np.ndarray:
        return self._t

    def thrust(self, t: float) -> float:
        return float(np.interp(t, self._t, self._F, left=0.0, right=0.0))

    def grain_masses(self, t: float) -> np.ndarray:
        return np.array([np.interp(t, self._t, self._mg[:, i]) for i in range(self._mg.shape[1])])

    def mass(self, t: float) -> float:
        return self.hardware_mass + float(self.grain_masses(t).sum())

    def mass_elements(self, t: float) -> list[MassElement]:
        out = [self._hardware]
        for g, (top, L), m in zip(self.grains, self.grain_stations, self.grain_masses(t)):
            R = g.outer_diameter / 2.0
            rc = math.sqrt(max(R * R - m / (g.density * math.pi * L), 0.0))
            out.append(_cylinder("grain", float(m), top + L / 2.0, L, R, rc))
        return out

    def total_impulse(self) -> float:
        return float(np.trapezoid(self._F, self._t))
