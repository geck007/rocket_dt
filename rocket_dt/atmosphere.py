"""International Standard Atmosphere (troposphere + lower stratosphere) and gravity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_AIR = 287.05287  # J/(kg K)
GAMMA_AIR = 1.4
G0 = 9.80665


@dataclass(frozen=True)
class AirState:
    temperature: float  # K
    pressure: float  # Pa
    density: float  # kg/m^3
    speed_of_sound: float  # m/s
    kinematic_viscosity: float  # m^2/s


class ISAAtmosphere:
    """ISA with an optional sea-level temperature offset (hot/cold day)."""

    def __init__(self, temperature_offset: float = 0.0, sea_level_pressure: float = 101325.0):
        self.dT = temperature_offset
        self.p0 = sea_level_pressure

    def at(self, altitude_msl: float) -> AirState:
        h = max(altitude_msl, -500.0)
        t0 = 288.15 + self.dT
        if h <= 11000.0:
            T = t0 - 0.0065 * h
            p = self.p0 * (T / t0) ** (G0 / (0.0065 * R_AIR))
        else:
            t11 = t0 - 0.0065 * 11000.0
            p11 = self.p0 * (t11 / t0) ** (G0 / (0.0065 * R_AIR))
            T = t11
            p = p11 * np.exp(-G0 * (h - 11000.0) / (R_AIR * t11))
        rho = p / (R_AIR * T)
        a = np.sqrt(GAMMA_AIR * R_AIR * T)
        mu = 1.458e-6 * T ** 1.5 / (T + 110.4)  # Sutherland
        return AirState(T, p, rho, a, mu / rho)


def gravity(latitude_deg: float, altitude_msl: float) -> float:
    """International gravity formula with free-air correction (m/s^2)."""
    phi = np.radians(latitude_deg)
    g_surface = 9.780327 * (1 + 0.0053024 * np.sin(phi) ** 2 - 0.0000058 * np.sin(2 * phi) ** 2)
    return g_surface - 3.086e-6 * altitude_msl
