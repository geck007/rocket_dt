"""Mass properties.

Body frame used throughout the simulation: origin at the nose tip, +x towards the
nose (i.e. forward), y/z completing a right-handed set. A point at OpenRocket
station X (metres aft of the nose tip) therefore has body x = -X.

Every component is converted to one or more :class:`MassElement` (mass, CG, inertia
tensor about its own CG in body axes); groups are summed with the parallel-axis
theorem. Time dependence comes only from the motor grains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rocket_dt.geometry import body_radius, fin_planform
from rocket_dt.io.ork import (CROSS_SECTION_VOLUME_FACTOR, Component, FinSet, MassComponent, RingComponent,
                              SymmetricComponent)


@dataclass
class MassElement:
    name: str
    mass: float
    cg: np.ndarray  # body frame (x = -station)
    inertia: np.ndarray  # 3x3 about own CG, body axes


def combine(elements: list[MassElement], name: str = "combined") -> MassElement:
    m = sum(e.mass for e in elements)
    if m <= 0:
        return MassElement(name, 0.0, np.zeros(3), np.zeros((3, 3)))
    cg = sum(e.mass * e.cg for e in elements) / m
    inertia = np.zeros((3, 3))
    for e in elements:
        d = e.cg - cg
        inertia += e.inertia + e.mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return MassElement(name, m, cg, inertia)


def _station_point(station: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.array([-station, y, z])


def _cylinder(name: str, mass: float, station_center: float, length: float, r_out: float, r_in: float = 0.0,
              y: float = 0.0) -> MassElement:
    ixx = mass * (r_out ** 2 + r_in ** 2) / 2.0
    iyy = mass * ((r_out ** 2 + r_in ** 2) / 4.0 + length ** 2 / 12.0)
    return MassElement(name, mass, _station_point(station_center, y), np.diag([ixx, iyy, iyy]))


def _shell_of_revolution(name: str, density: float, x0: float, length: float, r_out_fn, thickness: float,
                         filled: bool, n: int = 400) -> MassElement:
    """Thick axisymmetric shell from station x0 over `length`, outer radius r_out_fn(local x)."""
    edges = np.linspace(0.0, length, n + 1)
    xm = 0.5 * (edges[1:] + edges[:-1])
    dx = np.diff(edges)
    ro = r_out_fn(xm)
    slope = np.gradient(r_out_fn(edges), edges)
    slope = 0.5 * (slope[1:] + slope[:-1])
    # wall thickness is measured normal to the surface (as OpenRocket does)
    ri = np.zeros_like(ro) if filled else np.maximum(ro - thickness * np.sqrt(1.0 + slope ** 2), 0.0)
    dm = density * np.pi * (ro ** 2 - ri ** 2) * dx
    m = float(dm.sum())
    if m <= 0:
        return MassElement(name, 0.0, _station_point(x0), np.zeros((3, 3)))
    xs = x0 + xm
    xc = float(np.sum(dm * xs) / m)
    ixx = float(np.sum(dm * (ro ** 2 + ri ** 2) / 2.0))
    iyy = float(np.sum(dm * ((ro ** 2 + ri ** 2) / 4.0 + dx ** 2 / 12.0 + (xs - xc) ** 2)))
    return MassElement(name, m, _station_point(xc), np.diag([ixx, iyy, iyy]))


def _apply_component_overrides(c: Component, e: MassElement) -> MassElement:
    if c.override_mass is not None:
        s = c.override_mass / e.mass if e.mass > 0 else 0.0
        e = MassElement(e.name, c.override_mass, e.cg.copy(), e.inertia * s)
    if c.override_cg is not None:
        e = MassElement(e.name, e.mass, _station_point(c.x_top + c.override_cg, e.cg[1], e.cg[2]), e.inertia)
    return e


def component_mass(c: Component) -> MassElement:
    """Mass element of one component (excluding its children)."""
    if isinstance(c, SymmetricComponent):
        rho = c.material.density
        parts = [_shell_of_revolution(c.path, rho, c.x_top, c.length, lambda xl: body_radius(c, xl),
                                      c.thickness, c.filled)]
        for sh, x_start in ((c.fore_shoulder, c.x_top - c.fore_shoulder.length),
                            (c.aft_shoulder, c.x_top + c.length)):
            if sh.length > 0 and sh.radius > 0:
                ri = max(sh.radius - sh.thickness, 0.0)
                m = rho * math.pi * (sh.radius ** 2 - ri ** 2) * sh.length
                parts.append(_cylinder("shoulder", m, x_start + sh.length / 2, sh.length, sh.radius, ri))
                if sh.capped:
                    xc = x_start + (sh.length - sh.thickness / 2 if x_start >= c.x_top else sh.thickness / 2)
                    mc = rho * math.pi * ri ** 2 * sh.thickness
                    parts.append(_cylinder("cap", mc, xc, sh.thickness, ri))
        e = combine(parts, c.path)
    elif isinstance(c, FinSet):
        e = combine(fin_elements(c), c.path)
    elif isinstance(c, MassComponent):
        e = _cylinder(c.path, c.mass, c.x_top + c.length / 2, c.length, c.radius, 0.0, c.radial_position)
    elif isinstance(c, RingComponent):
        m = c.material.density * math.pi * (c.outer_radius ** 2 - c.inner_radius ** 2) * c.length
        e = _cylinder(c.path, m, c.x_top + c.length / 2, c.length, c.outer_radius, c.inner_radius, c.radial_position)
    else:
        raise TypeError(f"No mass model for {c.kind}")
    return _apply_component_overrides(c, e)


def fin_root_radius(fs: FinSet) -> float:
    p = fs.parent
    if not isinstance(p, SymmetricComponent) or p.kind != "bodytube":
        raise ValueError(f"Fin set '{fs.path}' must be attached to a body tube")
    return p.aft_radius


def fin_angles(fs: FinSet) -> np.ndarray:
    return fs.angle_offset + 2 * np.pi * np.arange(fs.fin_count) / fs.fin_count


def fin_elements(fs: FinSet) -> list[MassElement]:
    """One element per fin, modelled as a uniform plate."""
    pf = fin_planform(fs.planform)
    vol_factor = CROSS_SECTION_VOLUME_FACTOR.get(fs.cross_section)
    if vol_factor is None:
        raise ValueError(f"'{fs.path}': unknown cross section '{fs.cross_section}'")
    rho_area = fs.material.density * fs.thickness * vol_factor  # kg/m^2
    m = rho_area * pf.area
    sxx, syy, sxy = pf.second_moments
    t2 = m * fs.thickness ** 2 / 12.0
    # local axes: a = aft along the body, b = radial (span), c = fin-normal
    i_local = np.array([[rho_area * syy + t2, -rho_area * sxy, 0.0],
                        [-rho_area * sxy, rho_area * sxx + t2, 0.0],
                        [0.0, 0.0, rho_area * (sxx + syy)]])
    r_t = fin_root_radius(fs)
    out = []
    for k, phi in enumerate(fin_angles(fs)):
        e_r = np.array([0.0, math.cos(phi), math.sin(phi)])
        n = np.array([0.0, -math.sin(phi), math.cos(phi)])
        Q = np.column_stack([[-1.0, 0.0, 0.0], e_r, n])  # local -> body
        cg = _station_point(fs.x_top + pf.centroid_x) + (r_t + pf.mac_span) * e_r
        out.append(MassElement(f"{fs.path}[{k}]", m, cg, Q @ i_local @ Q.T))
    return out


# ----------------------------------------------------------------------------- time-varying model


@dataclass
class MassState:
    mass: float
    cg: np.ndarray  # body frame
    inertia: np.ndarray  # total system inertia about system CG (fin can included, as if locked)
    inertia_dot: np.ndarray
    mass_dot: float  # <= 0 while burning
    fincan_roll_inertia: float  # 0 when there is no free fin can

    @property
    def cg_station(self) -> float:
        return -float(self.cg[0])


class MassModel:
    """Tabulated mass properties versus time, linearly interpolated."""

    def __init__(self, static_body: MassElement, fincan: MassElement | None, motor,
                 body_cg_offset=(0.0, 0.0), body_axis_tilt=(0.0, 0.0)):
        """body_cg_offset (m, body y, z) and body_axis_tilt (rad, about body y, z) are build errors of the
        airframe (``static_body``). The motor stays on the axis, so the offset of the whole vehicle's CG,
        body_cg_offset * m_body / m(t), grows as the propellant burns; both errors also produce
        products of inertia."""
        a, b = body_axis_tilt
        ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
        rot = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]]) @ np.array([[cb, -sb, 0.0], [sb, cb, 0.0],
                                                                                      [0.0, 0.0, 1.0]])
        static_body = MassElement(static_body.name, static_body.mass,
                                  static_body.cg + np.array([0.0, *body_cg_offset]),
                                  rot @ static_body.inertia @ rot.T)
        self.static_body = static_body
        self.fincan = fincan
        self.motor = motor
        t = np.unique(np.concatenate([motor.time_grid(), [motor.burn_time + 1e-6]]))
        self.t = t
        n = len(t)
        self._m = np.empty(n)
        self._cg = np.empty((n, 3))
        self._I = np.empty((n, 3, 3))
        for k, tk in enumerate(t):
            parts = [static_body, *motor.mass_elements(tk)]
            if fincan is not None:
                parts.append(fincan)
            tot = combine(parts)
            self._m[k] = tot.mass
            self._cg[k] = tot.cg
            self._I[k] = tot.inertia
        dt = np.diff(t)
        self._mdot = np.append(np.diff(self._m) / dt, 0.0)
        self._Idot = np.concatenate([np.diff(self._I, axis=0) / dt[:, None, None], np.zeros((1, 3, 3))])
        self.fincan_roll_inertia = float(fincan.inertia[0, 0]) if fincan is not None else 0.0

    def at(self, t: float) -> MassState:
        t_tab = self.t
        if t >= t_tab[-1]:
            k, w = len(t_tab) - 2, 1.0
        elif t <= t_tab[0]:
            k, w = 0, 0.0
        else:
            k = int(np.searchsorted(t_tab, t, side="right") - 1)
            w = (t - t_tab[k]) / (t_tab[k + 1] - t_tab[k])
        m = self._m[k] + w * (self._m[k + 1] - self._m[k])
        cg = self._cg[k] + w * (self._cg[k + 1] - self._cg[k])
        inertia = self._I[k] + w * (self._I[k + 1] - self._I[k])
        burning = t < t_tab[-1]
        mdot = self._mdot[k] if burning else 0.0
        idot = self._Idot[k] if burning else np.zeros((3, 3))
        return MassState(m, cg, inertia, idot, mdot, self.fincan_roll_inertia)
