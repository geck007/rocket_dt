"""Geometry helpers: body-of-revolution profiles and fin planform properties."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rocket_dt.io.ork import SymmetricComponent


def _shape_radius(shape: str, x: np.ndarray, length: float, radius: float, param: float) -> np.ndarray:
    """OpenRocket nose/transition profile rising from 0 at x=0 to `radius` at x=length."""
    if radius <= 0 or length <= 0:
        return np.zeros_like(x)
    xi = np.clip(x / length, 0.0, 1.0)
    if shape == "conical":
        return radius * xi
    if shape == "ogive":
        p = param
        if p < 1e-3:
            return radius * xi
        R = math.sqrt((length ** 2 + radius ** 2) * (((2 - p) * length) ** 2 + (p * radius) ** 2)
                      / (4 * (p * radius) ** 2))
        L = length / p
        y0 = math.sqrt(R * R - L * L)
        return np.sqrt(np.maximum(R * R - (L - xi * length) ** 2, 0.0)) - y0
    if shape == "ellipsoid":
        return radius * np.sqrt(np.maximum(2 * xi - xi * xi, 0.0))
    if shape == "power":
        return radius * xi ** max(param, 1e-6)
    if shape == "parabolic":
        k = param
        return radius * (2 * xi - k * xi * xi) / (2 - k)
    if shape == "haack":
        theta = np.arccos(1 - 2 * xi)
        return radius / math.sqrt(math.pi) * np.sqrt(np.maximum(theta - np.sin(2 * theta) / 2
                                                                + param * np.sin(theta) ** 3, 0.0))
    raise ValueError(f"Unknown shape '{shape}'")


def body_radius(c: SymmetricComponent, x_local: np.ndarray) -> np.ndarray:
    """Outer radius of a body component at local station x_local (0 = component fore end)."""
    x_local = np.asarray(x_local, dtype=float)
    r0, r1, L = c.fore_radius, c.aft_radius, c.length
    if c.kind == "bodytube" or abs(r1 - r0) < 1e-12:
        return np.full_like(x_local, r0)
    if r1 >= r0:
        return r0 + _shape_radius(c.shape, x_local, L, r1 - r0, c.shape_param)
    return r1 + _shape_radius(c.shape, L - x_local, L, r0 - r1, c.shape_param)


@dataclass
class BodyProfile:
    """Sampled outer profile of one body component (absolute stations)."""
    x: np.ndarray  # absolute stations of sample points
    r: np.ndarray  # outer radius at those points

    @property
    def length(self) -> float:
        return float(self.x[-1] - self.x[0])

    def volume(self) -> float:
        return float(np.trapezoid(np.pi * self.r ** 2, self.x))

    def wetted_area(self) -> float:
        dr = np.diff(self.r)
        dx = np.diff(self.x)
        rm = 0.5 * (self.r[1:] + self.r[:-1])
        return float(np.sum(2 * np.pi * rm * np.hypot(dx, dr)))



def sample_body(c: SymmetricComponent, n: int = 201) -> BodyProfile:
    xl = np.linspace(0.0, c.length, n)
    return BodyProfile(x=c.x_top + xl, r=body_radius(c, xl))


@dataclass
class FinPlanform:
    """Aerodynamic properties of a single fin's planform (Barrowman / OpenRocket definitions).

    Coordinates: x from root leading edge (aft positive), y spanwise from root.
    """
    area: float
    span: float
    root_chord: float
    mac_length: float
    mac_lead: float  # x of the MAC leading edge
    mac_span: float  # y of the MAC (= area centroid span position)
    midchord_sweep: float  # rad
    le_sweep: float  # rad
    te_sweep: float  # rad
    y: np.ndarray  # spanwise strip centres
    chord: np.ndarray  # chord at strip centres
    dy: np.ndarray  # strip widths
    centroid_x: float  # area centroid (from root LE)
    second_moments: tuple[float, float, float]  # ∫x'^2 dA, ∫y'^2 dA, ∫x'y' dA about centroid
    le: np.ndarray  # leading-edge x at strip centres (from root LE)


def fin_planform(points: list[tuple[float, float]], n_strips: int = 24) -> FinPlanform:
    pts = np.asarray(points, dtype=float)
    span = float(pts[:, 1].max())
    if span <= 0:
        raise ValueError("Fin has zero span")

    # Leading/trailing edge as functions of y from the polygon (fin outline is
    # root LE -> ... -> tip -> ... -> root TE).
    ys_fine = np.linspace(0.0, span, 2001)
    le = np.empty_like(ys_fine)
    te = np.empty_like(ys_fine)
    for k, y in enumerate(ys_fine):
        xs = []
        for (x1, y1), (x2, y2) in zip(pts, np.roll(pts, -1, axis=0)):
            if (y1 - y) * (y2 - y) <= 0 and y1 != y2:
                xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
            elif y1 == y2 == y:
                xs += [x1, x2]
        le[k], te[k] = (min(xs), max(xs)) if xs else (np.nan, np.nan)
    c = te - le
    area = float(np.trapezoid(c, ys_fine))
    mac_length = float(np.trapezoid(c * c, ys_fine) / area)
    mac_lead = float(np.trapezoid(le * c, ys_fine) / area)
    mac_span = float(np.trapezoid(ys_fine * c, ys_fine) / area)

    root_mid = 0.5 * (le[0] + te[0])
    tip_mid = 0.5 * (le[-1] + te[-1])
    midchord_sweep = math.atan2(tip_mid - root_mid, span)
    le_sweep = math.atan2(le[-1] - le[0], span)
    te_sweep = math.atan2(te[-1] - te[0], span)

    # Strip discretisation for roll and local-flow calculations
    edges = np.linspace(0.0, span, n_strips + 1)
    yc = 0.5 * (edges[1:] + edges[:-1])
    chord = np.interp(yc, ys_fine, c)
    dy = np.diff(edges)
    chord *= area / np.sum(chord * dy)  # exact area conservation

    # Area centroid & second moments (for mass properties of a uniform plate)
    xc_strip = np.interp(yc, ys_fine, 0.5 * (le + te))
    le_strip = xc_strip - 0.5 * chord
    a = chord * dy
    cx = float(np.sum(a * xc_strip) / area)
    cy = float(np.sum(a * yc) / area)
    sxx = float(np.sum(a * ((xc_strip - cx) ** 2 + chord ** 2 / 12.0)))
    syy = float(np.sum(a * ((yc - cy) ** 2 + dy ** 2 / 12.0)))
    sxy = float(np.sum(a * (xc_strip - cx) * (yc - cy)))

    return FinPlanform(area=area, span=span, root_chord=float(c[0]), mac_length=mac_length, mac_lead=mac_lead,
                       mac_span=mac_span, midchord_sweep=midchord_sweep, le_sweep=le_sweep, te_sweep=te_sweep,
                       y=yc, chord=chord, dy=dy, centroid_x=cx, second_moments=(sxx, syy, sxy), le=le_strip)
