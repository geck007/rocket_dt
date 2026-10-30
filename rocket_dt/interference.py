"""Fin-body and fin-fin interference of deflected and rolling fins (slender-body theory).

Barrowman's K_T = 1 + r/(s + r) is the interference factor of fins whose *body* is at
an angle of attack (it approximates K_W(B) of Pitts, Nielsen and Kaattari, NACA
Report 1307). It does not apply to fins deflected relative to the body (canard
commands, cant, misalignment) or to the angle of attack along the span caused by roll
rate. For those, slender-body theory gives smaller loads. A deflected panel loads the
body (carry-over, k_B(W)) and its neighbours, so its own load k_W(B) is less than 1.
Neighbouring panels lose load near the root when all of them deflect together.

This module solves the slender-body crossflow problem numerically for N panels
(span s) on a circular body (radius a), the same in every cross-section:

* each panel is a sheet of constant-strength doublet panels (a jump in potential),
  cosine-spaced from root to tip; the body is a circle, represented by
  circle-theorem images of every vortex
* the panels' normal velocity is prescribed at the panel centres

The results are normalised by the slender-body load of the same panel as half of
an isolated two-panel wing (the basis of Barrowman's single-fin lift slope), and
depend only on tau = a / s and N. They are used in aero.py as:

    deflection   panel k carries load_matrix[k, j] times its isolated load per unit
                 deflection of panel j (j = k is k_W(B), j != k is fin-fin interference)
    carry-over   the body carries carryover times the panel's isolated load, along the
                 panel normal (canards only, scaled by ``aero.canard_body_carryover``)
    roll rate    the roll-damping moment is roll_damping times that of isolated panels

For 4 canards at tau = 0.48:
  - one canard's own load factor is 0.90, against 1.48 from K_T
  - a pitch pair gives k_W(B) = 0.94 and k_B(W) = 0.50
  - all four in roll give a roll moment of 0.70
  - roll damping is 0.71
k_W(B) + k_B(W) equals the closed-form K_W(B) (reciprocity), which tests/test_interference.py
checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class PanelInterference:
    load_matrix: np.ndarray  # (N, N): load on panel k per unit deflection of panel j, / isolated panel load
    carryover: float  # body load per unit deflection of one panel, / isolated panel load (sealed root)
    roll_damping: float  # roll-damping moment / that of N isolated panels


def k_wb_alpha(tau: float) -> float:
    """Closed-form K_W(B) of slender-body theory (panels on a body at angle of attack), tau = a/s."""
    t = tau
    if t <= 0:
        return 1.0
    return 2 / math.pi * ((1 + t ** 4) * (0.5 * math.atan(0.5 * (1 / t - t)) + math.pi / 4)
                          - t * t * ((1 / t - t) + 2 * math.atan(t))) / (1 - t) ** 2


def _solve(a: float, s: float, n: int, w_panels, m: int):
    """Doublet strengths (n, m) for normal velocity w_panels[k](r) on panel k (counter-clockwise normal)."""
    edges = a + (s - a) * 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, m + 1)))
    mid = 0.5 * (edges[1:] + edges[:-1])
    rot = np.exp(2j * math.pi * np.arange(n) / n)
    col = (rot[:, None] * mid[None, :]).ravel()
    normal = np.repeat(1j * rot, m)
    z_end = rot[:, None] * edges[None, :]  # (n, m + 1) panel end points

    def u_vortex(z0):  # velocity (u + iv) at the collocation points of unit vortices at z0, with images
        z0 = z0[None, :]
        w = -1j / (2 * math.pi) * (1 / (col[:, None] - z0))
        if a > 0:
            w = w + 1j / (2 * math.pi) * (1 / (col[:, None] - a * a / np.conj(z0)) - 1 / col[:, None])
        return np.conj(w)

    v = u_vortex(z_end.ravel()).reshape(len(col), n, m + 1)
    v = v[:, :, :-1] - v[:, :, 1:]  # doublet panel = +1 vortex at its root end, -1 at its tip end
    A = (v.reshape(len(col), n * m) * np.conj(normal)[:, None]).real
    rhs = np.concatenate([w(mid) for w in w_panels])
    mu = np.linalg.solve(A, rhs).reshape(n, m)
    return mu, edges, mid, z_end


def _body_force(mu, z_end, a):
    """Total crossflow force (complex, per rho V) of the vortex system incl. images: F = -i sum Gamma z."""
    g = np.zeros(z_end.shape, dtype=complex)
    g[:, :-1] += mu
    g[:, 1:] -= mu
    z = z_end
    s = np.sum(g * z)
    if a > 0:
        s -= np.sum(g * a * a / np.conj(z))
    return -1j * s


def _factors(tau: float, n: int, m: int) -> tuple[np.ndarray, float, float]:
    a, s = tau, 1.0
    c = s - a
    zero = lambda r: np.zeros_like(r)  # noqa: E731
    one = lambda r: np.ones_like(r)  # noqa: E731
    iso_load = math.pi * c * c / 2  # half of a two-panel slender wing, unit normal velocity

    mu, edges, mid, z_end = _solve(a, s, n, [one] + [zero] * (n - 1), m)
    dr = np.diff(edges)
    loads = mu @ dr
    rot = np.exp(2j * math.pi * np.arange(n) / n)
    panel_force = np.sum(1j * rot * loads)
    body = ((_body_force(mu, z_end, a) - panel_force) * np.conj(1j)).real  # along panel 0's normal
    first_col = loads / iso_load
    load_matrix = np.array([[first_col[(k - j) % n] for j in range(n)] for k in range(n)])

    # roll damping: normal velocity r on every panel, against an isolated panel (half of a symmetric
    # two-panel wing with normal velocity a + |y|, no body)
    mu_r, edges, mid, _ = _solve(a, s, n, [lambda r: r] * n, m)
    roll = float(np.sum(mu_r @ (mid * dr)))
    mu_i, e_i, mid_i, _ = _solve(0.0, c, 2, [lambda r: a + r, lambda r: -(a + r)], m)
    iso_roll = float(mu_i[0] @ ((a + mid_i) * np.diff(e_i)))
    return load_matrix, body / iso_load, roll / (n * iso_roll)


@lru_cache(maxsize=32)
def _cached(tau: float, n: int) -> PanelInterference:
    m = 120
    lm1, c1, r1 = _factors(tau, n, m)
    lm2, c2, r2 = _factors(tau, n, 2 * m)
    # discretisation error is ~1/m: Richardson extrapolation
    return PanelInterference(2 * lm2 - lm1, 2 * c2 - c1, 2 * r2 - r1)


def panel_interference(root_radius: float, span: float, n: int) -> PanelInterference:
    """Interference factors of n equally spaced panels of exposed span ``span`` on a body of radius ``root_radius``."""
    return _cached(round(root_radius / (root_radius + span), 6), n)
