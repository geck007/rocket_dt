"""Aerodynamic model (subsonic): extended Barrowman as documented by Niskanen
("Development of an Open Source model rocket simulation software", 2009), which is
also what OpenRocket implements.

Normal forces are evaluated with the *local* air velocity at each lifting element,
v_local = v_cg + omega x r, so pitch/yaw damping and roll damping emerge from the
same lift model rather than from separate empirical damping terms:

* Nose / transitions: slender-body normal force, CN_alpha = 2 (A_aft - A_fore) / A_ref,
  at the Barrowman CP of the component, extended to all angles with Jorgensen's
  form CN = CN_alpha/2 * sin(2a) cos(a/2) (reverses sign in reverse flow).
* Body lift: crossflow term K * A_plan * sin^2(alpha) (K = 1.1), integrated over
  axial slices of the whole body, which also yields the body's pitch damping.
* Fins (incl. canards): per-fin Diederich lift slope with body interference
  K_T = 1 + r/(s + r), at the MAC quarter chord. Each fin is split into spanwise
  strips so roll rate produces a linearly varying local angle of attack (roll
  damping) and per-fin deflections produce roll (and pitch/yaw) control moments.
  Beyond small angles each strip follows CN = CN_alpha sin(a)cos(a) +
  C_dc (A_fin/A_ref) sin(a)|sin(a)| (linear lift + flat-plate crossflow), so the
  fins keep producing a physically sized restoring force up to 90 deg instead of
  being clipped at a stall angle (OpenRocket clips at 20 deg, which creates a
  spurious stable tail-first trim during ballistic descent).
* Canard vortices (optional, ``aero.canard_vortex.model: vortex``): each canard
  panel sheds one rolled-up tip vortex whose circulation follows from that panel's
  own normal force (Kutta-Joukowski with the body image system),
  Gamma_i = F_i / (rho V (r_v - R^2/r_v)), at r_v = R + (pi/4) s (elliptic loading
  centroid). The vortex and its image inside the body (R^2/r_v, -Gamma_i) are
  carried to each fin set aft of the canards: the radius is mapped so that the
  annulus between body and vortex keeps its area, the body rolls under the vortex
  during the convection time dx/u, and the crossflow drifts it by -v_cross dx/u.
  Their crossflow velocity (Lamb-Oseen core) at every fin strip adds
  -w_n / u to that strip's angle of attack, so the coupling is resolved per
  canard / fin pair, depends on the fin-can angle, and acts on roll, pitch and
  yaw alike. Not modelled: vortex self-induced motion, convection delay, vortices
  shed by the body.
* Axial force: zero-lift drag build-up (skin friction with roughness limit,
  pressure, base and fin edge drag) times OpenRocket's angle-of-attack multiplier.

Sign conventions (body frame, +x forward):
  canard deflection delta_i > 0 produces a positive roll moment about +x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rocket_dt.config import CanardVortexConfig
from rocket_dt.geometry import BodyProfile, FinPlanform, fin_planform, sample_body
from rocket_dt.io.ork import FINISH_ROUGHNESS, FinSet, OrkRocket, SymmetricComponent, reference_length, rocket_length
from rocket_dt.mass import fin_angles, fin_root_radius

BODY_LIFT_K = 1.1
FIN_PLATE_CDC = 1.2  # broadside normal-force coefficient of a low-aspect-ratio flat plate
_A17 = math.radians(17.0)


def _base_cd(mach: float) -> float:
    return 0.12 + 0.13 * mach * mach


def _stagnation_cd(mach: float) -> float:
    return 0.85 * (1.0 + mach ** 2 / 4.0 + mach ** 4 / 40.0)


def _axial_multiplier(alpha: float) -> float:
    """OpenRocket's axial-drag multiplier versus angle of attack (alpha in [0, pi/2])."""
    if alpha < _A17:
        s = alpha / _A17
        return 1.0 + 0.3 * (3 * s * s - 2 * s ** 3)
    h = 1.0 - (alpha - _A17) / (math.pi / 2 - _A17)
    return 1.3 * h ** 3 * (4.0 - 3.0 * h)


@dataclass
class FinSetAero:
    path: str
    count: int
    station_cp: float  # MAC quarter chord station
    root_radius: float
    planform: FinPlanform
    thickness: float
    cross_section: str
    roughness: float
    angles: np.ndarray  # roll position of each fin (rad) relative to its carrier
    cant: np.ndarray  # per-fin built-in cant incl. misalignment (rad)
    on_fincan: bool
    is_canard: bool
    ref_area: float
    root_le_station: float  # station of the root leading edge
    outline: np.ndarray  # planform polygon (x from root LE, y span), for drawing
    interference: float = field(init=False)
    _rho: np.ndarray = field(init=False)
    _w: np.ndarray = field(init=False)

    def __post_init__(self):
        s = self.planform.span
        self.interference = 1.0 + self.root_radius / (s + self.root_radius)
        self._rho = self.root_radius + self.planform.y
        self._w = self.planform.chord * self.planform.dy / self.planform.area

    def cn_alpha_single(self, beta: float) -> float:
        """Diederich lift slope of one fin, referenced to A_ref (per rad)."""
        pf = self.planform
        s2 = pf.span ** 2
        return (2 * math.pi * s2 / self.ref_area) / (
            1.0 + math.sqrt(1.0 + (beta * s2 / (pf.area * math.cos(pf.midchord_sweep))) ** 2))

    def cn_alpha_set(self, beta: float) -> float:
        """Normal-force slope of the whole set incl. body interference (N >= 3)."""
        return 0.5 * self.count * self.interference * self.cn_alpha_single(beta)


@dataclass
class AeroResult:
    force: np.ndarray  # body frame, N
    moment: np.ndarray  # about CG, body frame, N m
    fincan_roll_moment: float  # aerodynamic roll moment acting on the fin can, N m
    diag: dict | None = None


@dataclass
class _Vortices:
    """Canard tip vortices as shed at the canards (body frame, crossflow plane)."""
    gamma: np.ndarray  # circulation of each canard's tip vortex about +x, m^2/s
    angles: np.ndarray  # roll position of each canard (rad)
    r_v: float  # vortex radius at the canards
    r_body: float  # body radius at the canards
    station: float  # station the vortices are shed from


class AeroModel:
    def __init__(self, ork: OrkRocket, fincan_paths: set[str], canard_path: str | None,
                 fin_misalignment: dict[str, list[float]], drag_scale: float = 1.0,
                 canard_vortex: CanardVortexConfig | None = None, slice_length: float = 0.025):
        self.ref_length = reference_length(ork)
        self.ref_area = math.pi * self.ref_length ** 2 / 4.0
        self.length = rocket_length(ork)
        self.drag_scale = drag_scale
        self.vortex = canard_vortex if canard_vortex is not None and canard_vortex.model == "vortex" else None

        bodies = [c for c in ork.components if isinstance(c, SymmetricComponent)]
        if [c for c in ork.components if not isinstance(c, SymmetricComponent)]:
            raise ValueError("Top-level components must all be nose cones, body tubes or transitions")
        self.profiles: list[tuple[SymmetricComponent, BodyProfile]] = [(c, sample_body(c)) for c in bodies]
        self.max_diameter = 2 * max(float(p.r.max()) for _, p in self.profiles)

        # Slender-body (potential) normal force of nose / transitions
        cna, xcp = [], []
        for c, p in self.profiles:
            a0, a1 = math.pi * p.r[0] ** 2, math.pi * p.r[-1] ** 2
            if abs(a1 - a0) > 1e-12:
                cna.append(2.0 * (a1 - a0) / self.ref_area)
                xcp.append(c.x_top + (c.length * a1 - p.volume()) / (a1 - a0))
        self.body_cna = np.array(cna)
        self.body_cp_station = np.array(xcp)

        # Axial slices for crossflow body lift
        xs, rs, dxs = [], [], []
        for c, p in self.profiles:
            n = max(1, int(math.ceil(c.length / slice_length)))
            edges = np.linspace(p.x[0], p.x[-1], n + 1)
            mid = 0.5 * (edges[1:] + edges[:-1])
            xs.append(mid)
            rs.append(np.interp(mid, p.x, p.r))
            dxs.append(np.diff(edges))
        self.slice_station = np.concatenate(xs)
        self.slice_area = 2.0 * np.concatenate(rs) * np.concatenate(dxs)  # planform area per slice

        # Fin sets
        fin_sets = [c for c in ork.walk() if isinstance(c, FinSet)]
        if canard_path is not None and canard_path not in {f.path for f in fin_sets}:
            raise ValueError(f"Canard fin set '{canard_path}' not found; fin sets: {[f.path for f in fin_sets]}")
        unknown = set(fin_misalignment) - {f.path for f in fin_sets}
        if unknown:
            raise ValueError(f"fin_misalignment_deg refers to unknown fin sets {sorted(unknown)}")
        self.fins: list[FinSetAero] = []
        for fs in fin_sets:
            if fs.fin_count < 3:
                raise ValueError(f"'{fs.path}': fin sets with fewer than 3 fins are not supported")
            pf = fin_planform(fs.planform)
            mis = np.radians(np.asarray(fin_misalignment.get(fs.path, [0.0] * fs.fin_count), dtype=float))
            if mis.shape != (fs.fin_count,):
                raise ValueError(f"fin_misalignment_deg['{fs.path}'] needs {fs.fin_count} values")
            self.fins.append(FinSetAero(
                path=fs.path, count=fs.fin_count, station_cp=fs.x_top + pf.mac_lead + 0.25 * pf.mac_length,
                root_radius=fin_root_radius(fs), planform=pf, thickness=fs.thickness,
                cross_section=fs.cross_section, roughness=FINISH_ROUGHNESS[fs.finish], angles=fin_angles(fs),
                cant=fs.cant + mis, on_fincan=_in_subtree(fs, fincan_paths), is_canard=fs.path == canard_path,
                ref_area=self.ref_area, root_le_station=fs.x_top, outline=np.asarray(fs.planform, dtype=float)))
        if canard_path is not None and next(f for f in self.fins if f.is_canard).on_fincan:
            raise ValueError("The canards cannot be part of the fin can")
        self.n_canards = next((f.count for f in self.fins if f.is_canard), 0)
        canards = next((f for f in self.fins if f.is_canard), None)
        # canards first, so their vortices are known when the fin sets behind them are evaluated
        self._eval_order = sorted(self.fins, key=lambda f: not f.is_canard)
        self._vortex_targets = {id(f) for f in self.fins if canards is not None and not f.is_canard
                                and f.station_cp > canards.station_cp}
        if self.vortex is None:
            self._vortex_targets = set()
        self._init_drag()

    # ------------------------------------------------------------------ drag

    def _init_drag(self):
        self.fineness = self.length / self.max_diameter
        self._body_wet = [(p.wetted_area(), FINISH_ROUGHNESS[c.finish]) for c, p in self.profiles]
        self._fin_wet = [(2.0 * f.count * f.planform.area, f.roughness, f.thickness / f.planform.mac_length)
                         for f in self.fins]
        fwd, boattail, steps_up, steps_down = [], [], [], []
        for c, p in self.profiles:
            r0, r1 = float(p.r[0]), float(p.r[-1])
            da = math.pi * abs(r1 * r1 - r0 * r0) / self.ref_area
            if r1 > r0 + 1e-9:
                slope = (p.r[-1] - p.r[-2]) / (p.x[-1] - p.x[-2]) if c.kind == "nosecone" else (r1 - r0) / c.length
                fwd.append((0.8 * math.sin(math.atan(slope)) ** 2, da))
            elif r0 > r1 + 1e-9:
                gamma = c.length / (2.0 * (r0 - r1))
                factor = 1.0 if gamma < 1 else ((3.0 - gamma) / 2.0 if gamma < 3 else 0.0)
                boattail.append((factor, da))
        for (_, pa), (_, pb) in zip(self.profiles[:-1], self.profiles[1:]):
            ra, rb = float(pa.r[-1]), float(pb.r[0])
            da = math.pi * abs(rb * rb - ra * ra) / self.ref_area
            if rb > ra + 1e-6:
                steps_up.append(da)
            elif ra > rb + 1e-6:
                steps_down.append(da)
        self._fwd, self._boattail = fwd, boattail
        self._steps_up, self._steps_down = steps_up, steps_down
        r_base = float(self.profiles[-1][1].r[-1])
        self._base_area = math.pi * r_base ** 2 / self.ref_area

        # Precomputed friction terms: wetted area (incl. form factors) and roughness-limited Cf per surface
        ff_body = 1.0 + 1.0 / (2.0 * self.fineness)
        self._fric_area = np.array([a * ff_body for a, _ in self._body_wet]
                                   + [a * (1.0 + 2.0 * tc) for a, _, tc in self._fin_wet]) / self.ref_area
        self._fric_cf_rough = np.array([0.032 * (r / self.length) ** 0.2 if r > 0 else 0.0
                                        for r in [r for _, r in self._body_wet] + [r for _, r, _ in self._fin_wet]])
        # Fin edge drag: (LE sweep factor, frontal area ratio, cross section) per fin set
        self._fin_edges = []
        for f in self.fins:
            if f.cross_section not in ("square", "rounded", "airfoil"):
                raise ValueError(f"Unknown cross section {f.cross_section}")
            self._fin_edges.append((math.cos(f.planform.le_sweep) ** 2,
                                    f.count * f.planform.span * f.thickness / self.ref_area, f.cross_section))
        self._fwd_sum = sum(c * da for c, da in self._fwd)
        self._boattail_sum = sum(f * da for f, da in self._boattail)
        self._steps_up_sum = sum(self._steps_up)
        self._steps_down_sum = sum(self._steps_down)

    def drag_breakdown(self, mach: float, reynolds: float) -> dict:
        """Zero-angle-of-attack drag coefficients (referenced to A_ref)."""
        m2 = mach * mach
        comp = 1.0 - 0.1 * m2
        cf_t = 1.48e-2 if reynolds < 1e4 else 1.0 / (1.50 * math.log(reynolds) - 5.6) ** 2
        friction = float(np.maximum(cf_t, self._fric_cf_rough) @ self._fric_area) * comp

        cdb = _base_cd(mach)
        stag = _stagnation_cd(mach)
        pressure = self._fwd_sum + self._boattail_sum * cdb + self._steps_up_sum * stag + self._steps_down_sum * cdb
        rounded_le = (1.0 - min(m2, 0.81)) ** -0.417 - 1.0
        for sweep_fac, area, xs in self._fin_edges:
            if xs == "square":
                le, te = stag, cdb
            else:
                le, te = rounded_le, (cdb / 2.0 if xs == "rounded" else 0.0)
            pressure += (le * sweep_fac + te) * area
        base = cdb * self._base_area
        s = self.drag_scale
        return {"friction": friction * s, "pressure": pressure * s, "base": base * s,
                "total": (friction + pressure + base) * s}

    # ------------------------------------------------------------------ static

    def static_cp(self, mach: float) -> tuple[float, float]:
        """(CN_alpha total, CP station) of the linear (small-alpha) model."""
        beta = math.sqrt(max(1.0 - mach * mach, 0.05))
        cna = list(self.body_cna) + [f.cn_alpha_set(beta) for f in self.fins]
        x = list(self.body_cp_station) + [f.station_cp for f in self.fins]
        if self._vortex_targets:
            # canard downwash on the fins behind them, from the full model at small alpha
            a, V = math.radians(0.5), 100.0
            v = np.array([V * math.cos(a), 0.0, V * math.sin(a)])
            res = [self.evaluate(v, np.zeros(3), 0.0, 0.0, None, 1.225, V / mach, 1.46e-5, 0.0, vortex=w)
                   for w in (True, False)]
            dfz = res[0].force[2] - res[1].force[2]
            if abs(dfz) > 1e-12:
                cna.append(-dfz / (0.5 * 1.225 * V * V * self.ref_area * a))
                x.append((res[0].moment[1] - res[1].moment[1]) / dfz)
        tot = float(sum(cna))
        return tot, float(np.dot(cna, x) / tot)

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, v_b: np.ndarray, omega_b: np.ndarray, fincan_rate: float, fincan_angle: float,
                 canard_deflection: np.ndarray | None, rho: float, sound_speed: float, nu: float,
                 cg_station: float, diagnostics: bool = False, vortex: bool = True) -> AeroResult:
        """Aerodynamic force and moment about the CG.

        v_b          : velocity of the CG relative to the air, body frame
        omega_b      : body angular velocity
        fincan_rate  : fin-can roll rate relative to the body (rad/s)
        fincan_angle : fin-can roll angle relative to the body (rad)
        vortex       : False switches the canard vortices off for this call
        """
        V = float(np.linalg.norm(v_b))
        if V < 1e-3:
            return AeroResult(np.zeros(3), np.zeros(3), 0.0, {} if diagnostics else None)
        q = 0.5 * rho * V * V
        mach = V / sound_speed
        beta = math.sqrt(max(1.0 - mach * mach, 0.05))
        wx, wy, wz = omega_b
        force = np.zeros(3)
        moment = np.zeros(3)

        # --- body: potential normal force + crossflow lift (vectorised over points)
        st = np.concatenate([self.body_cp_station, self.slice_station])
        r_ax = cg_station - st  # body-x of each point relative to CG
        vy = v_b[1] + wz * r_ax
        vz = v_b[2] - wy * r_ax
        vx = v_b[0]
        nb = len(self.body_cna)
        vmag = np.sqrt(vx * vx + vy * vy + vz * vz)
        cmag = np.sqrt(vy * vy + vz * vz)
        coef = np.empty_like(st)
        # Jorgensen: CN = CNa/2 sin(2a) cos(a/2); force = -q A CN c_hat = -0.5 rho |v| A CNa cos(a) cos(a/2) c
        cos_a = vx / np.maximum(vmag[:nb], 1e-9)
        cos_half = np.sqrt(np.maximum(0.5 * (1.0 + cos_a), 0.0))
        coef[:nb] = 0.5 * rho * vmag[:nb] * self.ref_area * self.body_cna * cos_a * cos_half
        coef[nb:] = 0.5 * rho * BODY_LIFT_K * self.slice_area * cmag[nb:]
        fy = -coef * vy
        fz = -coef * vz
        force[1] += fy.sum()
        force[2] += fz.sum()
        moment[1] += -float(r_ax @ fz)
        moment[2] += float(r_ax @ fy)

        # --- fins and canards
        fincan_roll = 0.0
        canard_roll = 0.0
        vortices = None
        vortex_roll = 0.0
        vortex_roll_each = np.zeros(self.n_canards)
        u = vx
        ua = max(abs(u), 1e-6)
        for f in self._eval_order:
            phi = f.angles + (fincan_angle if f.on_fincan else 0.0)
            cphi, sphi = np.cos(phi), np.sin(phi)
            r_cp = cg_station - f.station_cp
            ty = v_b[1] + wz * r_cp
            tz = v_b[2] - wy * r_cp
            vn = -sphi * ty + cphi * tz  # velocity component along each fin normal
            p_eff = wx + (fincan_rate if f.on_fincan else 0.0)
            delta = f.cant.copy()
            if f.is_canard and canard_deflection is not None:
                delta = delta + canard_deflection
            k = f.interference
            alpha = (k * (np.arctan2(vn, ua) - delta * math.copysign(1.0, u)))[:, None] \
                + np.arctan2(p_eff * f._rho[None, :], ua)
            d_alpha = None
            if vortex and vortices is not None and id(f) in self._vortex_targets:
                d_alpha = self._vortex_alpha(vortices, f, phi, v_b, omega_b, cg_station)
                strip_f, fn, roll = self._strip_loads(f, alpha + d_alpha.sum(axis=0), beta, q)
                if diagnostics:
                    roll0 = self._strip_loads(f, alpha, beta, q)[2]
                    vortex_roll += roll - roll0
                    for i in range(len(d_alpha)):
                        vortex_roll_each[i] += self._strip_loads(f, alpha + d_alpha[i], beta, q)[2] - roll0
            else:
                strip_f, fn, roll = self._strip_loads(f, alpha, beta, q)
            force[1] += -float(sphi @ fn)
            force[2] += float(cphi @ fn)
            # r_cp x̂ × (fn n̂) = -r_cp fn ê_r
            moment[1] += -r_cp * float(fn @ cphi)
            moment[2] += -r_cp * float(fn @ sphi)
            moment[0] += roll
            if f.on_fincan:
                fincan_roll += roll
            if f.is_canard:
                canard_roll += roll
                if self._vortex_targets and u > 1e-3:
                    r_v = f.root_radius + 0.25 * math.pi * f.planform.span
                    gamma = self.vortex.strength_factor * fn / (rho * V * (r_v - f.root_radius ** 2 / r_v))
                    vortices = _Vortices(gamma, phi, r_v, f.root_radius, f.station_cp)

        # --- axial force
        reynolds = V * self.length / nu
        dr = self.drag_breakdown(mach, reynolds)
        cos_a = max(-1.0, min(1.0, vx / V))
        aoa = math.acos(cos_a)
        mul = _axial_multiplier(aoa if aoa <= math.pi / 2 else math.pi - aoa)
        ca = mul * dr["total"] * (1.0 if aoa <= math.pi / 2 else -1.0)
        force[0] -= q * self.ref_area * ca

        diag = None
        if diagnostics:
            cna, cp = self.static_cp(mach)
            diag = {"mach": mach, "aoa": aoa, "q": q, "reynolds": reynolds, "cd": dr["total"],
                    "cd_friction": dr["friction"], "cd_pressure": dr["pressure"], "cd_base": dr["base"],
                    "ca": ca, "cn": float(math.hypot(force[1], force[2]) / (q * self.ref_area)),
                    "cn_alpha": cna, "cp_station": cp, "canard_roll_moment": canard_roll,
                    "vortex_roll_moment": vortex_roll, "vortex_roll_moment_each": vortex_roll_each}
        return AeroResult(force, moment, fincan_roll, diag)


    def _strip_loads(self, f: FinSetAero, alpha: np.ndarray, beta: float, q: float):
        """Strip forces along each fin normal, per-fin normal force and total roll moment of a fin set."""
        alpha = np.clip(alpha, -0.5 * math.pi, 0.5 * math.pi)
        sa = np.sin(alpha)
        cn = f.cn_alpha_single(beta) * sa * np.cos(alpha) \
            + FIN_PLATE_CDC * f.planform.area / self.ref_area * sa * np.abs(sa)
        strip_f = -q * self.ref_area * f._w[None, :] * cn
        return strip_f, strip_f.sum(axis=1), float((strip_f @ f._rho).sum())

    def _vortex_alpha(self, vx: _Vortices, f: FinSetAero, phi: np.ndarray, v_b: np.ndarray,
                      omega_b: np.ndarray, cg_station: float) -> np.ndarray:
        """Angle-of-attack change of every strip of fin set ``f`` induced by each canard vortex.

        Returns shape (n_canards, n_fins, n_strips).
        """
        u = v_b[0]
        dx = f.station_cp - vx.station
        dt = dx / u
        r_fin = f.root_radius
        r_t = math.sqrt(max(vx.r_v ** 2 + r_fin ** 2 - vx.r_body ** 2, 0.0))
        ang = vx.angles - (omega_b[0] * dt if self.vortex.roll_lag else 0.0)
        pos = r_t * np.column_stack([np.cos(ang), np.sin(ang)])
        r_mid = cg_station - 0.5 * (vx.station + f.station_cp)
        pos -= dt * np.array([v_b[1] + omega_b[2] * r_mid, v_b[2] - omega_b[1] * r_mid])
        rc = self.vortex.core_radius
        rad2 = np.sum(pos * pos, axis=1)
        rad_min = r_fin + rc
        pos *= np.sqrt(np.maximum(rad2, rad_min ** 2) / rad2)[:, None]  # a vortex cannot enter the body
        img = pos * (r_fin ** 2 / np.sum(pos * pos, axis=1))[:, None]
        n = len(vx.gamma)
        src = np.concatenate([pos, img])  # vortices, then their images (opposite sign)
        cphi, sphi = np.cos(phi), np.sin(phi)
        dy = f._rho[None, None, :] * cphi[None, :, None] - src[:, 0, None, None]  # (2n, n_fins, n_strips)
        dz = f._rho[None, None, :] * sphi[None, :, None] - src[:, 1, None, None]
        r2 = np.maximum(dy * dy + dz * dz, 1e-12)
        fac = (1.0 - np.exp(-r2 / (rc * rc))) / r2
        # counter-clockwise velocity (-dz, dy) * Gamma / (2 pi r^2), projected on the fin normal (-sin phi, cos phi)
        w = fac * (dz * sphi[None, :, None] + dy * cphi[None, :, None])
        wn = (w[:n] - w[n:]) * (vx.gamma / (2.0 * math.pi))[:, None, None]
        return -wn / u

    def vortex_roll_map(self, mach: float, fincan_angles: np.ndarray, deflection: np.ndarray,
                        sound_speed: float = 340.3, rho: float = 1.225, nu: float = 1.46e-5) -> dict:
        """Canard and vortex-induced roll moments at zero incidence versus fin-can angle."""
        V = mach * sound_speed
        canard, induced, each = [], [], []
        for a in fincan_angles:
            d = self.evaluate(np.array([V, 0.0, 0.0]), np.zeros(3), 0.0, float(a), deflection, rho, sound_speed,
                              nu, 0.0, diagnostics=True).diag
            canard.append(d["canard_roll_moment"])
            induced.append(d["vortex_roll_moment"])
            each.append(d["vortex_roll_moment_each"])
        return {"canard": np.array(canard), "induced": np.array(induced), "each": np.array(each)}


def _in_subtree(c, roots: set[str]) -> bool:
    while c is not None and c.kind != "stage":
        if c.path in roots:
            return True
        c = c.parent
    return False
