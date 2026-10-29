"""Plotly figure builders for the dashboard (no Streamlit dependency)."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {"body": "#9aa4b1", "nose": "#d9dde3", "canards": "#ff6b4a", "fincan": "#4da3ff", "fins": "#a78bfa"}
PALETTE = ["#4da3ff", "#ff8c42", "#3ddc97", "#c084fc", "#ffd166", "#2ec4b6"]
NEUTRAL = "#aab4c3"  # readable on light and dark backgrounds
EVENT_LINE = "rgba(160,170,185,0.4)"


def register_dark_template(name: str = "rocket_dark") -> str:
    """Plotly template matching the dashboard's dark theme; returns its name."""
    import plotly.io as pio
    grid, axis_bg, text = "#243040", "#0f151d", "#d5dbe5"
    ax = dict(gridcolor=grid, zerolinecolor=grid, linecolor="#33404f", tickcolor="#33404f")
    scene_ax = dict(backgroundcolor=axis_bg, gridcolor=grid, zerolinecolor=grid, showbackground=True,
                    color=text)
    pio.templates[name] = go.layout.Template(
        layout=dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=axis_bg, font=dict(color=text, size=12),
                    colorway=PALETTE, xaxis=ax, yaxis=ax, hoverlabel=dict(bgcolor="#1a2230"),
                    scene=dict(xaxis=scene_ax, yaxis=scene_ax, zaxis=scene_ax),
                    legend=dict(bgcolor="rgba(0,0,0,0)")))
    return name
EVENT_LABELS = {"liftoff": "liftoff", "rail_exit": "rail exit", "burnout": "burnout", "apogee": "apogee",
                "ground_hit": "ground"}


# ----------------------------------------------------------------------------- geometry


def rocket_geometry(rocket) -> dict:
    """Plain-data description of the airframe for drawing (body frame stations)."""
    st, r = [], []
    for c, p in rocket.aero.profiles:
        st.append(p.x)
        r.append(p.r)
    fins = [{"path": f.path, "root_le": f.root_le_station, "outline": f.outline, "root_radius": f.root_radius,
             "angles": f.angles, "on_fincan": f.on_fincan, "is_canard": f.is_canard} for f in rocket.aero.fins]
    nose_len = rocket.aero.profiles[0][0].length if rocket.aero.profiles[0][0].kind == "nosecone" else 0.0
    return {"stations": np.concatenate(st), "radii": np.concatenate(r), "fins": fins, "length": rocket.length,
            "ref_length": rocket.aero.ref_length, "nose_length": nose_len,
            "motor": (rocket.motor.fore_station, rocket.motor.aft_station, rocket.motor.case_radius)}


def _body_mesh(geom, n_theta=18, n_st=48):
    s_all, r_all = geom["stations"], geom["radii"]
    s = np.linspace(s_all[0], s_all[-1], n_st)
    r = np.interp(s, s_all, r_all)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    S, TH = np.meshgrid(s, th, indexing="ij")
    R = np.repeat(r[:, None], n_theta, axis=1)
    V = np.column_stack([-S.ravel(), (R * np.cos(TH)).ravel(), (R * np.sin(TH)).ravel()])
    I, J, K = [], [], []
    for a in range(n_st - 1):
        for b in range(n_theta):
            p0, p1 = a * n_theta + b, a * n_theta + (b + 1) % n_theta
            p2, p3 = p0 + n_theta, p1 + n_theta
            I += [p0, p1]
            J += [p1, p3]
            K += [p2, p2]
    return V, np.array(I), np.array(J), np.array(K)


def _fin_meshes(geom, fincan_angle: float) -> dict[str, tuple]:
    """Fin polygons as triangle fans, grouped (canards / fincan / fins)."""
    groups: dict[str, list] = {}
    for f in geom["fins"]:
        key = "canards" if f["is_canard"] else ("fincan" if f["on_fincan"] else "fins")
        out = f["outline"]
        for phi in f["angles"] + (fincan_angle if f["on_fincan"] else 0.0):
            rad = f["root_radius"] + out[:, 1]
            pts = np.column_stack([-(f["root_le"] + out[:, 0]), rad * math.cos(phi), rad * math.sin(phi)])
            groups.setdefault(key, []).append(pts)
    meshes = {}
    for key, polys in groups.items():
        V, I, J, K = [], [], [], []
        off = 0
        for pts in polys:
            n = len(pts)
            V.append(pts)
            for k in range(1, n - 1):
                I.append(off)
                J.append(off + k)
                K.append(off + k + 1)
            off += n
        meshes[key] = (np.vstack(V), np.array(I), np.array(J), np.array(K))
    return meshes


def _quat_matrix(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _to_world(V_body, d, i, scale, center):
    """Body-frame points (origin at nose tip) -> ENU, about the CG, scaled, placed at center."""
    R = _quat_matrix(d["qw"][i], d["qx"][i], d["qy"][i], d["qz"][i])
    cg = np.array([-d["cg_station"][i], 0.0, 0.0])
    return center + scale * (V_body - cg) @ R.T


def frame_indices(d, n_frames: int, t_end: float | None = None) -> np.ndarray:
    t = d["t"]
    t_end = t[-1] if t_end is None else min(t_end, t[-1])
    # denser frames during the short boost phase, where most happens
    tt = np.unique(np.concatenate([np.linspace(0, min(3.0, t_end), n_frames // 4),
                                   np.linspace(0, t_end, n_frames - n_frames // 4)]))
    return np.unique(np.clip(np.searchsorted(t, tt), 0, len(t) - 1))


def _rocket_traces(geom, d, i, scale, center, body_mesh, full=True):
    """Mesh traces of the vehicle at log index i. With full=False only vertex
    coordinates are emitted (animation frames reuse the constant connectivity)."""
    parts = [("body", body_mesh)]
    meshes = _fin_meshes(geom, d["fincan_angle_rel"][i])
    parts += [(k, meshes[k]) for k in ("canards", "fincan", "fins") if k in meshes]
    traces = []
    for name, (V, I, J, K) in parts:
        W = _to_world(V, d, i, scale, center)
        if not full:
            traces.append(go.Mesh3d(x=W[:, 0], y=W[:, 1], z=W[:, 2]))
            continue
        traces.append(go.Mesh3d(x=W[:, 0], y=W[:, 1], z=W[:, 2], i=I, j=J, k=K, color=COLORS[name], name=name,
                                hoverinfo="skip", flatshading=name != "body",
                                lighting=dict(ambient=0.55, diffuse=0.8, specular=0.3)))
    return traces


def _anim_controls(frame_names, duration=60):
    return dict(
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=0.0, xanchor="left", yanchor="top",
                          pad=dict(t=10, r=10), direction="left",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=duration, redraw=True),
                                                         transition=dict(duration=0), fromcurrent=True)]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate")])])],
        sliders=[dict(active=0, x=0.12, y=0.0, len=0.86, pad=dict(t=10), currentvalue=dict(prefix="t = ", suffix=" s"),
                      steps=[dict(method="animate", label=n,
                                  args=[[n], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                  transition=dict(duration=0))]) for n in frame_names])])


def _status(d, i):
    return (f"t = {d['t'][i]:6.2f} s   alt {d['z'][i]:7.1f} m   speed {d['speed'][i]:6.1f} m/s   "
            f"Mach {d['mach'][i]:.2f}   AoA {math.degrees(d['aoa'][i]):5.1f}°   "
            f"roll body {math.degrees(d['p'][i]):7.1f}°/s   fin can {math.degrees(d['fincan_rate_abs'][i]):7.1f}°/s")


# ----------------------------------------------------------------------------- 3D figures


def fig_launch(d, geom, events, n_frames=120, t_end=None, rocket_scale=None, height=720) -> go.Figure:
    """Whole-flight 3D view with an (enlarged) rocket model flying along the trajectory."""
    idx = frame_indices(d, n_frames, t_end)
    x, y, z = d["x"], d["y"], d["z"]
    last = idx[-1]
    zmax = max(float(z[: last + 1].max()), 10.0)
    W = max(float(np.abs(x[: last + 1]).max()), float(np.abs(y[: last + 1]).max()), 0.25 * zmax) * 1.15
    scale = rocket_scale or 0.08 * zmax / geom["length"]
    body_mesh = _body_mesh(geom, n_theta=12, n_st=30)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x[: last + 1], y=y[: last + 1], z=z[: last + 1], mode="lines",
                               line=dict(color="rgba(120,120,120,0.35)", width=3, dash="dot"), name="trajectory",
                               hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 0], mode="lines",
                               line=dict(color="#ffb703", width=6), name="flown path"))
    ev_t = [(k, v) for k, v in events.items() if k in EVENT_LABELS and v <= d["t"][last] + 1e-9]
    ev_i = [min(np.searchsorted(d["t"], v), len(d["t"]) - 1) for _, v in ev_t]
    fig.add_trace(go.Scatter3d(x=x[ev_i], y=y[ev_i], z=z[ev_i], mode="markers+text",
                               text=[EVENT_LABELS[k] for k, _ in ev_t], textposition="middle right",
                               marker=dict(size=4, color=NEUTRAL), name="events"))
    pad = W * 0.6
    fig.add_trace(go.Mesh3d(x=[-pad, pad, pad, -pad], y=[-pad, -pad, pad, pad], z=[0, 0, 0, 0], i=[0, 0], j=[1, 2],
                            k=[2, 3], color="#3a7d44", opacity=0.25, hoverinfo="skip", name="ground"))
    n_static = len(fig.data)
    for tr in _rocket_traces(geom, d, idx[0], scale, np.array([x[idx[0]], y[idx[0]], z[idx[0]]]), body_mesh):
        fig.add_trace(tr)
    n_rocket = len(fig.data) - n_static

    frames = []
    for i in idx:
        c = np.array([x[i], y[i], z[i]])
        data = [go.Scatter3d(x=x[: i + 1], y=y[: i + 1], z=z[: i + 1])]
        data += _rocket_traces(geom, d, i, scale, c, body_mesh, full=False)
        frames.append(go.Frame(name=f"{d['t'][i]:.2f}", data=data,
                               traces=[1] + list(range(n_static, n_static + n_rocket)),
                               layout=dict(title_text=_status(d, i))))
    fig.frames = frames
    aspect_z = min(zmax / (2 * W), 1.8)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=40, b=0), title=dict(text=_status(d, idx[0]), font=dict(size=13)),
        scene=dict(xaxis=dict(title="East [m]", range=[-W, W]), yaxis=dict(title="North [m]", range=[-W, W]),
                   zaxis=dict(title="Altitude [m]", range=[0, zmax * 1.08]),
                   aspectmode="manual", aspectratio=dict(x=1, y=1, z=aspect_z),
                   camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9), center=dict(x=0, y=0, z=-0.1))),
        legend=dict(x=0.01, y=0.98), **_anim_controls([f.name for f in frames]))
    return fig


def fig_attitude(d, geom, n_frames=120, t_end=None, height=620) -> go.Figure:
    """Close-up of the vehicle at its CG: attitude, relative wind, spinning fin can."""
    idx = frame_indices(d, n_frames, t_end)
    L = geom["length"]
    body_mesh = _body_mesh(geom)
    vax = np.column_stack([d["vx"] - d["wind_e"], d["vy"] - d["wind_n"], d["vz"]])

    def arrows(i):
        v = vax[i]
        n = np.linalg.norm(v)
        u = v / n * 0.8 * L if n > 1e-6 else np.zeros(3)
        R = _quat_matrix(d["qw"][i], d["qx"][i], d["qy"][i], d["qz"][i])
        ax = R[:, 0] * 0.9 * L
        return [go.Scatter3d(x=[0, u[0]], y=[0, u[1]], z=[0, u[2]], mode="lines",
                             line=dict(color="#3ddc97", width=7), name="velocity rel. air"),
                go.Scatter3d(x=[0, ax[0]], y=[0, ax[1]], z=[0, ax[2]], mode="lines",
                             line=dict(color=NEUTRAL, width=3, dash="dash"), name="body axis")]

    fig = go.Figure()
    i0 = idx[0]
    for tr in _rocket_traces(geom, d, i0, 1.0, np.zeros(3), body_mesh) + arrows(i0):
        fig.add_trace(tr)
    n = len(fig.data)
    frames = [go.Frame(name=f"{d['t'][i]:.2f}",
                       data=_rocket_traces(geom, d, i, 1.0, np.zeros(3), body_mesh, full=False) + arrows(i),
                       traces=list(range(n)), layout=dict(title_text=_status(d, i))) for i in idx]
    fig.frames = frames
    r = 0.75 * L
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=40, b=0), title=dict(text=_status(d, i0), font=dict(size=13)),
        scene=dict(xaxis=dict(title="E", range=[-r, r]), yaxis=dict(title="N", range=[-r, r]),
                   zaxis=dict(title="Up", range=[-r, r]), aspectmode="cube",
                   camera=dict(eye=dict(x=1.3, y=-1.3, z=0.5))),
        legend=dict(x=0.01, y=0.98), **_anim_controls([f.name for f in frames]))
    return fig


# ----------------------------------------------------------------------------- 2D figures


def add_event_lines(fig, events, rows=None, t_max=None):
    for k, v in events.items():
        if k in EVENT_LABELS and (t_max is None or v <= t_max):
            fig.add_vline(x=v, line=dict(color=EVENT_LINE, width=1, dash="dot"))
    return fig


def fig_panels(d, events, panels, cols=2, height_per_row=260, t_max=None, cursor=None) -> go.Figure:
    """panels: list of (title, [(label, y_array), ...])."""
    rows = math.ceil(len(panels) / cols)
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=False, subplot_titles=[p[0] for p in panels],
                        vertical_spacing=min(0.14, 0.42 / rows), horizontal_spacing=0.07)
    t = d["t"]
    sel = t <= (t_max if t_max is not None else t[-1]) + 1e-9
    legends = {}
    for n, (title, lines) in enumerate(panels):
        r, c = n // cols + 1, n % cols + 1
        leg = "legend" if n == 0 else f"legend{n + 1}"
        for k, (label, yv) in enumerate(lines):
            fig.add_trace(go.Scatter(x=t[sel], y=np.asarray(yv)[sel], name=label, mode="lines",
                                     line=dict(width=1.6, color=PALETTE[k % len(PALETTE)]),
                                     legend=leg, showlegend=len(lines) > 1,
                                     hovertemplate=f"{label}: %{{y:.4g}}<extra></extra>"), row=r, col=c)
        fig.update_xaxes(title_text="t [s]" if r == rows else None, row=r, col=c)
        # each panel gets its own legend, inside the top-right corner of its plot area
        sp = fig.get_subplot(r, c)
        legends[leg] = dict(x=sp.xaxis.domain[1], y=sp.yaxis.domain[1], xanchor="right", yanchor="top",
                            bgcolor="rgba(18,24,32,0.65)", bordercolor="rgba(140,150,165,0.3)", borderwidth=1,
                            font=dict(size=10))
    for k, v in events.items():
        if k in EVENT_LABELS and v <= t[sel][-1] + 1e-9:
            fig.add_vline(x=v, line=dict(color=EVENT_LINE, width=1, dash="dot"))
    if cursor is not None:
        fig.add_vline(x=cursor, line=dict(color="rgba(235,240,248,0.75)", width=1.2, dash="dash"))
    fig.update_annotations(font_size=12, yshift=4)
    fig.update_layout(height=height_per_row * rows + 40, margin=dict(l=40, r=20, t=40, b=40),
                      hovermode="x unified", **legends)
    return fig


def fig_ground_track(d) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Ground track", "Altitude vs downrange"])
    fig.add_trace(go.Scatter(x=d["x"], y=d["y"], mode="lines", name="track", line=dict(color="#4da3ff")), 1, 1)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="triangle-up", size=12,
                                                                     color=NEUTRAL), name="pad"), 1, 1)
    fig.add_trace(go.Scatter(x=[d["x"][-1]], y=[d["y"][-1]], mode="markers", marker=dict(size=10, color="#ff6b4a"),
                             name="end"), 1, 1)
    fig.add_trace(go.Scatter(x=np.hypot(d["x"], d["y"]), y=d["z"], mode="lines", name="profile",
                             line=dict(color="#3ddc97")), 1, 2)
    fig.update_xaxes(title_text="East [m]", row=1, col=1)
    fig.update_yaxes(title_text="North [m]", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="Downrange [m]", row=1, col=2)
    fig.update_yaxes(title_text="Altitude [m]", row=1, col=2)
    fig.update_layout(height=480, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def fig_side_view(geom, cg_stations: dict[str, float], cp_station: float) -> go.Figure:
    """2D profile of the rocket with CG / CP markers."""
    s, r = geom["stations"], geom["radii"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([s, s[::-1]]), y=np.concatenate([r, -r[::-1]]), fill="toself",
                             fillcolor="rgba(154,164,177,0.5)", line=dict(color=NEUTRAL), name="body"))
    fs, fa, rm = geom["motor"]
    fig.add_trace(go.Scatter(x=[fs, fa, fa, fs, fs], y=[-rm, -rm, rm, rm, -rm], line=dict(color="#c0392b", dash="dot"),
                             name="motor"))
    for f in geom["fins"]:
        key = "canards" if f["is_canard"] else ("fincan" if f["on_fincan"] else "fins")
        o = f["outline"]
        for sign in (1, -1):
            fig.add_trace(go.Scatter(x=f["root_le"] + np.append(o[:, 0], o[0, 0]),
                                     y=sign * (f["root_radius"] + np.append(o[:, 1], o[0, 1])),
                                     fill="toself", line=dict(color=COLORS[key]), name=key, showlegend=sign == 1))
    ymax = float(r.max()) * 3
    for name, st in cg_stations.items():
        fig.add_trace(go.Scatter(x=[st], y=[0], mode="markers+text", text=[name], textposition="top center",
                                 marker=dict(symbol="circle-x", size=14, color=NEUTRAL), name=name))
    fig.add_trace(go.Scatter(x=[cp_station], y=[0], mode="markers+text", text=["CP"], textposition="bottom center",
                             marker=dict(symbol="circle-dot", size=14, color="#ff6b4a"), name="CP (M 0.3)"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[-ymax, ymax], title="radius [m]")
    fig.update_xaxes(title="station from nose tip [m]")
    fig.update_layout(height=330, margin=dict(l=40, r=20, t=20, b=40), legend=dict(orientation="h", y=-0.3))
    return fig


def static_curves(rocket, launch_altitude: float = 0.0) -> dict:
    """CP, CN_alpha and zero-lift drag versus Mach at the launch-site atmosphere."""
    from rocket_dt.atmosphere import ISAAtmosphere
    air = ISAAtmosphere().at(launch_altitude)
    mach = np.linspace(0.02, 0.8, 60)
    cna, cp, br = [], [], []
    for m in mach:
        a, c = rocket.aero.static_cp(float(m))
        cna.append(a)
        cp.append(c)
        br.append(rocket.aero.drag_breakdown(float(m), float(m * air.speed_of_sound * rocket.length
                                                            / air.kinematic_viscosity)))
    return {"mach": mach, "cn_alpha": np.array(cna), "cp": np.array(cp),
            **{f"cd_{k}": np.array([b[k] for b in br]) for k in br[0]}}


def vortex_coupling_curves(rocket, mach: float = 0.3, deflection_deg: float = 1.0) -> dict | None:
    """Vortex-induced roll on the fin sets behind the canards, as a fraction of the canard roll
    moment, versus fin-can angle: all canards deflected together, and canard 0 alone."""
    aero = rocket.aero
    if not aero._vortex_targets:
        return None
    ang = np.radians(np.linspace(0.0, 90.0, 37))
    d = np.radians(deflection_deg)
    out = {"angle_deg": np.degrees(ang), "mach": mach}
    for key, defl in (("all", np.full(aero.n_canards, d)), ("single", np.eye(aero.n_canards)[0] * d)):
        m = aero.vortex_roll_map(mach, ang, defl)
        out[key] = m["induced"] / m["canard"]
    return out
