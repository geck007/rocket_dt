"""Rocket digital-twin dashboard.

    streamlit run dashboard/app.py

The YAML config is the base; every value edited in the sidebar becomes an explicit,
listed override of that config, and the result goes through the same strict
validation and build path as the command-line tools.
"""

from __future__ import annotations

import copy
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from rocket_dt import viz
from rocket_dt.builder import build_rocket, build_simulation
from rocket_dt.config import ConfigError, get_by_path, load_raw, parse_config, set_by_path
from rocket_dt.control import RollDeflectionSchedule
from rocket_dt.io.ork import FinSet, load_ork
from rocket_dt.sim import FlightResult

ROOT = Path(__file__).resolve().parents[1]
DEG = 180.0 / math.pi

st.set_page_config(page_title="Rocket Digital Twin", layout="wide")
pio.templates.default = viz.register_dark_template()

st.markdown("""
<style>
.block-container {padding-top: 3.6rem; padding-bottom: 2rem; max-width: 100%;}  /* clears Streamlit's top toolbar */
.dt-header {display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap;
            border-bottom: 1px solid #223041; padding-bottom: .45rem; margin-bottom: .7rem;}
.dt-title {font-size: 1.3rem; font-weight: 600; color: #e6e9ef; letter-spacing: .02em;}
.dt-run {font-size: .95rem; color: #9fb3c8;}
.dt-meta {font-size: .75rem; color: #7d8899; font-family: ui-monospace, Menlo, monospace; margin-left: auto;}
.dt-badge {padding: 2px 8px; border-radius: 3px; font-size: .68rem; font-weight: 700; letter-spacing: .08em;
           text-transform: uppercase; border: 1px solid;}
.dt-ok {background: #10291a; color: #3ddc97; border-color: #24603a;}
.dt-warn {background: #2f2410; color: #ffd166; border-color: #6b5420;}
.dt-bad {background: #2f1212; color: #ff6b6b; border-color: #6b2424;}
.kpi-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: .45rem;
           margin-bottom: .6rem;}
.kpi {background: #111821; border: 1px solid #1f2b39; border-left: 3px solid #4da3ff; border-radius: 4px;
      padding: .4rem .65rem; min-width: 0;}
.kpi .l {font-size: .64rem; text-transform: uppercase; letter-spacing: .08em; color: #7d8899;
         white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.kpi .v {font-size: 1.05rem; font-weight: 600; color: #e6e9ef; font-variant-numeric: tabular-nums;
         white-space: nowrap; line-height: 1.35;}
.kpi .s {font-size: .68rem; color: #7d8899; white-space: nowrap;}
.dt-section {font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; color: #7d8899;
             border-bottom: 1px solid #1f2b39; margin: .9rem 0 .3rem; padding-bottom: .2rem;}
.dt-edits {font-size: .74rem; color: #9fb3c8; background: #111821; border: 1px dashed #2a3a4d;
           border-radius: 4px; padding: .3rem .6rem; margin-bottom: .5rem; font-family: ui-monospace, monospace;}
.stTabs [data-baseweb="tab"] {font-size: .78rem; letter-spacing: .06em; text-transform: uppercase;}
[data-testid="stSidebar"] .dt-side-title {font-size: 1rem; font-weight: 600; letter-spacing: .04em;
                                          color: #e6e9ef; margin-bottom: .3rem;}
</style>
""", unsafe_allow_html=True)

ACCENT = {"traj": "#4da3ff", "speed": "#3ddc97", "aero": "#ffb703", "roll": "#c084fc", "risk": "#ff6b4a"}


def kpi_cards(items) -> None:
    """items: (label, value, sub-text or None, accent colour)."""
    cards = "".join(
        f'<div class="kpi" style="border-left-color:{a}"><div class="l" title="{html.escape(lbl)}">'
        f'{html.escape(lbl)}</div><div class="v">{html.escape(v)}</div>'
        f'<div class="s">{html.escape(s) if s else "&nbsp;"}</div></div>' for lbl, v, s, a in items)
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="dt-section">{html.escape(title)}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- cached computations


@st.cache_data(show_spinner=False, max_entries=24)
def simulate(raw_json: str, schedule: str) -> dict:
    cfg = parse_config(json.loads(raw_json), ROOT / "configs")
    sim = build_simulation(cfg)
    if schedule.strip():
        pts = [tuple(float(v) for v in p.split(":")) for p in schedule.split(",") if p.strip()]
        sim.controller = RollDeflectionSchedule(sim.rocket.aero.n_canards, pts)
    res = sim.run()
    rk = sim.rocket
    m0, m1 = rk.mass.at(0.0), rk.mass.at(rk.motor.burn_time + 1.0)
    return {"data": res.data, "events": res.events, "notes": res.notes, "summary": res.summary(),
            "wall": res.wall_time, "name": res.rocket_name, "geom": viz.rocket_geometry(rk),
            "static": viz.static_curves(rk, cfg.launch.altitude_msl),
            "cg": {"CG liftoff": m0.cg_station, "CG burnout": m1.cg_station},
            "cp03": rk.aero.static_cp(0.3)[1], "ref_length": rk.aero.ref_length,
            "fincan_free": rk.fincan_free, "n_canards": rk.aero.n_canards,
            "vortex_map": viz.vortex_coupling_curves(rk), "sensors": res.sensors}


@st.cache_data(show_spinner=False)
def fin_set_paths(ork_path: str) -> list[str]:
    return [c.path for c in load_ork(ork_path).walk() if isinstance(c, FinSet)]


@st.cache_data(show_spinner=False, max_entries=8)
def montecarlo(raw_json: str, n_runs: int, stop_at_apogee: bool) -> pd.DataFrame:
    from rocket_dt.montecarlo import run_montecarlo
    raw = json.loads(raw_json)
    raw["montecarlo"]["n_runs"] = n_runs
    if stop_at_apogee:
        raw.setdefault("simulation", {})["stop_at"] = "apogee"
    cfg = parse_config(raw, ROOT / "configs")
    return pd.DataFrame(run_montecarlo(cfg))


@st.cache_data(show_spinner=False, max_entries=8)
def openrocket_compare(raw_json: str, schedule: str, csv: str):
    from rocket_dt.analysis.openrocket import compare
    run = simulate(raw_json, schedule)
    rk = build_rocket(parse_config(json.loads(raw_json), ROOT / "configs"))
    res = FlightResult(run["data"], run["events"], run["name"], run["wall"], run["notes"])
    report, s = compare(res, rk, csv)
    s.pop("at_or_times")
    return report, s


def is_openrocket_csv(p: Path) -> bool:
    """A full OpenRocket export (the comparison needs trajectory and aero columns)."""
    needed = ("Altitude (m)", "Total velocity", "Drag coefficient", "CP location", "Reynolds number")
    try:
        with open(p, encoding="utf-8") as fh:
            for _, line in zip(range(40), fh):
                if line.startswith("# Time"):
                    return all(n in line for n in needed)
    except (UnicodeDecodeError, OSError):
        pass
    return False


# ----------------------------------------------------------------------------- sidebar: scenario


st.sidebar.markdown('<div class="dt-side-title">ROCKET DIGITAL TWIN</div>', unsafe_allow_html=True)
configs = sorted((ROOT / "configs").glob("*.yaml"))
names = [c.name for c in configs]
cfg_name = st.sidebar.selectbox("Base configuration", names, index=names.index("baseline.yaml")
                                if "baseline.yaml" in names else 0)
base_raw = load_raw(ROOT / "configs" / cfg_name)
try:
    base_cfg = parse_config(copy.deepcopy(base_raw), ROOT / "configs")
except ConfigError as e:
    st.error(f"Config error in {cfg_name}: {e}")
    st.stop()

raw = copy.deepcopy(base_raw)
ui_overrides: list[str] = []


def knob(label, path, current, widget, **kw):
    """Widget whose value overrides `path` in the config only if it differs from the base."""
    val = widget(label, value=current, key=f"{cfg_name}:{path}", **kw)
    differs = (not math.isclose(val, current, rel_tol=1e-12, abs_tol=1e-12)) if isinstance(current, float) \
        else val != current
    if differs:
        set_by_path(raw, path, val)
        ui_overrides.append(f"{path}: {current} → {val}")
    return val


with st.sidebar.form("scenario"):
    c = base_cfg
    with st.expander("Launch", expanded=True):
        knob("Rail length [m]", "launch.rail_length", c.launch.rail_length, st.number_input, min_value=0.1, step=0.1)
        knob("Rail angle from vertical [°]", "launch.angle_from_vertical_deg", c.launch.angle_from_vertical_deg,
             st.number_input, min_value=0.0, max_value=30.0, step=0.5)
        knob("Rail azimuth [°]", "launch.azimuth_deg", c.launch.azimuth_deg, st.number_input, step=5.0)
        knob("Site altitude MSL [m]", "launch.altitude_msl", c.launch.altitude_msl, st.number_input, step=10.0)
    with st.expander("Wind & atmosphere", expanded=True):
        w = c.environment.wind
        knob("Wind speed [m/s]", "environment.wind.speed", w.speed, st.number_input, min_value=0.0, step=0.5)
        knob("Wind from [°]", "environment.wind.direction_from_deg", w.direction_from_deg, st.number_input, step=5.0)
        knob("Wind profile", "environment.wind.profile", w.profile,
             lambda label, value, **k: st.selectbox(label, ["constant", "power_law"],
                                                    index=["constant", "power_law"].index(value), **k))
        knob("Turbulence", "environment.wind.turbulence.model", w.turbulence.model,
             lambda label, value, **k: st.selectbox(label, ["none", "dryden"],
                                                    index=["none", "dryden"].index(value), **k))
        knob("Turbulence intensity", "environment.wind.turbulence.intensity", w.turbulence.intensity,
             st.number_input, min_value=0.0, max_value=1.0, step=0.05)
        knob("Turbulence seed", "environment.wind.turbulence.seed", w.turbulence.seed, st.number_input, step=1)
        knob("Temperature offset [K]", "environment.atmosphere.temperature_offset_K",
             c.environment.atmosphere.temperature_offset_K, st.number_input, step=1.0)
    if c.rocket.fin_can is not None:
        with st.expander("Fin can & bearing"):
            fc = c.rocket.fin_can
            knob("Fin can free to spin", "rocket.fin_can.enabled", fc.enabled, st.checkbox)
            if fc.roll_inertia is not None:
                knob("Fin-can roll inertia [kg m²]", "rocket.fin_can.roll_inertia", fc.roll_inertia,
                     st.number_input, min_value=1e-4, step=0.001, format="%.4f")
            knob("Bearing Coulomb torque [N m]", "rocket.fin_can.bearing.coulomb_torque", fc.bearing.coulomb_torque,
                 st.number_input, min_value=0.0, step=0.005, format="%.4f")
            knob("Bearing viscous coeff. [N m s/rad]", "rocket.fin_can.bearing.viscous_coeff",
                 fc.bearing.viscous_coeff, st.number_input, min_value=0.0, step=1e-4, format="%.5f")
    with st.expander("Canard vortices"):
        cv = c.aero.canard_vortex
        knob("Canard vortex model", "aero.canard_vortex.model", cv.model,
             lambda label, value, **k: st.selectbox(label, ["none", "vortex"],
                                                    index=["none", "vortex"].index(value), **k))
        knob("Vortex strength factor", "aero.canard_vortex.strength_factor", cv.strength_factor,
             st.number_input, min_value=0.0, max_value=3.0, step=0.1)
        knob("Vortex core radius [m]", "aero.canard_vortex.core_radius", cv.core_radius,
             st.number_input, min_value=0.0005, step=0.001, format="%.4f")
        knob("Body rolls under the vortex", "aero.canard_vortex.roll_lag", cv.roll_lag, st.checkbox)
    with st.expander("Disturbances"):
        dd = c.disturbances
        knob("Thrust misalignment pitch [°]", "disturbances.thrust_misalignment_deg.0", dd.thrust_misalignment_deg[0],
             st.number_input, step=0.05, format="%.3f")
        knob("Thrust misalignment yaw [°]", "disturbances.thrust_misalignment_deg.1", dd.thrust_misalignment_deg[1],
             st.number_input, step=0.05, format="%.3f")
        knob("Thrust scale", "disturbances.thrust_scale", dd.thrust_scale, st.number_input, step=0.01)
        knob("Drag scale", "disturbances.drag_scale", dd.drag_scale, st.number_input, step=0.01)
        for fp in fin_set_paths(str(c.rocket.ork)):
            cur = dd.fin_misalignment_deg.get(fp)
            txt = st.text_input(f"Fin misalignment [°] — {fp}", value=", ".join(f"{v:g}" for v in cur) if cur else "",
                                key=f"{cfg_name}:mis:{fp}", help="One value per fin, comma separated (empty = 0)")
            vals = [float(v) for v in txt.split(",") if v.strip()] if txt.strip() else None
            if vals != cur:
                raw.setdefault("disturbances", {}).setdefault("fin_misalignment_deg", {})
                if vals:
                    raw["disturbances"]["fin_misalignment_deg"][fp] = vals
                else:
                    raw["disturbances"]["fin_misalignment_deg"].pop(fp, None)
                ui_overrides.append(f"fin misalignment {fp}: {cur} → {vals}")
    with st.expander("Sensors & canard servos"):
        sc_ = c.sensors
        for k, imu in enumerate(sc_.imu):
            knob(f"IMU '{imu.name}'", f"sensors.imu.{k}.enabled", imu.enabled, st.checkbox)
        if sc_.baro is not None:
            knob("Barometer", "sensors.baro.enabled", sc_.baro.enabled, st.checkbox)
        if sc_.gnss is not None:
            knob("GNSS", "sensors.gnss.enabled", sc_.gnss.enabled, st.checkbox)
        if sc_.imu or sc_.baro or sc_.gnss:
            knob("Sensor error seed", "sensors.seed", sc_.seed, st.number_input, step=1)
        a_ = c.actuators.canards
        knob("Servo model", "actuators.canards.model", a_.model,
             lambda label, value, **k: st.selectbox(label, ["ideal", "servo"],
                                                    index=["ideal", "servo"].index(value), **k))
        knob("Servo natural frequency [Hz]", "actuators.canards.natural_frequency_hz", a_.natural_frequency_hz,
             st.number_input, min_value=0.5, step=1.0)
        knob("Servo damping ratio", "actuators.canards.damping_ratio", a_.damping_ratio, st.number_input,
             min_value=0.05, step=0.05)
        knob("Servo stiffness [N m/rad]", "actuators.canards.stiffness", a_.stiffness, st.number_input,
             min_value=0.1, step=5.0)
        knob("Servo deadband [°]", "actuators.canards.deadband_deg", a_.deadband_deg, st.number_input,
             min_value=0.0, step=0.05, format="%.3f")
        knob("Command resolution [°]", "actuators.canards.resolution_deg", a_.resolution_deg, st.number_input,
             min_value=0.0, step=0.01, format="%.3f")
        knob("Backlash [°]", "actuators.canards.backlash_deg", a_.backlash_deg, st.number_input,
             min_value=0.0, step=0.05, format="%.3f")
    with st.expander("Canards (open loop)"):
        schedule = st.text_input("Roll deflection schedule  t:deg, …", value="", key=f"{cfg_name}:schedule",
                                 help="e.g. 0:0, 3:0, 3.01:0.5, 6:0.5, 6.01:0 — linear interpolation. Empty = canards at 0.")
    with st.expander("Integrator"):
        s = c.simulation
        knob("Time step [s]", "simulation.dt", s.dt,
             lambda label, value, **k: st.select_slider(label, [0.0005, 0.001, 0.002, 0.005], value=value, **k))
        knob("Stop at", "simulation.stop_at", s.stop_at,
             lambda label, value, **k: st.radio(label, ["apogee", "ground"], index=["apogee", "ground"].index(value),
                                                horizontal=True, **k))
        knob("Max time [s]", "simulation.t_max", s.t_max, st.number_input, step=10.0)
    run_label = st.text_input("Run label", value="", placeholder="optional name for the run history")
    submitted = st.form_submit_button("Run simulation", type="primary", width="stretch")

raw_json = json.dumps(raw, sort_keys=True)
if "runs" not in st.session_state:
    st.session_state.runs = {}  # label -> (raw_json, schedule)
    submitted = True  # run the base configuration on first load

if submitted:
    try:
        parse_config(json.loads(raw_json), ROOT / "configs")
    except (ConfigError, ValueError) as e:
        st.error(f"Invalid scenario: {e}")
        st.stop()
    label = run_label.strip() or f"#{len(st.session_state.runs) + 1} {cfg_name.removesuffix('.yaml')}" + \
        (f" ({len(ui_overrides)} edits)" if ui_overrides else "")
    st.session_state.runs[label] = (raw_json, schedule, list(ui_overrides))
    st.session_state.current = label

runs = st.session_state.runs
current = st.sidebar.selectbox("Show run", list(runs), index=list(runs).index(st.session_state.current))
st.session_state.current = current
cur_json, cur_schedule, cur_overrides = runs[current]
with st.spinner("Simulating… (first run of a scenario takes 5–20 s, then it is cached)"):
    try:
        R = simulate(cur_json, cur_schedule)
    except Exception as e:  # show model errors (e.g. divergence) instead of a stack trace
        st.error(f"Simulation failed: {type(e).__name__}: {e}")
        st.stop()

d, ev, geom = R["data"], R["events"], R["geom"]
t = d["t"]
i_apo = int(np.argmax(d["z"]))
fast = d["airspeed"][: i_apo + 1] > 20.0

# ----------------------------------------------------------------------------- header + KPIs

min_margin = float(np.nanmin(d["stability_cal"][: i_apo + 1]))
warn = [n for n in R["notes"] if n.startswith("WARNING")]
if warn or min_margin <= 0:
    badge = '<span class="dt-badge dt-bad">Unstable</span>'
elif min_margin < 1.0:
    badge = '<span class="dt-badge dt-warn">Marginal stability</span>'
else:
    badge = '<span class="dt-badge dt-ok">Stable</span>'
fincan_mode = "fin can free" if R["fincan_free"] else "fin can locked"
st.markdown(
    f'<div class="dt-header"><span class="dt-title">Flight analysis</span>'
    f'<span class="dt-run">{html.escape(current)}</span>{badge}'
    f'<span class="dt-meta">{html.escape(fincan_mode)} · t_end {t[-1]:.1f} s · '
    f'{len(t)} samples · computed in {R["wall"]:.1f} s</span></div>', unsafe_allow_html=True)
kpi_cards([
    ("Apogee", f"{d['z'][i_apo]:.0f} m", f"{d['z'][i_apo] / 0.3048:.0f} ft AGL", ACCENT["traj"]),
    ("Time to apogee", f"{ev.get('apogee', float('nan')):.2f} s", None, ACCENT["traj"]),
    ("Max speed", f"{d['speed'].max():.1f} m/s", f"Mach {d['mach'].max():.3f}", ACCENT["speed"]),
    ("Max accel", f"{d['accel_axial'].max() / 9.81:.1f} g", f"{d['accel_axial'].max():.0f} m/s²", ACCENT["speed"]),
    ("Rail exit", f"{ev.get('rail_exit_speed', float('nan')):.1f} m/s",
     f"t = {ev.get('rail_exit', float('nan')):.3f} s", ACCENT["speed"]),
    ("Max AoA", f"{np.degrees(d['aoa'][: i_apo + 1][fast].max()):.2f}°", "airspeed > 20 m/s", ACCENT["aero"]),
    ("Min margin", f"{min_margin:.2f} cal", "ascent", ACCENT["aero"]),
    ("Max roll, body", f"{np.degrees(np.abs(d['p']).max()):.0f}°/s", None, ACCENT["roll"]),
    ("Max roll, fin can", f"{np.degrees(np.abs(d['fincan_rate_abs']).max()):.0f}°/s", "inertial", ACCENT["roll"]),
])
for w_ in warn:
    st.warning(w_)
if cur_overrides or cur_schedule:
    edits = cur_overrides + ([f"canard schedule {cur_schedule}"] if cur_schedule else [])
    st.markdown(f'<div class="dt-edits">scenario edits vs base config — {html.escape("; ".join(edits))}</div>',
                unsafe_allow_html=True)

tabs = st.tabs(["Launch", "Telemetry", "Explorer", "Trajectory", "Vehicle & model", "Monte Carlo", "OpenRocket"])

# ----------------------------------------------------------------------------- launch


with tabs[0]:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    n_frames = c1.slider("Animation frames", 40, 300, 140, 10)
    span = c2.radio("Span", ["Ascent", "Whole flight", "Boost (first 4 s)"], horizontal=True)
    rscale = c3.slider("Rocket size in 3D view (× true scale)", 0, 200, 0, 1,
                       help="0 = automatic (≈8 % of apogee)")
    view = c4.radio("View", ["Flight", "Vehicle close-up", "Both"], index=2, horizontal=True)
    t_end = {"Ascent": ev.get("apogee", t[-1]), "Whole flight": t[-1], "Boost (first 4 s)": 4.0}[span]
    if view in ("Flight", "Both"):
        st.plotly_chart(viz.fig_launch(d, geom, ev, n_frames, t_end, rocket_scale=rscale or None),
                        width="stretch")
    if view in ("Vehicle close-up", "Both"):
        st.caption("Close-up at the CG: grey body, red canards, blue fins on the free-spinning fin can "
                   "(watch them turn relative to the canards), green = velocity relative to the air.")
        st.plotly_chart(viz.fig_attitude(d, geom, n_frames, t_end), width="stretch")

    section("Flight events")
    st.dataframe(pd.DataFrame([{"event": k, "t [s]": round(v, 3),
                                "altitude [m]": round(float(np.interp(v, t, d["z"])), 1),
                                "speed [m/s]": round(float(np.interp(v, t, d["speed"])), 2)}
                               for k, v in ev.items() if k in viz.EVENT_LABELS]),
                 hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- telemetry

GROUPS = {
    "Trajectory": [
        ("Altitude AGL [m]", [("z", d["z"])]),
        ("Speed [m/s]", [("inertial", d["speed"]), ("airspeed", d["airspeed"]), ("vertical", d["vz"])]),
        ("Axial acceleration [g]", [("a_x", d["accel_axial"] / 9.81)]),
        ("Dynamic pressure [kPa]", [("q", d["q_dyn"] / 1000)]),
        ("Position [m]", [("East", d["x"]), ("North", d["y"])]),
        ("Mach", [("Mach", d["mach"])]),
    ],
    "Aerodynamics": [
        ("Angle of attack [°]", [("AoA", d["aoa"] * DEG)]),
        ("Drag coefficient (zero-lift)", [("total", d["cd"]), ("friction", d["cd_friction"]),
                                          ("pressure", d["cd_pressure"]), ("base", d["cd_base"])]),
        ("Stability margin [cal]", [("margin", d["stability_cal"])]),
        ("CP / CG station [m]", [("CP", d["cp_station"]), ("CG", d["cg_station"])]),
        ("Axial force coeff. C_A / CN_α [1/rad]", [("C_A", d["ca"]), ("CN_α / 10", d["cn_alpha"] / 10)]),
        ("Reynolds number [×10⁶]", [("Re", d["reynolds"] / 1e6)]),
    ],
    "Attitude & rates": [
        ("Zenith angle of axis [°]", [("zenith", d["zenith_angle"] * DEG)]),
        ("Heading of axis [°]", [("heading", d["heading"] * DEG)]),
        ("Pitch / yaw rates [°/s]", [("q", d["q"] * DEG), ("r", d["r"] * DEG)]),
        ("Body roll rate [°/s]", [("p", d["p"] * DEG)]),
        ("Roll angle [°]", [("roll", d["roll_angle"] * DEG)]),
        ("Attitude quaternion", [("w", d["qw"]), ("x", d["qx"]), ("y", d["qy"]), ("z", d["qz"])]),
    ],
    "Roll & fin can": [
        ("Roll rates [°/s]", [("body", d["p"] * DEG), ("fin can (inertial)", d["fincan_rate_abs"] * DEG),
                              ("fin can rel. body", d["fincan_rate_rel"] * DEG)]),
        ("Roll moments [N m]", [("fin-can aero", d["fincan_aero_roll_moment"]),
                                ("canard aero", d["canard_roll_moment"]), ("bearing → can", d["bearing_torque"]),
                                ("canard vortices → fins", d["vortex_roll_moment"]), ("total aero", d["aero_mx"])]),
        ("Vortex-induced fin roll per canard [N m]",
         [(f"canard {k}", d[f"vortex_roll_{k}"]) for k in range(R["n_canards"])]),
        ("Canard deflection [°]", [(f"canard {k}", d[f"canard_{k}"] * DEG) for k in range(R["n_canards"])]),
        ("Fin-can angle rel. body [rev]", [("angle", d["fincan_angle_rel"] / (2 * np.pi))]),
    ],
    "Forces & moments": [
        ("Thrust / drag / normal force [N]", [("thrust", d["thrust"]), ("drag", d["drag_force"]),
                                               ("normal", d["normal_force"])]),
        ("Aero force, body axes [N]", [("Fx", d["aero_fx"]), ("Fy", d["aero_fy"]), ("Fz", d["aero_fz"])]),
        ("Aero moment about CG [N m]", [("Mx", d["aero_mx"]), ("My", d["aero_my"]), ("Mz", d["aero_mz"])]),
    ],
    "Mass properties": [
        ("Mass [kg]", [("mass", d["mass"])]),
        ("CG station [m]", [("CG", d["cg_station"])]),
        ("Roll inertia Ixx [kg m²]", [("Ixx", d["Ixx"])]),
        ("Pitch inertia Iyy [kg m²]", [("Iyy", d["Iyy"])]),
    ],
    "Environment": [
        ("Wind [m/s]", [("East", d["wind_e"]), ("North", d["wind_n"])]),
        ("Air density [kg/m³]", [("ρ", d["air_density"])]),
        ("Air temperature [K]", [("T", d["air_temperature"])]),
    ],
}



def _hold(ts, vs):
    """Sensor stream resampled onto the log times (sample and hold)."""
    i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
    out = np.asarray(vs, dtype=float)[i]
    return np.where(t < ts[0], np.nan, out)


S = R.get("sensors") or {}
if S or any(f"canard_cmd_{k}" in d for k in range(R["n_canards"])):
    sens = [("Canard 0: command vs actual [°]", [("command", d["canard_cmd_0"] * DEG), ("actual", d["canard_0"] * DEG)]),
            ("Canard hinge moments [N m]", [(f"canard {k}", d[f"hinge_moment_{k}"]) for k in range(R["n_canards"])])]
    imus = [k for k, v in S.items() if "gx" in v]
    if imus:
        g0 = S[imus[0]]
        sens += [("Roll rate: gyros vs truth [°/s]", [("truth", _hold(g0["t"], g0["true_gx"]) * DEG)]
                  + [(k, _hold(S[k]["t"], S[k]["gx"]) * DEG) for k in imus]),
                 ("Pitch rate: gyros vs truth [°/s]", [("truth", _hold(g0["t"], g0["true_gy"]) * DEG)]
                  + [(k, _hold(S[k]["t"], S[k]["gy"]) * DEG) for k in imus]),
                 ("Axial specific force: accels vs truth [m/s²]", [("truth", _hold(g0["t"], g0["true_ax"]))]
                  + [(k, _hold(S[k]["t"], S[k]["ax"])) for k in imus])]
    if "baro" in S:
        b = S["baro"]
        from rocket_dt.sensors import pressure_altitude
        sens.append(("Baro altitude (zeroed on the pad) vs truth [m]",
                     [("truth", _hold(b["t"], b["true_altitude"] - b["true_altitude"][0])),
                      ("baro", _hold(b["t"], pressure_altitude(b["pressure"], b["pressure"][0])))]))
    if "gnss" in S:
        n = S["gnss"]
        sens += [("GNSS altitude vs truth [m]", [("truth", _hold(n["t"], n["true_u"])), ("GNSS", _hold(n["t"], n["u"]))]),
                 ("GNSS fix valid", [("valid", _hold(n["t"], n["valid"]))])]
    GROUPS["Sensors & servos"] = sens

with tabs[1]:
    c1, c2, c3 = st.columns([2, 3, 1])
    groups = c1.multiselect("Groups", list(GROUPS), default=["Trajectory", "Aerodynamics", "Roll & fin can"])
    t_lo, t_hi = c2.slider("Time window [s]", 0.0, float(t[-1]), (0.0, float(ev.get("apogee", t[-1]))), 0.1)
    cursor = c3.number_input("Cursor t [s]", 0.0, float(t[-1]), float(min(ev.get("burnout", 0.0), t[-1])), 0.1)
    for g in groups:
        section(g)
        fig = viz.fig_panels(d, ev, GROUPS[g], cols=3 if len(GROUPS[g]) > 4 else 2, t_max=t_hi, cursor=cursor)
        fig.update_xaxes(range=[t_lo, t_hi])
        st.plotly_chart(fig, width="stretch")
    i_c = int(min(np.searchsorted(t, cursor), len(t) - 1))
    with st.expander(f"Full state at t = {t[i_c]:.3f} s"):
        st.dataframe(pd.DataFrame({"value": {k: v[i_c] for k, v in d.items()}}), width="stretch")
    if S:
        section("Full-rate sensor logs (measurements and true values)")
        cols = st.columns(len(S))
        for col, (name, log) in zip(cols, S.items()):
            col.download_button(f"{name}.csv", pd.DataFrame(log).to_csv(index=False).encode(),
                                f"sensor_{name}.csv", "text/csv", key=f"dl_sensor_{name}")

# ----------------------------------------------------------------------------- explorer

with tabs[2]:
    st.markdown("Plot any logged variable; overlay runs from the history (sidebar → *Show run*).")
    keys = list(d)
    c1, c2, c3 = st.columns([3, 1, 2])
    ys = c1.multiselect("Variables", keys, default=["z", "speed"])
    xk = c2.selectbox("x axis", keys, index=keys.index("t"))
    others = c3.multiselect("Overlay runs", [r for r in runs if r != current])
    sep = st.checkbox("One subplot per variable", value=True)
    series = [(current, d)] + [(r, simulate(runs[r][0], runs[r][1])["data"]) for r in others]
    if ys:
        fig = make_subplots(rows=len(ys) if sep else 1, cols=1, shared_xaxes=True, subplot_titles=ys if sep else None)
        for n, yk in enumerate(ys):
            for k, (lbl, dd_) in enumerate(series):
                fig.add_trace(go.Scatter(x=dd_[xk], y=dd_[yk], mode="lines", name=f"{yk} · {lbl}",
                                         line=dict(dash=["solid", "dash", "dot", "dashdot"][k % 4])),
                              row=n + 1 if sep else 1, col=1)
        fig.update_layout(height=(250 * len(ys) if sep else 450) + 60, hovermode="x unified",
                          margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig, width="stretch")
    df = pd.DataFrame(d)
    st.download_button("Download full log (CSV)", df.to_csv(index=False).encode(), f"{current}.csv", "text/csv")
    with st.expander("Raw log table"):
        st.dataframe(df, width="stretch", height=300)

# ----------------------------------------------------------------------------- trajectory

with tabs[3]:
    st.plotly_chart(viz.fig_ground_track(d), width="stretch")
    kpi_cards([("End point East", f"{d['x'][-1]:.1f} m", None, ACCENT["traj"]),
               ("End point North", f"{d['y'][-1]:.1f} m", None, ACCENT["traj"]),
               ("Distance from pad", f"{math.hypot(d['x'][-1], d['y'][-1]):.1f} m", None, ACCENT["traj"]),
               ("Flight time", f"{t[-1]:.1f} s", None, ACCENT["traj"])])

# ----------------------------------------------------------------------------- vehicle & model

with tabs[4]:
    st.plotly_chart(viz.fig_side_view(geom, R["cg"], R["cp03"]), width="stretch")
    sc = R["static"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=["CP station [m] vs Mach", "CN_α [1/rad] vs Mach",
                                                        "Zero-lift drag vs Mach (sea-level Re)"])
    fig.add_trace(go.Scatter(x=sc["mach"], y=sc["cp"], name="CP"), 1, 1)
    for k, v in R["cg"].items():
        fig.add_hline(y=v, line=dict(dash="dot", color="grey"), annotation_text=k, row=1, col=1)
    fig.add_trace(go.Scatter(x=sc["mach"], y=sc["cn_alpha"], name="CN_α"), 1, 2)
    for k in ("total", "friction", "pressure", "base"):
        fig.add_trace(go.Scatter(x=sc["mach"], y=sc[f"cd_{k}"], name=f"CD {k}"), 1, 3)
    fig.update_xaxes(title_text="Mach")
    fig.update_layout(height=360, margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig, width="stretch")
    vm = R.get("vortex_map")
    if vm is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vm["angle_deg"], y=vm["all"], name="all canards deflected"))
        fig.add_trace(go.Scatter(x=vm["angle_deg"], y=vm["single"], name="canard 0 alone", line=dict(dash="dash")))
        fig.update_xaxes(title="fin-can angle relative to canards [°]")
        fig.update_yaxes(title="induced fin roll / canard roll")
        fig.update_layout(title=f"Canard vortex → fin roll coupling (M {vm['mach']:.1f}, α = 0)", height=320,
                          margin=dict(l=40, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        section("Build log — sources, overrides, checks")
        for n in R["notes"]:
            (st.warning if n.startswith("WARNING") else st.info if n.startswith("OVERRIDE") else st.text)(n)
    with c2:
        section("Effective configuration")
        eff = json.loads(cur_json)
        st.code(yaml.safe_dump(eff, sort_keys=False), language="yaml")
        st.download_button("Download as YAML", yaml.safe_dump(eff, sort_keys=False).encode(),
                           f"{current}.yaml", "text/yaml")

# ----------------------------------------------------------------------------- monte carlo

with tabs[5]:
    cur_raw = json.loads(cur_json)
    if "montecarlo" not in cur_raw:
        st.info("This configuration has no `montecarlo` section.")
    else:
        st.markdown("Dispersions (from the config):")
        st.json(cur_raw["montecarlo"].get("dispersions", {}), expanded=False)
        c1, c2, c3 = st.columns([1, 1, 2])
        n_runs = c1.number_input("Runs", 4, 2000, int(cur_raw["montecarlo"].get("n_runs", 50)), 4)
        to_apogee = c2.checkbox("Stop at apogee (faster)", value=True)
        if c3.button("Run Monte Carlo", type="primary") or st.session_state.get("mc_key") == (cur_json, n_runs,
                                                                                           to_apogee):
            st.session_state.mc_key = (cur_json, n_runs, to_apogee)
            with st.spinner(f"Running {n_runs} simulations in parallel…"):
                mc = montecarlo(cur_json, int(n_runs), to_apogee)
            kpi_cards([("Apogee", f"{mc.apogee.mean():.0f} ± {mc.apogee.std():.0f} m", f"{len(mc)} runs, 1σ",
                        ACCENT["traj"]),
                       ("Apogee range", f"{mc.apogee.min():.0f} – {mc.apogee.max():.0f} m", None, ACCENT["traj"]),
                       ("Max AoA, worst", f"{mc.max_aoa_deg.max():.1f}°", None, ACCENT["aero"]),
                       ("Min margin, worst", f"{mc.min_margin_cal.min():.2f} cal", None, ACCENT["aero"]),
                       ("Max body roll, worst", f"{mc.max_body_roll_dps.max():.0f}°/s", None, ACCENT["roll"]),
                       ("Rail exit, slowest", f"{mc.rail_exit_speed.min():.1f} m/s", None, ACCENT["speed"])])
            fig = make_subplots(rows=1, cols=3, subplot_titles=["End points (colour = apogee)", "Apogee [m]",
                                                                "Max roll rate during ascent [°/s]"])
            fig.add_trace(go.Scatter(x=mc.east_end, y=mc.north_end, mode="markers",
                                     marker=dict(color=mc.apogee, colorscale="Viridis", showscale=True, size=7,
                                                 colorbar=dict(x=0.3, len=0.9)),
                                     text=[f"run {r}" for r in mc.run], name="runs"), 1, 1)
            fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="triangle-up", size=12,
                                                                             color="#aab4c3"), name="pad"), 1, 1)
            fig.add_trace(go.Histogram(x=mc.apogee, name="apogee", nbinsx=25), 1, 2)
            fig.add_trace(go.Histogram(x=mc.max_body_roll_dps, name="body", opacity=0.7), 1, 3)
            fig.add_trace(go.Histogram(x=mc.max_fincan_roll_dps, name="fin can", opacity=0.7), 1, 3)
            fig.update_layout(barmode="overlay", height=430, margin=dict(l=40, r=20, t=40, b=40))
            fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
            st.plotly_chart(fig, width="stretch")
            num = [c for c in mc.columns if c not in ("run",)]
            c1, c2 = st.columns(2)
            xa = c1.selectbox("Sensitivity: input", [c for c in num if c not in
                                                     ("apogee", "t_apogee", "east_end", "north_end")], index=0)
            ya = c2.selectbox("Sensitivity: output", ["apogee", "max_aoa_deg", "max_body_roll_dps",
                                                      "max_fincan_roll_dps", "rail_exit_speed", "east_end",
                                                      "north_end"])
            st.plotly_chart(go.Figure(go.Scatter(x=mc[xa], y=mc[ya], mode="markers")).update_layout(
                xaxis_title=xa, yaxis_title=ya, height=350, margin=dict(l=40, r=20, t=20, b=40)),
                width="stretch")
            st.dataframe(mc, width="stretch", height=260)
            st.download_button("Download Monte Carlo results", mc.to_csv(index=False).encode(), "montecarlo.csv")

# ----------------------------------------------------------------------------- openrocket

with tabs[6]:
    csvs = [p for p in sorted((ROOT / "data").glob("*.csv")) if is_openrocket_csv(p)]
    if not csvs:
        st.info("No OpenRocket CSV exports found in data/.")
    else:
        st.markdown("Compare the current run against an OpenRocket export. Meaningful when the run uses the "
                    "same design (e.g. `validation_openrocket.yaml` for `all.csv`).")
        pick = st.selectbox("OpenRocket CSV", [p.name for p in csvs])
        if st.button("Compare"):
            st.session_state.or_pick = (current, pick)
        if st.session_state.get("or_pick") == (current, pick):
            with st.spinner("Comparing…"):
                report, s = openrocket_compare(cur_json, cur_schedule, str(ROOT / "data" / pick))
            st.code(report)
            orc, ours, a = s["or"], s["ours"], s["aero"]
            pairs = [("Altitude [m]", ours["z"] - s["z0"], orc["Altitude"]),
                     ("Total velocity [m/s]", ours["speed"], orc["Total velocity"]),
                     ("Mass [kg]", ours["mass"], orc["Mass"] / 1000),
                     ("Angle of attack [°]", np.degrees(ours["aoa"]), orc["Angle of attack"]),
                     ("Stability margin [cal]", ours["stability_cal"], orc["Stability margin calibers"]),
                     ("Zenith angle [°]", np.degrees(ours["zenith_angle"]), 90 - orc["Vertical orientation"])]
            fig = make_subplots(rows=2, cols=3, subplot_titles=[p[0] for p in pairs])
            for n, (title, y1, y2) in enumerate(pairs):
                r_, c_ = n // 3 + 1, n % 3 + 1
                fig.add_trace(go.Scatter(x=s["t_or"], y=y2, name="OpenRocket", line=dict(color="#ff6b4a", width=3),
                                         opacity=0.5, showlegend=n == 0), r_, c_)
                fig.add_trace(go.Scatter(x=ours["t"], y=y1, name="rocket_dt", line=dict(color="#4da3ff", width=1.5),
                                         showlegend=n == 0), r_, c_)
            fig.update_layout(height=620, margin=dict(l=40, r=20, t=40, b=40), hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Drag coefficient vs Mach", "CP station vs Mach"])
            fig.add_trace(go.Scatter(x=a["mach"], y=a["cd_or"], mode="markers", name="OpenRocket CD",
                                     marker=dict(size=3, color="#ff6b4a")), 1, 1)
            fig.add_trace(go.Scatter(x=a["mach"], y=a["cd_ours"], mode="markers", name="rocket_dt CD",
                                     marker=dict(size=3, color="#4da3ff")), 1, 1)
            fig.add_trace(go.Scatter(x=a["mach"], y=a["cp_or"], mode="markers", name="OpenRocket CP",
                                     marker=dict(size=3, color="#ff6b4a")), 1, 2)
            fig.add_trace(go.Scatter(x=a["mach"], y=a["cp_ours"], mode="markers", name="rocket_dt CP",
                                     marker=dict(size=3, color="#4da3ff")), 1, 2)
            fig.update_xaxes(title_text="Mach")
            fig.update_layout(height=380, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig, width="stretch")

