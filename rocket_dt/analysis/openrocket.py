"""Comparison against an OpenRocket CSV export (all variables, with comments)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from rocket_dt.sim import FlightResult, RocketModel


def load_openrocket_csv(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Returns (columns keyed by OpenRocket variable name without unit, events)."""
    header, rows, events = None, [], {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("# Time"):
            header = line[2:].split(",")
        elif line.startswith("# Event"):
            m = re.match(r"# Event (\w+) occurred at t=([\d.]+)", line)
            if m:
                events.setdefault(m.group(1).lower(), float(m.group(2)))
        elif line and not line.startswith("#"):
            rows.append([float(v) for v in line.split(",")])
    if header is None:
        raise ValueError(f"{path}: no '# Time ...' header line (export with comments enabled)")
    data = np.array(rows)
    names = [re.sub(r"\s*\(.*\)\s*$", "", h).strip() for h in header]
    return {n: data[:, i] for i, n in enumerate(names)}, events


def compare(result: FlightResult, rocket: RocketModel, or_csv: str | Path) -> tuple[str, dict]:
    """Text report of key metrics plus a dict of aligned series for plotting."""
    orc, orev = load_openrocket_csv(or_csv)
    d = result.data
    t_or = orc["Time"]
    z0 = d["z"][0]  # OpenRocket altitude starts at 0 at the initial position

    def at_or_times(key):
        return np.interp(t_or, d["t"], d[key])

    lines = ["Metric                         ours        OpenRocket   diff"]

    def row(name, ours, theirs, unit=""):
        diff = (ours - theirs) / theirs * 100 if theirs else float("nan")
        lines.append(f"{name:28s} {ours:10.3f}  {theirs:10.3f} {unit:6s} {diff:+6.2f} %")

    row("apogee [m]", d["z"].max() - z0, orc["Altitude"].max())
    row("time to apogee [s]", result.events["apogee"], orev.get("apogee", np.nan))
    row("max speed [m/s]", d["speed"].max(), orc["Total velocity"].max())
    row("max Mach", d["mach"].max(), orc["Mach number"].max())
    row("liftoff mass [kg]", d["mass"][0], orc["Mass"][0] / 1000)
    row("burnout mass [kg]", d["mass"][-1], orc["Mass"][-1] / 1000)
    row("liftoff CG station [m]", d["cg_station"][0], orc["CG location"][0] / 100)
    row("Ixx (roll) liftoff [kg m2]", d["Ixx"][0], orc["Rotational moment of inertia"][0])
    row("Iyy (pitch) liftoff [kg m2]", d["Iyy"][0], orc["Longitudinal moment of inertia"][0])
    i_or = np.searchsorted(t_or, 1.0)
    row("altitude at t=1 s [m]", float(np.interp(t_or[i_or], d["t"], d["z"])) - z0, orc["Altitude"][i_or])
    i_bo = np.searchsorted(t_or, result.events["burnout"])
    row("speed at burnout [m/s]", float(np.interp(t_or[i_bo], d["t"], d["speed"])), orc["Total velocity"][i_bo])

    # Aero coefficients evaluated at OpenRocket's own (Mach, Re) -> isolates aero from trajectory
    ok = (orc["Mach number"] > 0.05) & np.isfinite(orc["CP location"]) & (t_or < orev.get("apogee", 1e9))
    mach, re_ = orc["Mach number"][ok], orc["Reynolds number"][ok]
    br = [rocket.aero.drag_breakdown(m, r) for m, r in zip(mach, re_)]
    cp = np.array([rocket.aero.static_cp(m)[1] for m in mach])
    aero = {"t": t_or[ok], "mach": mach,
            "cd_ours": np.array([b["total"] for b in br]), "cd_or": orc["Drag coefficient"][ok],
            "cdf_ours": np.array([b["friction"] for b in br]), "cdf_or": orc["Friction drag coefficient"][ok],
            "cdp_ours": np.array([b["pressure"] for b in br]), "cdp_or": orc["Pressure drag coefficient"][ok],
            "cdb_ours": np.array([b["base"] for b in br]), "cdb_or": orc["Base drag coefficient"][ok],
            "cp_ours": cp, "cp_or": orc["CP location"][ok] / 100}
    for k in ("cd", "cdf", "cdp", "cdb"):
        e = aero[f"{k}_ours"] - aero[f"{k}_or"]
        lines.append(f"{k.upper():5s} at OR (M,Re): mean diff {np.mean(e):+.4f}, max |diff| {np.max(np.abs(e)):.4f} "
                     f"(OR mean {np.mean(aero[k + '_or']):.4f})")
    e = aero["cp_ours"] - aero["cp_or"]
    lines.append(f"CP    at OR Mach:  mean diff {np.mean(e) * 100:+.2f} cm, max |diff| {np.max(np.abs(e)) * 100:.2f} cm")

    series = {"t_or": t_or, "or": orc, "ours": d, "z0": z0, "aero": aero, "at_or_times": at_or_times}
    return "\n".join(lines), series


def plot_comparison(series: dict, out_dir: str | Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    orc, d, a = series["or"], series["ours"], series["aero"]
    t_or = series["t_or"]
    apo = float(d["t"][np.argmax(d["z"])])
    files = []

    fig, ax = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    pairs = [
        ("Altitude [m]", d["z"] - series["z0"], orc["Altitude"]),
        ("Total velocity [m/s]", d["speed"], orc["Total velocity"]),
        ("Vertical velocity [m/s]", d["vz"], orc["Vertical velocity"]),
        ("Mass [kg]", d["mass"], orc["Mass"] / 1000),
        ("Zenith angle of body axis [deg]", np.degrees(d["zenith_angle"]), 90 - orc["Vertical orientation"]),
        ("Angle of attack [deg]", np.degrees(d["aoa"]), orc["Angle of attack"]),
    ]
    for axi, (title, ours, theirs) in zip(ax.flat, pairs):
        axi.plot(t_or, theirs, label="OpenRocket", lw=2, alpha=0.6)
        axi.plot(d["t"], ours, label="rocket_dt", lw=1)
        axi.set_title(title)
        axi.set_xlabel("t [s]")
        axi.axvline(apo, color="0.7", lw=0.8, ls="--")
        axi.grid(alpha=0.3)
    ax[2, 1].set_xlim(0, apo)
    ax[2, 1].set_ylim(0, 15)
    ax[0, 0].legend()
    files.append(out_dir / "trajectory_vs_openrocket.png")
    fig.savefig(files[-1], dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axi, k, title in ((ax[0, 0], "cd", "Total CD"), (ax[0, 1], "cdf", "Friction CD"),
                          (ax[1, 0], "cdp", "Pressure CD"), (ax[1, 1], "cdb", "Base CD")):
        axi.plot(a["mach"], a[f"{k}_or"], ".", ms=3, label="OpenRocket", alpha=0.6)
        axi.plot(a["mach"], a[f"{k}_ours"], ".", ms=2, label="rocket_dt")
        axi.set_title(f"{title} vs Mach (at OpenRocket's Reynolds number)")
        axi.set_xlabel("Mach")
        axi.grid(alpha=0.3)
    ax[0, 0].legend()
    files.append(out_dir / "drag_vs_openrocket.png")
    fig.savefig(files[-1], dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax[0].plot(a["mach"], a["cp_or"], ".", ms=3, label="OpenRocket (at actual AoA)", alpha=0.6)
    ax[0].plot(a["mach"], a["cp_ours"], ".", ms=2, label="rocket_dt (small-AoA)")
    ax[0].set_title("CP station [m] vs Mach")
    ax[0].set_xlabel("Mach")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(t_or, orc["Stability margin calibers"], label="OpenRocket", lw=2, alpha=0.6)
    ax[1].plot(d["t"], d["stability_cal"], label="rocket_dt", lw=1)
    ax[1].set_xlim(0, apo)
    ax[1].set_title("Stability margin [cal]")
    ax[1].set_xlabel("t [s]")
    ax[1].grid(alpha=0.3)
    files.append(out_dir / "stability_vs_openrocket.png")
    fig.savefig(files[-1], dpi=110)
    plt.close(fig)
    return files
