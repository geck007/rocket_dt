"""Standard flight plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rocket_dt.sim import FlightResult


def plot_flight(result: FlightResult, out_dir: str | Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = result.data
    t = d["t"]
    apo = result.events.get("apogee")
    files = []

    fig, ax = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    panels = [
        ("Altitude AGL [m]", [("z", d["z"])]),
        ("Speed [m/s]", [("inertial", d["speed"]), ("air", d["airspeed"])]),
        ("Mach", [("Mach", d["mach"])]),
        ("Angle of attack [deg]", [("AoA", np.degrees(d["aoa"]))]),
        ("Stability margin [cal]", [("margin", d["stability_cal"])]),
        ("Drag coefficient", [("CD0", d["cd"]), ("C_A", d["ca"])]),
        ("Roll rates [deg/s]", [("body p", np.degrees(d["p"])), ("fin can (inertial)", np.degrees(d["fincan_rate_abs"]))]),
        ("Pitch/yaw rates [deg/s]", [("q", np.degrees(d["q"])), ("r", np.degrees(d["r"]))]),
        ("Roll moments [N m]", [("fin-can aero", d["fincan_aero_roll_moment"]),
                                ("canard aero", d["canard_roll_moment"]), ("bearing on can", d["bearing_torque"])]),
    ]
    for axi, (title, lines) in zip(ax.flat, panels):
        for lbl, y in lines:
            axi.plot(t, y, label=lbl, lw=1)
        axi.set_title(title)
        axi.set_xlabel("t [s]")
        axi.grid(alpha=0.3)
        if apo:
            axi.axvline(apo, color="0.6", ls="--", lw=0.8)
        if len(lines) > 1:
            axi.legend(fontsize=8)
    if apo:
        for axi in (ax[1, 0], ax[1, 1]):
            axi.set_xlim(0, apo)
        ax[1, 0].set_ylim(0, min(30, np.nanmax(np.degrees(d["aoa"][t <= apo])) * 1.1 + 1))
    files.append(out_dir / "flight.png")
    fig.savefig(files[-1], dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax[0].plot(d["x"], d["y"])
    ax[0].plot(0, 0, "k^")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].set_xlabel("East [m]")
    ax[0].set_ylabel("North [m]")
    ax[0].set_title("Ground track")
    ax[0].grid(alpha=0.3)
    ax[1].plot(np.hypot(d["x"], d["y"]), d["z"])
    ax[1].set_xlabel("Downrange [m]")
    ax[1].set_ylabel("Altitude [m]")
    ax[1].set_title("Trajectory")
    ax[1].grid(alpha=0.3)
    files.append(out_dir / "trajectory.png")
    fig.savefig(files[-1], dpi=100)
    plt.close(fig)
    return files
