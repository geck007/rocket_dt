"""Run one simulation from a config (with the controller named in its ``controller`` section).

    python scripts/run_sim.py configs/vertical_hold.yaml [--apogee] [--locked]
        [--canard-schedule "t0:deg0,t1:deg1,..."] [--out output/run]

--canard-schedule replaces the controller with an open-loop roll deflection schedule.
Writes flight.csv, one sensor_<name>.csv per sensor and plots to the output directory.
"""

import argparse
from pathlib import Path

from rocket_dt import build_simulation, load_config
from rocket_dt.analysis.plots import plot_flight
from rocket_dt.control import RollDeflectionSchedule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--apogee", action="store_true", help="stop at apogee")
    ap.add_argument("--locked", action="store_true", help="lock the fin can to the body")
    ap.add_argument("--canard-schedule", help="open-loop roll deflection, e.g. '0:0,3:0,3.05:5,6:5,6.05:0'")
    ap.add_argument("--out", default=None, help="output directory for CSV and plots")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.apogee:
        cfg.simulation.stop_at = "apogee"
    if a.locked and cfg.rocket.fin_can:
        cfg.rocket.fin_can.enabled = False
    sim = build_simulation(cfg)
    if a.canard_schedule:
        pts = [tuple(float(v) for v in p.split(":")) for p in a.canard_schedule.split(",")]
        sim.controller = RollDeflectionSchedule(sim.rocket.aero.n_canards, pts)
    print("\n".join(sim.notes), "\n")
    result = sim.run()
    print(result.summary())

    out = Path(a.out) if a.out else Path("output") / Path(a.config).stem
    out.mkdir(parents=True, exist_ok=True)
    result.save_csv(out / "flight.csv")
    for f in result.save_sensor_csvs(out):
        print("wrote", f)
    for f in plot_flight(result, out):
        print("wrote", f)


if __name__ == "__main__":
    main()
