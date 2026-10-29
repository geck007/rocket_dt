"""Monte Carlo dispersion analysis using the config's 'montecarlo' section.

    python scripts/montecarlo.py configs/baseline.yaml [--runs N] [--apogee] [--out output/mc]
"""

import argparse
from pathlib import Path

import numpy as np

from rocket_dt.config import load_config
from rocket_dt.montecarlo import plot_results, run_montecarlo, save_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--runs", type=int)
    ap.add_argument("--apogee", action="store_true", help="stop each run at apogee")
    ap.add_argument("--out", default="output/montecarlo")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.runs:
        cfg.montecarlo.n_runs = a.runs
    if a.apogee:
        cfg.raw.setdefault("simulation", {})["stop_at"] = "apogee"
    results = run_montecarlo(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_results(results, out / "runs.csv")
    apo = np.array([r["apogee"] for r in results])
    print(f"{len(results)} runs: apogee {apo.mean():.1f} +- {apo.std():.1f} m (min {apo.min():.0f}, max {apo.max():.0f})")
    for k in ("max_aoa_deg", "min_margin_cal", "rail_exit_speed", "max_body_roll_dps", "max_fincan_roll_dps"):
        v = np.array([r[k] for r in results])
        print(f"  {k:22s} mean {v.mean():8.2f}  min {v.min():8.2f}  max {v.max():8.2f}")
    print("wrote", plot_results(results, out))


if __name__ == "__main__":
    main()
