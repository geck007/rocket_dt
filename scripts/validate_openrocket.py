"""Run the validation config and compare against the OpenRocket CSV export.

    python scripts/validate_openrocket.py [config] [openrocket_csv] [out_dir]
"""

import sys
from pathlib import Path

from rocket_dt import build_simulation, load_config
from rocket_dt.analysis.openrocket import compare, plot_comparison

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "configs/validation_openrocket.yaml"
    or_csv = sys.argv[2] if len(sys.argv) > 2 else ROOT / "data/all.csv"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "output/validation"
    sim = build_simulation(load_config(cfg_path))
    print("\n".join(sim.rocket.notes), "\n")
    result = sim.run()
    print(result.summary(), "\n")
    report, series = compare(result, sim.rocket, or_csv)
    print(report)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(report + "\n")
    for f in plot_comparison(series, out):
        print("wrote", f)


if __name__ == "__main__":
    main()
