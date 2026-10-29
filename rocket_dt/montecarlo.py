"""Monte Carlo dispersion runs.

Dispersions are applied to the *raw* config dict before it is parsed, so every run
goes through exactly the same validation and build path as a single run.

    montecarlo:
      n_runs: 100
      seed: 1
      dispersions:
        environment.wind.speed: {dist: uniform, low: 0, high: 6}
        disturbances.thrust_scale: {dist: normal, mean: 1.0, sigma: 0.03}
"""

from __future__ import annotations

import copy
import multiprocessing as mp
from pathlib import Path

import numpy as np

from rocket_dt.config import Config, ConfigError, parse_config, set_by_path


def sample(spec: dict, rng: np.random.Generator):
    """One draw; with ``size: n`` a list of n independent draws (e.g. per-fin errors)."""
    dist = spec.get("dist")
    n = spec.get("size")
    if dist == "normal":
        v = rng.normal(spec["mean"], spec["sigma"], n)
    elif dist == "uniform":
        v = rng.uniform(spec["low"], spec["high"], n)
    else:
        raise ConfigError(f"Unknown dispersion distribution {spec!r} (use normal or uniform)")
    return [float(x) for x in v] if n else float(v)


def _run_one(args) -> dict:
    from rocket_dt.builder import build_simulation

    i, raw, base_dir, values = args
    cfg = parse_config(raw, Path(base_dir))
    sim = build_simulation(cfg)
    r = sim.run()
    d = r.data
    i_apo = int(np.argmax(d["z"]))
    fast = d["airspeed"][:i_apo + 1] > 20.0
    out = {"run": i, **values,
           "apogee": float(d["z"][i_apo]), "t_apogee": r.events.get("apogee", np.nan),
           "rail_exit_speed": r.events.get("rail_exit_speed", np.nan),
           "max_mach": float(d["mach"].max()),
           "max_aoa_deg": float(np.degrees(d["aoa"][:i_apo + 1][fast].max())),
           "min_margin_cal": float(np.nanmin(d["stability_cal"][:i_apo + 1])),
           "max_body_roll_dps": float(np.degrees(np.abs(d["p"][:i_apo + 1]).max())),
           "max_fincan_roll_dps": float(np.degrees(np.abs(d["fincan_rate_abs"][:i_apo + 1]).max())),
           "east_end": float(d["x"][-1]), "north_end": float(d["y"][-1])}
    return out


def run_montecarlo(cfg: Config) -> list[dict]:
    mc = cfg.montecarlo
    if mc is None:
        raise ConfigError("Config has no 'montecarlo' section")
    rng = np.random.default_rng(mc.seed)
    base_dir = str(cfg.source.parent) if cfg.source else "."
    jobs = []
    for i in range(mc.n_runs):
        raw = copy.deepcopy(cfg.raw)
        raw.pop("montecarlo", None)
        values = {}
        for path, spec in mc.dispersions.items():
            v = sample(spec, rng)
            set_by_path(raw, path, v)
            if isinstance(v, list):
                values.update({f"{path}[{k}]": x for k, x in enumerate(v)})
            else:
                values[path] = v
        # independent turbulence realisation per run
        set_by_path(raw, "environment.wind.turbulence.seed", int(rng.integers(0, 2 ** 31)))
        jobs.append((i, raw, base_dir, values))
    parse_config(jobs[0][1], Path(base_dir))  # fail fast on bad dispersion paths
    with mp.Pool(mc.processes) as pool:
        return pool.map(_run_one, jobs)


def save_results(results: list[dict], path: str | Path) -> None:
    keys = list(results[0])
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in results:
            fh.write(",".join(f"{r[k]:.6g}" for k in keys) + "\n")


def plot_results(results: list[dict], out_dir: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    e = np.array([r["east_end"] for r in results])
    n = np.array([r["north_end"] for r in results])
    apo = np.array([r["apogee"] for r in results])
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    ax[0].scatter(e, n, s=10)
    ax[0].plot(0, 0, "k^")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].set_title("End points (ground hit or apogee)")
    ax[0].set_xlabel("East [m]")
    ax[0].set_ylabel("North [m]")
    ax[0].grid(alpha=0.3)
    ax[1].hist(apo, bins=20)
    ax[1].set_title(f"Apogee: mean {apo.mean():.0f} m, sd {apo.std():.0f} m")
    ax[1].set_xlabel("m AGL")
    ax[2].hist([r["max_body_roll_dps"] for r in results], bins=20, alpha=0.7, label="body")
    ax[2].hist([r["max_fincan_roll_dps"] for r in results], bins=20, alpha=0.7, label="fin can")
    ax[2].set_title("Max roll rate during ascent [deg/s]")
    ax[2].legend()
    f = out_dir / "montecarlo.png"
    fig.savefig(f, dpi=100)
    plt.close(fig)
    return f
