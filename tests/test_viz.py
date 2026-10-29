"""Smoke tests for the dashboard figure builders and the Streamlit app."""

import numpy as np
import pytest

plotly = pytest.importorskip("plotly")

from rocket_dt import viz  # noqa: E402
from rocket_dt.builder import build_simulation  # noqa: E402
from rocket_dt.config import load_config  # noqa: E402


@pytest.fixture(scope="module")
def flight():
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/fincan_disturbance_demo.yaml")
    cfg.simulation.t_max = 3.0
    sim = build_simulation(cfg)
    return sim, sim.run()


def test_figures_build(flight):
    sim, r = flight
    g = viz.rocket_geometry(sim.rocket)
    d = r.data
    launch = viz.fig_launch(d, g, r.events, n_frames=20)
    att = viz.fig_attitude(d, g, n_frames=20)
    assert len(launch.frames) > 5 and len(att.frames) > 5
    # frames only carry coordinates; mesh connectivity stays in the base traces
    assert att.frames[0].data[0].i is None and att.data[0].i is not None
    viz.fig_panel(d, r.events, "alt", [("z", d["z"]), ("v", d["speed"])], cursor=1.0)
    viz.fig_ground_track(d)
    viz.fig_altitude_downrange(d)
    viz.fig_side_view(g, {"CG": 1.2}, 1.5)
    s = viz.static_curves(sim.rocket)
    assert np.all(np.diff(s["cp"]) > -1e-6)  # CP moves aft with Mach (subsonic fins)


def test_fincan_rotates_fins_in_drawing(flight):
    sim, _ = flight
    g = viz.rocket_geometry(sim.rocket)
    a = viz._fin_meshes(g, 0.0)["fincan"][0]
    b = viz._fin_meshes(g, 0.5)["fincan"][0]
    assert not np.allclose(a, b)
    assert np.allclose(viz._fin_meshes(g, 0.0)["canards"][0], viz._fin_meshes(g, 0.5)["canards"][0])


@pytest.mark.slow
def test_dashboard_runs_headless(root):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(root / "dashboard/app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    assert any("kpi-grid" in m.value and "Apogee" in m.value for m in at.markdown)
