# rocket_dt

6-DOF flight simulation of the canard-controlled rocket: L1926 motor, a free-spinning fin can on a bearing,
and four individually servoed canards. It models the airframe aerodynamics and mass properties, the canard
servos and the flight computer's sensors, and has a slot for a controller.

* Airframe: OpenRocket file `data/rocket.ork.zip`
* Motor: openMotor and RASP files in `data/`
* Everything else (launch site, wind, sensors, servos, controller, Monte Carlo spreads): YAML files in `configs/`

Model, validation, placeholders and findings are in [DETAILS.md](DETAILS.md).

## Setup

Python 3.11 or newer and git.

```bash
git clone https://github.com/geck007/mini_sram_dt.git
cd mini_sram_dt
python3 -m venv venv
source venv/bin/activate              # Windows (PowerShell): venv\Scripts\Activate.ps1
pip install -e ".[test,dashboard]"
```

Check the install:

```bash
pytest -m "not slow"                  # under a minute
streamlit run dashboard/app.py        # dashboard on http://localhost:8501
```

The venv has to be activated in each new terminal. The package is installed in editable mode, so code
changes take effect without reinstalling.

## Commands

| Command | What it does |
|---|---|
| `python scripts/run_sim.py configs/baseline.yaml` | One flight, no controller → `output/baseline/` (flight.csv, sensor CSVs, plots) |
| `python scripts/run_sim.py configs/vertical_hold.yaml` | Same rocket with the example controller (5° rail, 4 m/s wind) |
| `python scripts/run_sim.py configs/baseline.yaml --apogee --canard-schedule "0:0,3:0,3.01:0.5,6:0.5,6.01:0"` | Open-loop canard schedule (replaces the controller) |
| `python scripts/montecarlo.py configs/vertical_hold.yaml --runs 200 --apogee` | Monte Carlo with the config's dispersions |
| `python scripts/validate_openrocket.py` | Comparison with the OpenRocket export `data/all.csv` |
| `pytest` | All tests (a few minutes) |
| `pytest -m "not slow"` | Everything except the full flights |

## Layout

```
configs/                baseline.yaml (no control), vertical_hold.yaml (example controller),
                        fincan_disturbance_demo.yaml (build errors), validation_openrocket.yaml
data/                   rocket.ork.zip, motor files, OpenRocket exports, CAD
dashboard/app.py        Streamlit dashboard
scripts/                command-line tools (run_sim, montecarlo, validate_openrocket)
rocket_dt/
  sim.py                6-DOF equations, integration loop, logging
  aero.py               aerodynamics (lift, drag, canard vortices, hinge moments)
  mass.py, motor.py     mass properties, burning motor
  atmosphere.py, wind.py
  sensors.py            IMU, barometer and GNSS models
  actuators.py          canard servo model
  control.py            controller interface
  controllers/          one module per controller, found automatically; vertical_hold.py is the
                        example, estimation.py a strapdown estimator (helper, not a controller)
  config.py             YAML schema
  builder.py            combines the .ork, motor files and YAML
  io/, analysis/, viz.py
tests/
.github/workflows/      tests on every push to main and every pull request
```

## Branches and files

Each controller lives in its own module and config, e.g. for `alex_pid`:
* `rocket_dt/controllers/alex_pid.py`
* `configs/alex_pid.yaml` (and `configs/alex_pid_*.yaml` for more scenarios)

Controller work happens on a branch per person and is merged into `main` via pull request. The model itself
is `rocket_dt/*.py` and `configs/baseline.yaml`.

```bash
git checkout main && git pull
git checkout -b alex_pid                # once
git add rocket_dt/controllers/alex_pid.py configs/alex_pid*.yaml
git commit -m "alex_pid: add roll damping"
git push -u origin alex_pid

git checkout main && git pull           # pick up model changes
git checkout alex_pid && git merge main
```

CI runs the tests on every pull request. One test flies every controller in `rocket_dt/controllers/` for
the first few seconds, so a controller that raises or returns the wrong shape fails the build. `venv/` and
`output/` are ignored by git.

## Dashboard

`streamlit run dashboard/app.py`

* **Sidebar:** pick a base config, edit launch, wind, fin can, canard vortices, disturbances, sensors,
  servos, controller and integrator settings, then *Run*. Edits are applied as YAML overrides with the same
  checks as the command-line tools. Runs are kept in a history.
* **Launch:** animated 3D flight, attitude close-up, event table.
* **Telemetry:** time histories in groups (controller, trajectory, aerodynamics, attitude, roll & fin can,
  forces, mass, environment, sensors & servos), with a time window, cursor and full state at the cursor.
  The *Controller* group shows the controller's `telemetry()` and its estimates against the truth.
* **Explorer:** any logged variable against any other, overlaid across runs.
* **Trajectory**, **Vehicle & model**, **Monte Carlo**, **OpenRocket** comparison.

Each graph has a fullscreen button at its top right; drag to zoom, double-click to reset. More in
[DETAILS.md](DETAILS.md#dashboard).

## Writing a controller

Each simulation step:
1. The sensors sample the true state.
2. Every `1/rate_hz` the controller receives the new samples and returns one deflection per canard.
3. The servo model moves the canards toward the command under the aerodynamic load.

The controller sees what the flight computer would see: sensor data after errors and delays.

The example is `rocket_dt/controllers/vertical_hold.py` (with `estimation.py`), run with
`configs/vertical_hold.yaml`. It is described in [DETAILS.md](DETAILS.md#example-controller-vertical_hold).

### Controller module

A controller module has a `Params` dataclass, a `Controller` subclass with `reset()` and `update()`, and a
`CONTROLLER` line naming that class:

```python
from dataclasses import dataclass
import numpy as np
from rocket_dt.control import Controller, ControllerInput

@dataclass
class Params:
    gain: float = 1.0                        # settable from the YAML and the dashboard

class MyController(Controller):
    def __init__(self, rocket, cfg, params: Params):
        self.p = params
        self.n = rocket.aero.n_canards       # 4
        self.reset()

    def reset(self):                         # called before every run
        self.roll_rate = 0.0

    def update(self, inp: ControllerInput) -> np.ndarray:
        for s in inp.sensors.imu.get("imu0", []):   # new samples since the last call, oldest first
            self.roll_rate = s.gyro[0]               # rad/s
        return np.full(self.n, -0.001 * self.p.gain * self.roll_rate)   # rad, one per canard

    def telemetry(self) -> dict:             # optional: logged as ctrl_<key>, shown in the dashboard
        return {"roll_rate_est": self.roll_rate}


CONTROLLER = MyController                    # controller type "my_controller" (the file name)
```

Any module in `rocket_dt/controllers/` with a `CONTROLLER` line is registered under its file name. A config
selects it:

```yaml
extends: baseline.yaml
controller:
  type: my_controller
  rate_hz: 100
  params: {gain: 2.0}
```

### Controller inputs

| | |
|---|---|
| `inp.t` | time, s. Negative on the pad before ignition (`simulation.pad_time_s`, 5 s in the baseline); ignition at 0 |
| `inp.sensors.imu[name]` | list of `ImuSample(t, accel, gyro)`, sensor frame. Accel is specific force in m/s² (+g along +x on the pad); gyro in rad/s |
| `inp.sensors.baro` | list of `BaroSample(t, pressure [Pa], temperature [K])`; `rocket_dt.sensors.pressure_altitude()` converts to altitude |
| `inp.sensors.gnss` | list of `GnssSample(t, valid, position, velocity)`, ENU metres from the pad; NaN without a fix (no fix above 4 g) |
| `inp.truth` | true state, for debugging; not available to real flight software |
| `cfg.sensors` | mounting, rate and noise of each sensor; `rocket.aero.fins` gives the canard positions |

Sample lists can be empty: disabled sensors produce nothing, and each sensor runs at its own rate.
`telemetry()` has to return the same keys on every call.

### Conventions

* **Body frame:** +x out of the nose, origin at the nose tip; station s is at x = −s.
* **Ground frame:** ENU (x east, y north, z up) from the pad.
* **Attitude:** quaternion, body → ENU (`rocket_dt/quaternion.py`).
* **Canard numbering:** canard k at roll angle φ_k = k·90° from body +y; radial direction
  e_r = (0, cos φ_k, sin φ_k).
* **Canard deflection:** positive gives positive roll about +x and a pitching moment about −e_r. Canards k and
  k+2 deflected equal and opposite pitch the rocket without rolling it (mixing in `vertical_hold.py`).
* **Units:** commands in radians, limited to ±`actuators.canards.limit_deg`.
* **Scaling:** canard force and aerodynamic stiffness both scale with dynamic pressure q, so a gain in
  degrees of deflection per degree of tilt holds across speed, while rate damping scales with 1/speed.
  One canard: ≈ 0.0080·q N m pitch moment per rad (0.0063–0.0097 over `aero.canard_body_carryover`).
  Restoring moment ≈ 0.048·q N m per rad of angle of attack. All four canards: ≈ 0.29 N m roll per degree
  at 100 m/s.

## Testing guidelines

Settings and tools available for testing a controller:

* **Servos:** `actuators.canards.model: ideal` or `servo`. The servo's deadband (0.1°), resolution (0.09°)
  and backlash (0.2°) are on the order of the deflections roll control needs.
* **State knowledge:** `inp.truth` or the sensors.
* **Launch conditions:** rail tilt (`launch.angle_from_vertical_deg`), wind and turbulence.
* **Build errors:** all under `disturbances` in the config and in the dashboard sidebar. Each is zero (a
  perfect rocket) by default:
  * thrust misalignment and offset
  * fin and canard misalignment
  * CG offset in mm
  * nose cone misalignment
  * airframe principal-axis tilt
  * fin can imbalance (CG offset and axis tilt)

  `configs/fincan_disturbance_demo.yaml` has errors that make the body roll. Suggested progression:
  1. Get the controller working on the perfect rocket.
  2. Add one error at a time and check the tilt charts. A controller without integral action settles at a
     constant tilt.
  3. Run the Monte Carlo, which disperses all of the errors at once.
* **Monte Carlo:** `scripts/montecarlo.py` or the dashboard tab; reports the spread and worst case of
  `max_tilt_deg`, apogee and roll.
* **Estimate vs truth:** *Telemetry → Controller* in the dashboard.
* **Tilt:** *Tilt of axis from vertical* (`zenith_angle`, `tilt_east`, `tilt_north`) is the angle between the
  rocket's axis and the local vertical. The angle of attack is measured from the relative wind and goes to
  zero once the rocket weathercocks, regardless of tilt.
