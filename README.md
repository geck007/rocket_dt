# rocket_dt — digital twin of the canard-controlled rocket

A 6-DOF flight simulation of our rocket: an L1926 solid motor, a fin can that spins freely on a
bearing, and four canards that are each driven by their own servo. It covers the airframe
aerodynamics and mass properties, the canard servos, the flight computer's sensors, and a controller
slot where the flight software plugs in.

* The airframe comes from the OpenRocket file `data/rocket.ork.zip`.
* The motor comes from the openMotor and RASP files in `data/`.
* Everything else (launch site, wind, sensors, servos, controller, Monte Carlo spreads) comes from a
  YAML file in `configs/`.

## Getting started

```bash
cd junior_larp
python3 -m venv venv && source venv/bin/activate   # once (Python ≥ 3.11)
pip install -e .[test,dashboard]                   # once

streamlit run dashboard/app.py                     # dashboard on http://localhost:8501
```

| Command | What it does |
|---|---|
| `python scripts/run_sim.py configs/baseline.yaml` | One flight, no controller → `output/baseline/` (flight.csv, sensor CSVs, plots) |
| `python scripts/run_sim.py configs/vertical_hold.yaml` | The same rocket flown by the example controller (5° rail, 4 m/s wind) |
| `python scripts/run_sim.py configs/baseline.yaml --apogee --canard-schedule "0:0,3:0,3.01:0.5,6:0.5,6.01:0"` | Open-loop canard roll test (replaces the controller) |
| `python scripts/montecarlo.py configs/vertical_hold.yaml --runs 200 --apogee` | Monte Carlo with the dispersions in the config |
| `python scripts/validate_openrocket.py` | Comparison with the OpenRocket export `data/all.csv` |
| `pytest` | All tests (about 1.5 minutes) |
| `pytest -m "not slow"` | Everything except the full flights (about 5 s) |

## Layout

```
configs/                baseline.yaml (the rocket, no control), vertical_hold.yaml (example controller),
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
  control.py            controller interface (what a controller receives and returns)
  controllers/          controllers you can pick in a config; vertical_hold.py is the example,
                        estimation.py a simple strapdown estimator
  config.py             strict YAML schema
  builder.py            puts the .ork, motor files and YAML together
  io/, analysis/, viz.py
tests/
```

## The rocket today

| | |
|---|---|
| Length / diameter | 1.975 m / 130 mm |
| Liftoff mass | 18.26 kg (dry airframe 9.86 kg, motor hardware 6.60 kg, propellant 1.80 kg) |
| Motor | L1926: 4103 Ns, 2.1 s burn |
| Canards | 4 × airfoil, 64 mm span, CP at station 0.80 m; hinge line at the planform centroid; one servo each |
| Tail fins | 4 × airfoil, 160 mm span, on the free fin can (1.83 kg, roll inertia 0.026 kg m²) |
| Static margin | 1.84 cal at liftoff, 1.8–2.2 cal during the ascent |
| Avionics | 2 IMUs, a barometer; GNSS configured but switched off |

Baseline flight (vertical rail, 2 m/s wind, no controller): apogee **1793 m** at 19.2 s, max speed 209 m/s
(Mach 0.62), max acceleration 11.7 g, rail exit at 14.2 m/s. Without control the rocket turns into the wind
and is 45° off vertical by apogee (39° with ideal servos: with zero command, a canard resting against its
backlash is enough to steer it). With the example controller it stays within about 7° on a 5° rail in 4 m/s wind.

## Dashboard

`streamlit run dashboard/app.py`. The figure builders live in `rocket_dt/viz.py`.

* **Sidebar:** pick a base config, then edit any of these and press *Run*:
  * launch, wind/turbulence, fin can and bearing, canard vortices, disturbances
  * sensors on/off, servo parameters
  * the controller and its parameters, or an open-loop canard schedule
  * integrator settings, including the time on the pad

  Every edit becomes a listed override of the YAML and goes through the same strict checks as the
  command-line tools. Each run is kept in a run history (*Show run*).
* **Every graph is its own chart.** Hover a graph and use the fullscreen button at its top right to
  enlarge it. Drag to zoom; double-click to reset.
* **Launch:** animated 3D flight; close-up of the attitude, relative wind and the fin can turning; event table.
* **Telemetry:** groups of time histories:
  * controller, trajectory, aerodynamics, attitude, roll & fin can
  * forces, mass, environment, sensors & servos

  Pick the graphs to show in each group, choose how many graphs go in a row, set a time window and a cursor,
  and see the full state at the cursor. The *Controller* group plots everything the controller reports
  (`telemetry()`) and its estimates against the truth. Full-rate sensor logs download as CSV.
  *Tilt of axis from vertical* (Trajectory and Attitude groups) is the angle between the rocket's axis and
  the local vertical, split into its East and North leans (`zenith_angle`, `tilt_east`, `tilt_north`).
  Use it, not the angle of attack, to judge pitch/yaw control: the AoA is measured from the relative
  wind and goes to zero once the rocket weathercocks, however far it leans.
* **Explorer:** any logged variable against any other, one graph per variable or overlaid, with runs
  from the history overlaid; CSV download.
* **Trajectory:** ground track, altitude vs downrange.
* **Vehicle & model:**
  * side view with CG and CP
  * CP, CN_α and drag vs Mach
  * canard vortex coupling
  * build log and effective config
* **Monte Carlo:** runs the config's dispersions; end points, apogee and roll histograms, max tilt,
  input→output sensitivity plots, CSV.
* **OpenRocket:** overlays and coefficient comparison with an OpenRocket CSV export in `data/`.

## Writing a controller

The controller is the flight software's guidance and control loop. Each simulation step goes like this:
1. The sensors sample the true state.
2. Every `1/rate_hz` the controller gets the new samples and returns one deflection per canard.
3. The servo model moves the canards toward that command, under the aerodynamic load.

Your controller sees exactly what the flight computer would see: the sensors, after their errors and delays.

### 1. Run the example first

```bash
python scripts/run_sim.py configs/vertical_hold.yaml --apogee
streamlit run dashboard/app.py          # pick vertical_hold.yaml, open Telemetry → Controller
```

`rocket_dt/controllers/vertical_hold.py` keeps the rocket's axis vertical and damps roll. Read it next
to `rocket_dt/controllers/estimation.py`; together they are about 200 lines.

### 2. Make your own

Copy `vertical_hold.py` to `rocket_dt/controllers/my_controller.py`. A controller is a `Params` dataclass
(its tunable numbers) plus a class with `reset()` and `update()`:

```python
from dataclasses import dataclass
import numpy as np
from rocket_dt.control import Controller, ControllerInput

@dataclass
class Params:
    gain: float = 1.0                        # every field can be set from the YAML and the dashboard

class MyController(Controller):
    def __init__(self, rocket, cfg, params: Params):
        self.p = params
        self.n = rocket.aero.n_canards       # 4
        self.reset()

    def reset(self):                         # called before every run
        self.roll_rate = 0.0

    def update(self, inp: ControllerInput) -> np.ndarray:
        for s in inp.sensors.imu.get("imu0", []):   # all new samples since the last call, oldest first
            self.roll_rate = s.gyro[0]               # rad/s
        return np.full(self.n, -0.001 * self.p.gain * self.roll_rate)   # rad, one per canard

    def telemetry(self) -> dict:             # optional: logged as ctrl_<key>, shown in the dashboard
        return {"roll_rate_est": self.roll_rate}
```

Register it in `rocket_dt/controllers/__init__.py`. Import your module next to `vertical_hold`, and add a
line to `CONTROLLERS`:

```python
from rocket_dt.controllers import my_controller, vertical_hold

CONTROLLERS = {
    "none": (None, lambda rocket, cfg, params: NoControl(rocket.aero.n_canards)),
    "vertical_hold": (vertical_hold.Params, vertical_hold.VerticalHold),
    "my_controller": (my_controller.Params, my_controller.MyController),
}
```

Then select it in a config that extends the baseline:

```yaml
extends: baseline.yaml
controller:
  type: my_controller
  rate_hz: 100
  params: {gain: 2.0}
```

### 3. What the controller gets

| | |
|---|---|
| `inp.t` | time, s. Negative on the pad before ignition (`simulation.pad_time_s`, 5 s in the baseline); ignition is at 0 |
| `inp.sensors.imu[name]` | list of `ImuSample(t, accel, gyro)`, in the sensor frame. Accel is specific force in m/s² (on the pad it reads +g along +x); gyro is rad/s |
| `inp.sensors.baro` | list of `BaroSample(t, pressure [Pa], temperature [K])`. `rocket_dt.sensors.pressure_altitude()` converts pressure to altitude |
| `inp.sensors.gnss` | list of `GnssSample(t, valid, position, velocity)`, ENU metres from the pad. NaN while there is no fix (no fix above 4 g, so none during the boost) |
| `inp.truth` | the true state. Use it only for debugging, or to develop control with perfect knowledge before writing the estimator. Flight software can't have it |
| `cfg.sensors` | where each sensor is mounted, its rate and its noise. `rocket.aero.fins` gives the canard positions |

### 4. Conventions (get these wrong and the rocket flips)

* **Body frame:** +x out of the nose, origin at the nose tip; station s (from the nose) is at x = −s.
* **Ground frame:** ENU (x east, y north, z up) from the pad.
* **Attitude:** quaternion, body → ENU (`rocket_dt/quaternion.py`).
* **Canard numbering:** canard k sits at roll angle φ_k = k·90° from body +y. Its radial direction is
  e_r = (0, cos φ_k, sin φ_k).
* **Canard deflection:** positive gives positive roll about +x. It also gives a pitching moment about
  −e_r. So canards k and k+2 deflected by equal and opposite amounts pitch the rocket and don't roll it.
  `vertical_hold.py` shows the mixing.
* **Units:** commands are in radians and limited to ±`actuators.canards.limit_deg`.
* **Gain scheduling:** canard force and the rocket's own aerodynamic stiffness both scale with dynamic
  pressure q. Gains expressed as "degrees of deflection per degree of tilt" therefore work at any speed;
  rate damping should scale with 1/speed. One canard gives about 0.0080·q N m of pitch moment per rad
  (0.0063–0.0097 over the range of `aero.canard_body_carryover`). The restoring moment is about
  0.048·q N m per rad of angle of attack. All four canards give ≈ 0.29 N m of roll per degree at 100 m/s.
* **Empty sample lists:** a disabled sensor produces no samples, and samples arrive at each sensor's own
  rate. Expect empty lists and missing names.
* **Telemetry keys:** `telemetry()` must return the same keys on every call.

### 5. Test it properly

* **Start simple, then make it real.**
  * Begin with an ideal world: `actuators.canards.model: ideal`, or `inp.truth` instead of the estimator.
  * Then use the real sensors and the servo model. Deadband (0.1°), resolution (0.09°) and backlash (0.2°)
    are as large as the deflections roll control needs.
* **Try the hard cases.**
  * Tilt the rail (`launch.angle_from_vertical_deg`) and add wind.
  * Use `fincan_disturbance_demo.yaml`, which has build errors that make the body roll.
  * Run a Monte Carlo and look at the worst case of `max_tilt_deg`, not the mean.
* **Look at the estimate against the truth.** *Telemetry → Controller* plots it. Most controller problems
  are really estimator problems.

### The example: `vertical_hold`

**How it works.** The estimator (`controllers/estimation.py`):
* averages the IMU on the pad, giving the gyro bias and the direction of gravity (the initial tilt)
* detects launch when the IMU reads more than 3 g for five samples in a row
* then integrates the gyro for attitude and the accelerometer for velocity, and takes air density from the barometer

The controller:
* works out the tilt error as e = b × up, where b is the estimated body axis
* commands each canard pair with a PID on that error: kp·e + ki·∫e in the ground frame, minus (kd/V)·(pitch and yaw rate)
* damps roll with −k_roll·p/V
* is active only once the estimated speed exceeds 15 m/s (off the rail) and until apogee

**Results** (`vertical_hold.yaml`: 5° rail, 4 m/s wind):

| | No controller | `vertical_hold` |
|---|---|---|
| Max tilt from vertical | 88° | 7° |
| Drift at apogee | 505 m | 95 m |
| Apogee | 1747 m | 1785 m |

The gains hold up across the canard-authority uncertainty: max tilt is 7.7° with no body carry-over and
6.7° with all of it.

The price is a higher angle of attack just off the rail (≈ 11°). Holding the rocket vertical at low speed
means not turning into the wind.

**Limits** (good first projects):

* **Estimator accuracy is about 2–4° of tilt**, so the controller holds its *estimate* vertical.
  * The pad levelling can't tell accelerometer bias (20 mg ≈ 1.2°) or misalignment (0.5°) from tilt.
    Using the measured rail angle instead would remove that.
  * Gyro g-sensitivity (0.1 °/s per g × 11 g during the boost) adds up to ≈ 4°.
  * There's no aiding after launch: no barometer altitude and vertical speed, no GNSS once it has a fix again.
* **Roll is only damped**, not held at an angle.
* **The target is a fixed direction.** Following a path f(x, y, z) = 0 needs position feedback (guidance) on top.

## Configuration

| Quantity | Source | Notes |
|---|---|---|
| Airframe geometry, materials, mass components, fin sets, motor-mount position | `.ork` | Changes go in the .ork, or in `rocket.overrides` (each needs a `note` and is logged) |
| Thrust(t), per-grain propellant mass(t) | openMotor CSV | |
| Grain geometry and order | openMotor `.ric` | Cross-checked against the CSV grain masses |
| Loaded motor mass, case diameter/length | `.eng` header | `.eng` thrust and propellant mass are only cross-checked |
| Launch site, wind, atmosphere, disturbances, integrator | YAML | Simulation settings stored in the .ork are ignored |
| Canard vortex model, canard body carry-over, body crossflow C_d | YAML `aero.canard_vortex`, `aero.canard_body_carryover`, `aero.body_crossflow_cd` | |
| Canard servos and hinge line | YAML `actuators.canards` | |
| Sensors | YAML `sensors` | Each one has `enabled`; GNSS is off in the baseline |
| Controller | YAML `controller` (`type`, `rate_hz`, `params`) | Types are registered in `rocket_dt/controllers/__init__.py` |
| Monte Carlo spreads | YAML `montecarlo.dispersions` | Every run also gets new turbulence and sensor-error seeds |

The config is strict: unknown keys, missing required keys, unknown override fields, unknown controller
parameters, ambiguous component paths and disagreeing motor files all raise errors. Child configs use
`extends: baseline.yaml` and list only what differs. `Transition#1` / `Transition#2` address siblings
that share a name.

## Model

* **Frames:** inertial ENU with its origin at the pad. Body frame: origin at the nose tip, +x forward. A point
  at OpenRocket station X has body x = −X. Attitude is a unit quaternion (body → ENU), so there is no gimbal
  lock.
* **State:** CG position and velocity, quaternion, body rates, and the fin can's roll angle and rate relative to
  the body. Fixed-step RK4 (2 ms). Canard deflections and turbulence are held constant within a step.
* **Fin can:** an axisymmetric rotor on a bearing. Its spin enters the rigid-body equations (gyroscopic
  coupling), and only bearing friction couples its roll to the body (Coulomb + viscous, smoothed with tanh).
  `fin_can.enabled: false` locks it.
* **Aerodynamics** (`aero.py`): the extended Barrowman method as implemented in OpenRocket (Niskanen 2009),
  evaluated with the local flow at every lifting element, so pitch and roll damping come out of the lift model.
  * Nose and transitions: slender-body CNα, in Jorgensen's form for large α.
  * Body crossflow lift η·C_dc·A_plan·sin²α (Jorgensen), over 2.5 cm slices: C_dc = 1.2 (subcritical
    cylinder, `aero.body_crossflow_cd`), η = 0.72 for L/D = 15. OpenRocket's K = 1.1 overstates the
    broadside force by ≈ 25 %.
  * Fins and canards: per-fin Diederich lift slope, in spanwise strips, plus a flat-plate crossflow term
    (C_dc 1.2) at high α. Both use the dynamic pressure of the flow in the fin's chordwise plane (spanwise
    flow does not load a fin), the interference factors below act on the attached-flow lift only, and the
    lift fades as cos α so a broadside fin carries crossflow only. Lift acts at the MAC quarter chord
    (three-quarter chord in reverse flow), crossflow at mid-chord.
  * Fin interference (`interference.py`):
    * Angle of attack of the body uses Barrowman's K_T = 1 + r/(s+r), as OpenRocket does.
    * Fin deflections (canard commands, cant, misalignment) and roll rate use slender-body factors,
      computed at build time for each fin set's r/s and fin count. For the canards these are: own
      load 0.90 (K_T would give 1.48), all four in roll 0.70, a pitch pair 0.94 plus 0.50 carried by the
      body, and roll damping 0.71.
    * The body carry-over of a deflected canard is scaled by `aero.canard_body_carryover`, since a hinge
      gap at the canard root loses part of it.
  * Drag buildup: skin friction with a roughness limit, pressure drag, fin edge drag by cross-section, base
    drag, and OpenRocket's axial-force-vs-α multiplier (zero at 90°). Tail-first, the build-up is redone for
    the reversed body: the base faces the flow, so CD is 0.94 against 0.39 nose-first.
  * Canard vortices (`aero.canard_vortex`): one tip vortex per canard, with its strength set by that canard's
    lift plus its body carry-over. It and its image in the body are carried to the tail fins (roll lag, crossflow drift, Lamb–Oseen core)
    and change every fin strip's angle of attack. The effect is resolved for each canard–fin pair, depends on the
    fin-can angle, and includes the downwash that `static_cp` reports.
  * Canard hinge moments: each strip's force acts at its local quarter chord, about the hinge line.
  * Valid subsonic only (to about Mach 0.8).
* **Mass** (`mass.py`): every component is a mass element with a full inertia tensor. The motor is the case
  hardware plus four burning grains whose masses follow openMotor. Jet damping and the dI/dt term are
  included.
* **Environment:**
  * ISA atmosphere (optional temperature offset), latitude-dependent gravity
  * constant or power-law wind, optional Dryden-type turbulence
* **Disturbances:**
  * thrust misalignment and offset, thrust and drag scale
  * lateral CG offset
  * misalignment of each fin; for the canard set this is the canard zero offset
* **Canard servos** (`actuators.py`): `ideal`, or `servo`. The servo model is a second-order position loop
  (natural frequency, damping, holding stiffness) with stall torque, a torque–speed rate limit, deadband,
  command quantisation, backlash and end stops, loaded by the hinge moment.
* **Sensors** (`sensors.py`): each has its own rate, latency, mounting position and `enabled` switch, and its
  own random stream from `sensors.seed`.
  * IMU: specific force at the sensor point (with lever-arm terms), turn-on bias, bias drift, noise, scale
    factor, misalignment, gyro g-sensitivity, saturation, quantisation.
  * Barometer: ISA pressure at the sensor, static-port error ∝ q, port lag, bias, noise, quantisation.
  * GNSS: ENU position and velocity with a slowly varying error, loss of lock above 4 g, re-acquisition after 1 s.
* **Pad phase:** for `simulation.pad_time_s` before ignition the rocket sits on the rail while the sensors,
  controller and servos run, so flight software can calibrate.

## Validation (`configs/validation_openrocket.yaml` vs `data/all.csv`)

`all.csv` came from an older revision of the design (`yay it works.rkt`). The validation config rebuilds that
revision from the current .ork with five logged overrides, locks the fin can (OpenRocket can't model it),
and switches the canard vortices off (OpenRocket ignores them).

| | rocket_dt | OpenRocket |
|---|---|---|
| Apogee | 1925.9 m | 1931.0 m (−0.26 %) |
| Time to apogee | 19.69 s | 19.74 s |
| Max speed | 223.5 m/s | 223.9 m/s |
| Liftoff mass / CG | 17.152 kg / 1.2400 m | 17.150 kg / 1.2406 m |
| CD at OpenRocket's (M, Re) | +0.002 mean diff | 0.42 |
| CP (small α) | +0.4 cm mean diff | |

Component masses match OpenRocket to better than 0.001 %. Known, intended differences:
* **Roll inertia (+16 %):** OpenRocket models the motor as a solid cylinder.
* **Damping:** OpenRocket uses empirical damping terms, so its angle-of-attack oscillation is less damped.
* **Descent:** OpenRocket clips fin lift at 20° and falls tail-first; this model turns nose-down ≈ 3 s after apogee.

## Placeholders to replace with measured data

| Item | Current value | Replace with |
|---|---|---|
| Fin-can roll inertia | 0.026 kg m² (upper-bound estimate) | CAD: select the fin can → Properties |
| Bearing friction (61820) | 0.02 N m Coulomb, 1e-4 N m s viscous | spin-down test of the built fin can |
| Grain position in case, case hardware CG | stack centred, no gaps; case centre | real motor drawing |
| Canard servo | 15 Hz, ζ 0.7, 40 N m/rad, 1.5 N m stall, 500 °/s, deadband 0.1°, resolution 0.09°, backlash 0.2° | bench tests: small-step response, torque vs deflection, play at the canard |
| Canard hinge line | planform centroid (54 mm aft of the root LE) | the real hinge position |
| IMUs | ICM-42688- and BMI088-class datasheet values, both at station 1.015 m | the chosen parts' datasheets and real mounting; bench calibration |
| Barometer | MS5611-class, static-port error 0 | datasheet; static-port coefficient from flight data |
| GNSS | M10-class, off | the chosen receiver's dynamics limits |
| Canard vortex strength / core radius | factor 1.0 / 6 mm (dispersed 0.5–1.5 in Monte Carlo) | CFD, or fin-can spin under a known canard offset in flight |
| Canard body carry-over | 0.5 of the slender-body value (dispersed 0–1 in Monte Carlo) | the hinge gap; wind tunnel or the pitch response to a canard step in flight |
| Body crossflow drag coefficient | 1.2 (subcritical cylinder) | only matters at high α; the descent is insensitive to it |
| `vertical_hold` gains | kp 5, ki 2.5, kd 75 m, k_roll 0.76 m | tuning against the measured servo and sensors |

## Findings so far

* **Static margin is 1.8–2.2 cal** during the ascent. Canard downwash on the tail costs ≈ 0.14 cal.
* **Roll authority is very high with a free fin can.** The canards are the body's only roll damping, so
  0.1° of canard misalignment gives ≈ 150 °/s of body roll, and 5° gives ≈ 9,000 °/s. Servo resolution,
  backlash and canard zero offset matter a lot.
* **The hinge moment pushes the canards away from neutral.** The canard's CP is 16 mm ahead of a centroid hinge.
  * With the placeholder servo, a 0.5° command becomes ≈ 0.72° in flight: quantisation to 0.54°, half the
    backlash and servo compliance.
  * Even with zero command, angle-of-attack loads hold the canards at ≈ ±0.17° inside the backlash.
  * A hinge about 38 mm aft of the root LE (at the CP) would remove most of the load.
* **Canard vortices reverse the roll on the tail.**
  * The torque they induce on the fin can is −0.87× the canard roll moment with the fins in line with the
    canards, and −1.35× with the vortices midway between fins.
  * The free fin can absorbs this, so it is kicked the opposite way during canard transients, while the
    body's roll response is unchanged.
  * Limitation: one vortex per canard, whose strength comes from an impulse balance (lift plus body
    carry-over). Slender-body theory puts the circulation shed at the root within ≈ 20 % of it.
* **Without recovery events the descent is ballistic.** The rocket hangs broadside for ≈ 2.5–3.5 s after
  apogee (≈ 12 m/s), then turns nose-down and hits the ground at ≈ 157 m/s ≈ 40 s after launch. Broadside the
  CP is only 0–0.3 cal aft of the CG (depending on the fin-can roll angle), so this phase is marginally stable;
  it barely changes over body crossflow C_dc 0.4–1.8. Not modelled: shielding of the leeward fins by the
  body's wake, and the drag crisis of the body (crossflow speed above ≈ 35 m/s).

## Next steps

1. Estimator: aid the attitude and velocity with the barometer (and GNSS when flown); level with the known rail angle.
2. Guidance: follow a path f(x, y, z) = 0 (a vertical line, or along a rail tilted up to ≈ 5°) instead of a fixed
   direction; add roll-angle hold.
3. Bench-test the servo and sensors and replace the placeholders above.
4. Recovery events (drogue/main) for the descent and landing dispersion.
5. Run the real flight software inside the simulation (software in the loop).
