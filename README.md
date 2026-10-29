# rocket_dt — solid-rocket digital twin

6-DOF flight simulation of the rocket in `data/rocket.ork.zip`, with a free-spinning
fin can on a bearing and roll-control canards. The airframe is read from the
OpenRocket file, the motor from the openMotor/RASP files, and everything else from a
YAML config.

```bash
source venv/bin/activate
pip install -e .[test,dashboard]

streamlit run dashboard/app.py                                  # interactive dashboard (http://localhost:8501)

python scripts/run_sim.py configs/baseline.yaml                 # one flight -> output/baseline/
python scripts/run_sim.py configs/baseline.yaml --locked        # fin can locked to the body
python scripts/run_sim.py configs/baseline.yaml --apogee --canard-schedule "0:0,3:0,3.01:0.5,6:0.5,6.01:0"
python scripts/montecarlo.py configs/baseline.yaml --runs 200   # dispersions from the config
python scripts/validate_openrocket.py                           # comparison with data/all.csv
pytest                                                          # 42 tests (incl. OpenRocket validation, dashboard)
```

## Dashboard

`dashboard/app.py` (Streamlit); figure builders are in `rocket_dt/viz.py`.

* **Sidebar:** pick a base config and edit launch, wind/turbulence, fin can and bearing, disturbances,
  per-fin misalignment, an open-loop canard schedule and integrator settings, then *Run*.
  Edited values become overrides of the YAML. They're listed under the run title and in the
  effective-config download, and they go through the same strict validation as the CLI.
  Results are cached per scenario, and every run is kept in a run history (*Show run*).
* **🎬 Launch:**
  * an animated 3D flight with an enlarged rocket model following the trajectory;
  * an animated close-up at the CG showing attitude, velocity relative to the air, and the fin can
    turning relative to the canards;
  * an event table.
* **📈 Telemetry:** grouped time histories (trajectory, aerodynamics, attitude & rates, roll &
  fin can, forces & moments, mass properties, environment), with event markers, a time window,
  a cursor, and the full state at the cursor.
* **🔍 Explorer:** any logged variable against any other, with overlays from previous runs and a CSV download.
* **🗺️ Trajectory:** ground track and altitude vs downrange.
* **🧩 Vehicle & model:**
  * side view with CG and CP;
  * CP, CN_α and drag breakdown vs Mach;
  * the build log (sources, overrides, motor checks);
  * the effective config as YAML.
* **🎲 Monte Carlo:** runs the config's dispersions in parallel, then shows end-point scatter,
  histograms, input→output sensitivity plots and a CSV download.
* **✅ OpenRocket:** overlays and coefficient comparison against a full OpenRocket CSV export in `data/`.

## Single source of truth

| Quantity | Source | Notes |
|---|---|---|
| Airframe geometry, materials, mass components, fin sets, motor-mount position | `.ork` | Changes go in the .ork, or in `rocket.overrides` (each needs a `note` and is logged) |
| Thrust(t), per-grain propellant mass(t) | openMotor CSV | |
| Grain geometry and order | openMotor `.ric` | Cross-checked against the CSV grain masses |
| Loaded motor mass, case diameter/length | `.eng` header | `.eng` thrust and propellant mass are only cross-checked |
| Launch site, wind, atmosphere, disturbances, integrator | YAML config | Simulation settings stored in the .ork are ignored |

The config is strict: unknown keys, missing required keys, unknown override fields,
ambiguous component paths and disagreeing motor files all raise errors. Child configs
use `extends: baseline.yaml` and list only what differs (no copy-pasted parameters).
`Transition#1` / `Transition#2` address siblings that share a name.

## Model

* **Frames:** inertial ENU with origin at the pad. Body frame: origin at the nose tip, +x forward;
  a point at OpenRocket station X has body x = −X. Attitude is a unit quaternion
  (body → ENU), so there is no gimbal lock. Euler-like angles are only derived for output.
* **State:** CG position and velocity, quaternion, body rates, and the fin can's roll angle and rate
  relative to the body. Fixed-step RK4 (default 2 ms). Canard commands and turbulence
  are held constant within a step.
* **Fin can:** an axisymmetric rotor on a bearing. Its spin angular momentum enters the
  rigid-body equations (gyroscopic coupling), and only bearing friction couples its roll to the body
  (Coulomb + viscous, smoothed with tanh). `fin_can.enabled: false` locks it.
* **Aerodynamics** (`aero.py`): the extended Barrowman method as implemented in OpenRocket (Niskanen 2009),
  evaluated with the *local* flow at every lifting element, so pitch and roll damping come out
  of the lift model:
  * nose/transitions: slender-body CNα (Jorgensen form for large α);
  * body crossflow lift (K = 1.1), integrated over 2.5 cm slices;
  * fins and canards: per-fin Diederich lift slope with K_T body interference, in spanwise strips (roll damping,
    per-fin deflection for roll now and pitch/yaw control later), plus a flat-plate term at high α;
  * drag buildup: skin friction with roughness limit, forward-facing/boattail pressure drag,
    fin leading/trailing-edge drag by cross-section, base drag, and OpenRocket's axial-force-vs-α multiplier;
  * canard vortices (`aero.canard_vortex`): each canard sheds one tip vortex whose strength comes from
    *that canard's* normal force. The vortex and its image inside the body are carried to the fins aft of the
    canards, with the body rolling under the vortex and the crossflow drifting it on the way. The velocity
    they induce (Lamb–Oseen core) changes the angle of attack of every fin strip. The coupling is therefore
    resolved per canard–fin pair (ready for independent canard actuation), depends on the fin-can angle,
    and acts on pitch and yaw as well as roll. `static_cp` includes the resulting downwash.
  
  The model is valid subsonic only (to roughly Mach 0.8).
* **Mass** (`mass.py`): every component is a mass element with a full inertia tensor
  (shells of revolution with wall thickness measured normal to the surface, fins as plates, mass components as cylinders).
  * The motor is the case hardware (a thick cylinder) plus four burning grains. Each grain's
    core grows so that its mass follows openMotor.
  * Jet damping and the dI/dt term are included.
* **Environment:** ISA (temperature offset optional), latitude-dependent gravity, constant or
  power-law wind, optional Dryden-type turbulence.
* **Disturbances:** thrust misalignment and offset, thrust and drag scale, lateral CG offset,
  per-fin misalignment for each fin set.
* **Control hooks** (`control.py`): `Controller.update(snapshot)` runs at `rate_hz`. `IdealActuator`
  clips to ±15°. Positive canard deflection gives positive roll about +x.

## Validation (configs/validation_openrocket.yaml vs data/all.csv)

`all.csv` came from an older revision of the design (`yay it works.rkt`). The validation config
rebuilds that revision from the current .ork with five logged overrides. The fin can is locked
because OpenRocket can't model it.

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
* **Damping:** OpenRocket uses empirical damping terms, so its AoA oscillation is less damped.
* **Descent:** OpenRocket clips fin lift at 20° and falls tail-first; this model falls broadside.

## Placeholders and assumptions to replace with measured data

| Item | Current value | Replace with |
|---|---|---|
| Fin-can roll inertia | 0.026 kg m² (upper-bound estimate) | CAD: select the Fin Can component → Properties |
| Bearing friction (61820) | 0.02 N m Coulomb, 1e-4 N m s viscous | spin-down test of the built fin can |
| Grain position in case | stack centred, no gaps | real motor drawing |
| Case hardware CG | case centre | real motor drawing |
| Canard actuator | ideal, ±15° | servo model (bandwidth, rate limit, backlash, resolution) |
| Canard vortex strength / core radius | factor 1.0 / 6 mm (dispersed 0.5–1.5 in Monte Carlo) | CFD, or fin-can spin rate under a known canard offset in flight |

## Findings so far (current .ork)

* **Apogee ≈ 1794 m** (tail fins `airfoil`, zero-lift CD ≈ 0.41 at Mach 0.3). The gap to the older revision's
  1926 m is mostly ≈ 1.1 kg of extra mass (liftoff 18.26 kg vs 17.15 kg). With the tail fins modelled as `square`
  (10 mm blunt edges), as an earlier .ork had them, CD would be ≈ 0.76 and apogee ≈ 1513 m.
* **Static margin ≈ 1.8–2.2 cal** during ascent. Canard downwash on the tail costs ≈ 0.14 cal
  (tail normal force −5 %). The validation config switches the vortices off, as OpenRocket ignores them.
* **Roll authority is very high with a free fin can.** The canards are the body's only roll damping,
  so the steady body roll rate is ≈ K_T·δ·V / r_canard: 0.1° of canard misalignment gives ≈ 300 °/s,
  and 5° gives ≈ 11,000 °/s. Roll control will work with deflections of hundredths to tenths of a
  degree, so servo resolution, backlash and canard zero-offset matter a lot.
* **Canard vortices reverse roll on the tail.** The torque they induce on the fin can is −0.65 × the canard
  roll moment when the fins are in line with the canards, and −1.0 × when the vortices pass midway between fins
  (Mach 0.3; plot in the dashboard's *Vehicle & model* tab). A free fin can absorbs this, so the
  body roll response is unchanged. The fin can is kicked the opposite way during canard transients (≈ −66 °/s
  for a 0.5° step). In steady roll, each canard's net lift is close to zero, and so is the coupling.
  * The in-line value depends on the core radius (−0.49 at 3 mm, −0.69 at 12 mm); the midway value does not.
  * Limitation: one vortex per canard, carrying only the net lift. When a canard's span loading changes sign
    (deflection against roll damping), the counter-rotating vorticity it really sheds is not represented.
    A vortex-sheet (multi-filament) model would fix that.

## Next steps

1. Servo actuator model and canard hinge moments.
2. Sensor models (IMU, baro, GNSS, magnetometer: bias, noise, scale factor, misalignment, latency,
   quantisation) between truth and the controller.
3. Roll controller and estimator; later, run the flight software inside the sim.
4. Recovery events (drogue/main) for full descent and landing dispersion.
