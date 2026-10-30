# rocket_dt details

## Rocket

| | |
|---|---|
| Length / diameter | 1.975 m / 130 mm |
| Liftoff mass | 18.26 kg (dry airframe 9.86 kg, motor hardware 6.60 kg, propellant 1.80 kg) |
| Motor | L1926: 4103 Ns, 2.1 s burn |
| Canards | 4 × airfoil, 64 mm span, CP at station 0.80 m; hinge line at the planform centroid; one servo each |
| Tail fins | 4 × airfoil, 160 mm span, on the free fin can (1.83 kg, roll inertia 0.026 kg m²) |
| Static margin | 1.84 cal at liftoff, 1.8–2.2 cal during the ascent |
| Avionics | 2 IMUs, a barometer; GNSS configured but off |

Baseline flight (vertical rail, 2 m/s wind, no controller): apogee 1793 m at 19.2 s, max speed 209 m/s
(Mach 0.62), max acceleration 11.7 g, rail exit at 14.2 m/s. Uncontrolled, the rocket turns into the wind and
is 45° off vertical at apogee (39° with ideal servos; with zero command, a canard resting against its
backlash is enough to steer it). With `vertical_hold` it stays within about 7° on a 5° rail in 4 m/s wind.

## Dashboard

`streamlit run dashboard/app.py`. Figure builders are in `rocket_dt/viz.py`.

* **Sidebar:** base config, then overrides for:
  * launch, wind/turbulence, fin can and bearing, canard vortices, disturbances
  * sensors on/off, servo parameters
  * controller and its parameters, or an open-loop canard schedule
  * integrator settings, including time on the pad

  Each edit is a listed YAML override, checked like the command-line tools. Runs are kept in a history
  (*Show run*).
* **Graphs:** each is its own chart with a fullscreen button at its top right. Drag to zoom, double-click to reset.
* **Launch:** animated 3D flight; close-up of attitude, relative wind and fin-can rotation; event table.
* **Telemetry:** groups of time histories:
  * controller, trajectory, aerodynamics, attitude, roll & fin can
  * forces, mass, environment, sensors & servos

  Graph selection per group, graphs per row, time window, cursor and full state at the cursor. The
  *Controller* group plots the controller's `telemetry()` and its estimates against the truth. Full-rate
  sensor logs download as CSV. *Tilt of axis from vertical* (Trajectory and Attitude groups) is the angle
  between the rocket's axis and the local vertical, split into East and North (`zenith_angle`, `tilt_east`,
  `tilt_north`).
* **Explorer:** any logged variable against any other, separate or overlaid, across runs; CSV download.
* **Trajectory:** ground track, altitude vs downrange.
* **Vehicle & model:** side view with CG and CP; CP, CN_α and drag vs Mach; canard vortex coupling; build log
  and effective config.
* **Monte Carlo:** runs the config's dispersions; end points, apogee and roll histograms, max tilt,
  input→output sensitivity plots, CSV.
* **OpenRocket:** overlays and coefficient comparison with an OpenRocket CSV export in `data/`.

## Example controller: `vertical_hold`

`rocket_dt/controllers/vertical_hold.py` and `estimation.py`, about 200 lines together. Run with
`configs/vertical_hold.yaml`.

### Workings

Estimator (`controllers/estimation.py`):
* averages the IMU on the pad for gyro bias and the direction of gravity (initial tilt)
* detects launch at more than 3 g for five consecutive samples
* integrates the gyro for attitude and the accelerometer for velocity; air density from the barometer

Controller:
* tilt error e = b × up, with b the estimated body axis
* each canard pair: PID on that error, kp·e + ki·∫e in the ground frame, minus (kd/V)·(pitch and yaw rate)
* roll damping −k_roll·p/V
* active from 15 m/s estimated speed (off the rail) until apogee

### Results

`vertical_hold.yaml`: 5° rail, 4 m/s wind.

| | No controller | `vertical_hold` |
|---|---|---|
| Max tilt from vertical | 88° | 7° |
| Drift at apogee | 505 m | 95 m |
| Apogee | 1747 m | 1785 m |

Max tilt is 7.7° with no canard body carry-over and 6.7° with all of it. Angle of attack just off the rail
is higher (≈ 11°), since the rocket is held vertical instead of turning into the wind.

### Limitations

* Estimator accuracy is about 2–4° of tilt, so the controller holds its estimate vertical.
  * Pad levelling can't separate accelerometer bias (20 mg ≈ 1.2°) or misalignment (0.5°) from tilt.
  * Gyro g-sensitivity (0.1 °/s per g × 11 g during the boost) adds up to ≈ 4°.
  * No aiding after launch (barometer altitude and vertical speed, GNSS after re-acquisition).
* Roll is damped, not held at an angle.
* The target is a fixed direction; following a path f(x, y, z) = 0 needs position feedback on top.

## Configuration

| Quantity | Source | Notes |
|---|---|---|
| Airframe geometry, materials, mass components, fin sets, motor-mount position | `.ork` | Changes in the .ork, or in `rocket.overrides` (each needs a `note` and is logged) |
| Thrust(t), per-grain propellant mass(t) | openMotor CSV | |
| Grain geometry and order | openMotor `.ric` | Cross-checked against the CSV grain masses |
| Loaded motor mass, case diameter/length | `.eng` header | `.eng` thrust and propellant mass are only cross-checked |
| Launch site, wind, atmosphere, disturbances, integrator | YAML | Simulation settings in the .ork are ignored |
| Canard vortex model, canard body carry-over, body crossflow C_d | YAML `aero.canard_vortex`, `aero.canard_body_carryover`, `aero.body_crossflow_cd` | |
| Canard servos and hinge line | YAML `actuators.canards` | |
| Sensors | YAML `sensors` | Each has `enabled`; GNSS off in the baseline |
| Controller | YAML `controller` (`type`, `rate_hz`, `params`) | Types found automatically in `rocket_dt/controllers/` |
| Monte Carlo spreads | YAML `montecarlo.dispersions` | Each run also gets new turbulence and sensor-error seeds |

The config is strict: unknown keys, missing required keys, unknown override fields, unknown controller
parameters, ambiguous component paths and disagreeing motor files raise errors. Child configs use
`extends: baseline.yaml` and list only what differs. `Transition#1` / `Transition#2` address siblings
with the same name.

## Model

* **Frames:** inertial ENU with origin at the pad. Body frame: origin at the nose tip, +x forward; OpenRocket
  station X is body x = −X. Attitude is a unit quaternion (body → ENU).
* **State:** CG position and velocity, quaternion, body rates, fin-can roll angle and rate relative to the
  body. Fixed-step RK4 (2 ms); canard deflections and turbulence held constant within a step.
* **Fin can:** axisymmetric rotor on a bearing. Its spin enters the rigid-body equations (gyroscopic
  coupling); only bearing friction couples its roll to the body (Coulomb + viscous, smoothed with tanh).
  `fin_can.enabled: false` locks it.
* **Aerodynamics** (`aero.py`): extended Barrowman as implemented in OpenRocket (Niskanen 2009), evaluated
  with the local flow at every lifting element, so pitch and roll damping come from the lift model.
  * Nose and transitions: slender-body CNα, Jorgensen's form for large α.
  * Body crossflow lift η·C_dc·A_plan·sin²α (Jorgensen) over 2.5 cm slices: C_dc = 1.2 (subcritical
    cylinder, `aero.body_crossflow_cd`), η = 0.72 for L/D = 15. OpenRocket's K = 1.1 overstates the
    broadside force by ≈ 25 %.
  * Fins and canards: per-fin Diederich lift slope in spanwise strips, plus a flat-plate crossflow term
    (C_dc 1.2) at high α. Both use the dynamic pressure of the flow in the fin's chordwise plane; the
    interference factors below act on the attached-flow lift only, and lift fades as cos α so a broadside
    fin carries crossflow only. Lift at the MAC quarter chord (three-quarter chord in reverse flow),
    crossflow at mid-chord.
  * Fin interference (`interference.py`):
    * Body angle of attack: Barrowman's K_T = 1 + r/(s+r), as in OpenRocket.
    * Fin deflections (canard commands, cant, misalignment) and roll rate: slender-body factors computed at
      build time for each fin set's r/s and fin count. Canards: own load 0.90 (K_T would give 1.48), all
      four in roll 0.70, a pitch pair 0.94 plus 0.50 carried by the body, roll damping 0.71.
    * Body carry-over of a deflected canard scaled by `aero.canard_body_carryover` (hinge gap at the root).
  * Drag: skin friction with a roughness limit, pressure drag, fin edge drag by cross-section, base drag,
    OpenRocket's axial-force-vs-α multiplier (zero at 90°). Tail-first, the build-up is redone for the
    reversed body (base facing the flow): CD 0.94 against 0.39 nose-first.
  * Canard vortices (`aero.canard_vortex`): one tip vortex per canard, strength from that canard's lift plus
    its body carry-over. The vortex and its image in the body are carried to the tail fins (roll lag,
    crossflow drift, Lamb–Oseen core) and change every fin strip's angle of attack. Resolved per
    canard–fin pair, dependent on fin-can angle, including the downwash `static_cp` reports.
  * Canard hinge moments: each strip's force at its local quarter chord, about the hinge line.
  * Subsonic only (to about Mach 0.8).
* **Mass** (`mass.py`): every component is a mass element with a full inertia tensor. Motor: case hardware
  plus four burning grains following openMotor. Jet damping and the dI/dt term included.
* **Environment:**
  * ISA atmosphere (optional temperature offset), latitude-dependent gravity
  * constant or power-law wind, optional Dryden-type turbulence
* **Disturbances** (build errors; all zero is a perfect rocket):
  * thrust misalignment and offset, thrust and drag scale
  * misalignment of each fin; for the canards this is the canard zero offset
  * CG offset (`cg_offset_mm`): the vehicle's lateral CG offset at liftoff. The off-axis mass is put in the
    airframe (without motor and fin can) and the motor stays on the axis, so the offset grows as m(0)/m(t)
    during the burn (≈ +10 %). Thrust, drag and the aerodynamic forces act about the offset CG, and the
    sensors' lever arms use it.
  * airframe principal-axis tilt (`inertia_axis_tilt_deg`): products of inertia. Negligible while the body
    roll stays slow.
  * nose cone misalignment (`nose_misalignment_deg`): the nose takes its angle of attack about its own axis,
    so it lifts at zero body angle of attack (CN = 2·δ at the nose CP for small angles).
  * fin can imbalance: CG off the bearing axis (`fincan_cg_offset_mm`, static) and principal axis tilted
    from it (`fincan_axis_tilt_deg`, couple). Both act as a product of inertia P that turns with the fin
    can, P = (1 − m_f/m)·L·m_f·e + (I_t − I_x)·τ, giving a moment Ω²·(e_x × P) on the vehicle at the
    fin can's spin rate Ω. The CG wobbles by m_f·e/m, and the IMUs feel the airframe's vibration
    Ω²·m_f·e/m. The Ω′ term and the gyroscopic coupling with the slow body rates are neglected.
* **Canard servos** (`actuators.py`): `ideal` or `servo`. The servo is a second-order position loop (natural
  frequency, damping, holding stiffness) with stall torque, torque–speed rate limit, deadband, command
  quantisation, backlash and end stops, loaded by the hinge moment.
* **Sensors** (`sensors.py`): own rate, latency, mounting position, `enabled` switch and random stream (from
  `sensors.seed`).
  * IMU: specific force at the sensor point (lever-arm terms), turn-on bias, bias drift, noise, scale factor,
    misalignment, gyro g-sensitivity, saturation, quantisation.
  * Barometer: ISA pressure at the sensor, static-port error ∝ q, port lag, bias, noise, quantisation.
  * GNSS: ENU position and velocity with a slowly varying error, loss of lock above 4 g, re-acquisition after 1 s.
* **Pad phase:** for `simulation.pad_time_s` before ignition the rocket sits on the rail with sensors,
  controller and servos running.

## Validation

`configs/validation_openrocket.yaml` vs `data/all.csv`. `all.csv` is from an older revision of the design
(`yay it works.rkt`). The validation config rebuilds that revision from the current .ork with five logged
overrides, locks the fin can and switches off the canard vortices (neither is modelled by OpenRocket).

| | rocket_dt | OpenRocket |
|---|---|---|
| Apogee | 1925.9 m | 1931.0 m (−0.26 %) |
| Time to apogee | 19.69 s | 19.74 s |
| Max speed | 223.5 m/s | 223.9 m/s |
| Liftoff mass / CG | 17.152 kg / 1.2400 m | 17.150 kg / 1.2406 m |
| CD at OpenRocket's (M, Re) | +0.002 mean diff | 0.42 |
| CP (small α) | +0.4 cm mean diff | |

Component masses agree to better than 0.001 %. Known differences:
* Roll inertia (+16 %): OpenRocket models the motor as a solid cylinder.
* Damping: OpenRocket's empirical damping terms give a less damped angle-of-attack oscillation.
* Descent: OpenRocket clips fin lift at 20° and falls tail-first; this model turns nose-down ≈ 3 s after apogee.

## Placeholders

| Item | Current value | Source for a measured value |
|---|---|---|
| Fin-can roll inertia | 0.026 kg m² (upper-bound estimate) | CAD: fin can → Properties |
| Bearing friction (61820) | 0.02 N m Coulomb, 1e-4 N m s viscous | spin-down test of the built fin can |
| Grain position in case, case hardware CG | stack centred, no gaps; case centre | motor drawing |
| Canard servo | 15 Hz, ζ 0.7, 40 N m/rad, 1.5 N m stall, 500 °/s, deadband 0.1°, resolution 0.09°, backlash 0.2° | bench tests: small-step response, torque vs deflection, play at the canard |
| Canard hinge line | planform centroid (54 mm aft of the root LE) | actual hinge position |
| IMUs | ICM-42688- and BMI088-class datasheet values, both at station 1.015 m | chosen parts' datasheets, mounting, bench calibration |
| Barometer | MS5611-class, static-port error 0 | datasheet; static-port coefficient from flight data |
| GNSS | M10-class, off | chosen receiver's dynamics limits |
| Canard vortex strength / core radius | factor 1.0 / 6 mm (dispersed 0.5–1.5 in Monte Carlo) | CFD, or fin-can spin under a known canard offset in flight |
| Canard body carry-over | 0.5 of the slender-body value (dispersed 0–1 in Monte Carlo) | hinge gap; wind tunnel or pitch response to a canard step in flight |
| Body crossflow drag coefficient | 1.2 (subcritical cylinder) | only matters at high α; descent is insensitive to it |
| `vertical_hold` gains | kp 5, ki 2.5, kd 75 m, k_roll 0.76 m | tuning against measured servo and sensors |

## Findings

* Static margin 1.8–2.2 cal during the ascent; canard downwash on the tail costs ≈ 0.14 cal.
* Roll authority is high with a free fin can. The canards are the body's only roll damping, so 0.1° of
  canard misalignment gives ≈ 150 °/s of body roll, and 5° gives ≈ 9,000 °/s. Servo resolution, backlash
  and canard zero offset have a large effect.
* The hinge moment pushes the canards away from neutral (canard CP 16 mm ahead of a centroid hinge).
  * With the placeholder servo, a 0.5° command becomes ≈ 0.72° in flight: quantisation to 0.54°, half the
    backlash, servo compliance.
  * With zero command, angle-of-attack loads hold the canards at ≈ ±0.17° inside the backlash.
  * A hinge about 38 mm aft of the root LE (at the CP) would remove most of the load.
* Canard vortices reverse the roll on the tail.
  * Induced torque on the fin can: −0.87× the canard roll moment with fins in line with the canards,
    −1.35× with the vortices midway between fins.
  * The free fin can absorbs this and is kicked the opposite way during canard transients; the body's roll
    response is unchanged.
  * One vortex per canard, strength from an impulse balance (lift plus body carry-over). Slender-body theory
    puts the circulation shed at the root within ≈ 20 % of it.
* Without recovery events the descent is ballistic. The rocket hangs broadside for ≈ 2.5–3.5 s after apogee
  (≈ 12 m/s), then turns nose-down and hits the ground at ≈ 157 m/s ≈ 40 s after launch. Broadside, the CP is
  0–0.3 cal aft of the CG (depending on fin-can roll angle), so this phase is marginally stable; it barely
  changes over body crossflow C_dc 0.4–1.8. Not modelled: wake shielding of the leeward fins, drag crisis of
  the body (crossflow above ≈ 35 m/s).
* **Build errors** (measured on the baseline):
  * 1 mm CG offset: max tilt to apogee 45° → 55° without control, and 7.0° → 15.4° with `vertical_hold`.
    With `vertical_hold` the extra tilt comes at apogee, where canard authority runs out.
  * 0.3° nose misalignment: 45° → 65° without control, 7.0° → 7.6° with `vertical_hold`.
  * 0.3° airframe principal-axis tilt: no visible effect.
  * Fin can imbalance needs the fin can to spin. In `fincan_disturbance_demo.yaml` it reaches only ≈ 1 Hz,
    where 0.5 mm of static imbalance gives < 0.01 N m. The moment grows with Ω²: it would matter at tens of
    Hz. The body roll and fin can spin stay 2.5–5× below the pitch natural frequency (0.3–2.5 Hz), so
    there is no roll resonance there.

## Open items

* Estimator aiding with barometer (and GNSS when flown); levelling with the known rail angle.
* Guidance along a path f(x, y, z) = 0 (vertical line, or a rail tilted up to ≈ 5°); roll-angle hold.
* Bench data for the servo and sensors to replace the placeholders.
* Recovery events (drogue/main) for descent and landing dispersion.
* Software in the loop with the real flight software.
