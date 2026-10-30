"""Strict YAML configuration.

Single-source-of-truth rules enforced here:

* Airframe geometry, materials and mass components come only from the .ork file.
  Deviations are expressed as explicit ``rocket.overrides`` entries, each of which
  is logged with its note when the rocket is built.
* Motor data comes only from the motor files (see ``rocket_dt.io.motor``).
* Launch site, environment, disturbances, sensors, canard servos, the controller and
  integrator settings come only from this config. Simulation settings stored inside
  the .ork are ignored.
* Unknown keys are an error (no silently ignored typos); required keys have no
  defaults.
"""

from __future__ import annotations

import copy
import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class OverrideSpec:
    component: str  # ork path, e.g. "motor case transition/Trapezoidal Fin Set"
    set: dict  # field -> value
    note: str  # where the value comes from (required: traceability)


@dataclass
class BearingConfig:
    coulomb_torque: float  # N m, dry friction torque of the fin-can bearing(s)
    viscous_coeff: float  # N m s/rad
    smoothing_rate: float = 0.5  # rad/s, width of the tanh() used to smooth sign()


@dataclass
class FinCanConfig:
    components: list[str]  # ork paths of components (with their subtrees) that spin with the fin can
    bearing: BearingConfig
    enabled: bool = True  # False = fin can rigidly locked to the body
    roll_inertia: float | None = None  # kg m^2 about the rocket axis; None = from geometry


@dataclass
class RocketConfig:
    ork: Path
    canards: str | None = None  # ork path of the deflectable fin set
    fin_can: FinCanConfig | None = None
    overrides: list[OverrideSpec] = field(default_factory=list)


@dataclass
class MotorConfig:
    burn_data: Path  # openMotor CSV export: thrust + per-grain propellant mass
    grains: Path  # openMotor .ric: grain geometry
    eng: Path  # RASP .eng: loaded mass, case diameter/length (thrust only cross-checked)
    grain_stack_offset: float = 0.0  # m, grain-stack centre aft of case centre
    grain_spacing: float = 0.0  # m, gap between grains
    hardware_cg_offset: float = 0.0  # m, case-hardware CG aft of case centre


@dataclass
class LaunchConfig:
    rail_length: float  # m, distance travelled while constrained
    angle_from_vertical_deg: float
    azimuth_deg: float  # direction the rail leans towards, clockwise from north
    altitude_msl: float
    latitude_deg: float


@dataclass
class TurbulenceConfig:
    model: str = "none"  # "none" | "dryden"
    intensity: float = 0.1  # sigma / mean wind speed
    seed: int = 0


@dataclass
class WindConfig:
    speed: float  # m/s at reference height (or everywhere for "constant")
    direction_from_deg: float  # meteorological: direction the wind blows FROM, clockwise from north
    profile: str = "constant"  # "constant" | "power_law"
    reference_height: float = 10.0
    exponent: float = 1.0 / 7.0
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig)


@dataclass
class AtmosphereConfig:
    temperature_offset_K: float = 0.0
    sea_level_pressure: float = 101325.0


@dataclass
class EnvironmentConfig:
    wind: WindConfig
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)


@dataclass
class DisturbanceConfig:
    thrust_misalignment_deg: list[float] = field(default_factory=lambda: [0.0, 0.0])  # about body y, z
    thrust_offset_m: list[float] = field(default_factory=lambda: [0.0, 0.0])  # nozzle offset in body y, z
    thrust_scale: float = 1.0
    drag_scale: float = 1.0
    cg_offset_m: list[float] = field(default_factory=lambda: [0.0, 0.0])  # lateral CG offset, body y, z
    fin_misalignment_deg: dict = field(default_factory=dict)  # ork path -> per-fin cant error list


@dataclass
class CanardVortexConfig:
    """Canard tip vortices acting on the fin sets aft of the canards (see aero.py)."""
    model: str = "none"  # "none" | "vortex"
    strength_factor: float = 1.0  # multiplies every vortex circulation (calibration)
    core_radius: float = 0.006  # m, Lamb-Oseen viscous core radius
    roll_lag: bool = True  # body rolls under the vortex while it convects to the fins


@dataclass
class AeroConfig:
    canard_vortex: CanardVortexConfig = field(default_factory=CanardVortexConfig)
    # fraction of the slender-body body carry-over load of a deflected canard that the body really
    # gets: 1 = sealed root, less with a hinge gap between the canard root and the body
    canard_body_carryover: float = 1.0


@dataclass
class CanardActuatorConfig:
    """Canard servos, one per canard (see actuators.py). Placeholder values until bench tests."""
    model: str = "ideal"  # "ideal" (follows the command instantly) | "servo" (second order + nonlinearities)
    limit_deg: float = 15.0  # mechanical stop and command limit
    natural_frequency_hz: float = 15.0  # closed-loop small-signal response (step test)
    damping_ratio: float = 0.7
    stiffness: float = 40.0  # N m/rad, holding stiffness under load (sets the inertia: J = K / wn^2)
    stall_torque: float = 1.5  # N m
    max_rate_dps: float = 500.0  # no-load speed (datasheet "0.12 s/60 deg" = 500 deg/s)
    deadband_deg: float = 0.0  # servo does not react to position errors smaller than this (full width)
    resolution_deg: float = 0.0  # command quantisation (PWM / DAC resolution)
    backlash_deg: float = 0.0  # free play between servo output and canard (full width)
    hinge_from_root_le: float | None = None  # m, chordwise hinge line position; None = planform area centroid


@dataclass
class ActuatorsConfig:
    canards: CanardActuatorConfig = field(default_factory=CanardActuatorConfig)


@dataclass
class AccelConfig:
    range_g: float = 16.0
    bits: int = 16
    noise_density_ug_rthz: float = 100.0  # white noise, micro-g / sqrt(Hz)
    bias_sigma_mg: float = 20.0  # turn-on bias (drawn once per flight)
    bias_instability_mg: float = 0.0  # slowly wandering bias (first-order Gauss-Markov)
    bias_correlation_s: float = 100.0
    scale_factor_sigma_pct: float = 0.5
    misalignment_sigma_deg: float = 0.5  # axis non-orthogonality + mounting error


@dataclass
class GyroConfig:
    range_dps: float = 2000.0
    bits: int = 16
    noise_density_mdps_rthz: float = 5.0  # milli-deg/s / sqrt(Hz)
    bias_sigma_dps: float = 0.5
    bias_instability_dps: float = 0.0
    bias_correlation_s: float = 100.0
    scale_factor_sigma_pct: float = 0.5
    misalignment_sigma_deg: float = 0.5
    g_sensitivity_dps_per_g: float = 0.0  # rate error caused by acceleration


@dataclass
class ImuConfig:
    name: str
    station: float  # m from the nose tip
    enabled: bool = True
    rate_hz: float = 500.0
    latency_s: float = 0.0
    offset_yz_m: list[float] = field(default_factory=lambda: [0.0, 0.0])  # lateral position, body y, z
    orientation_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # nominal mount roll, pitch, yaw
    accel: AccelConfig = field(default_factory=AccelConfig)
    gyro: GyroConfig = field(default_factory=GyroConfig)


@dataclass
class BaroConfig:
    station: float  # m from the nose tip
    enabled: bool = True
    rate_hz: float = 50.0
    latency_s: float = 0.0
    noise_pa: float = 2.0
    bias_sigma_pa: float = 100.0
    resolution_pa: float = 1.2
    lag_s: float = 0.02  # first-order lag of the vented bay / port
    static_error_coeff: float = 0.0  # static-port error: measured - true pressure = coeff * dynamic pressure
    temperature_noise_k: float = 0.1
    temperature_bias_sigma_k: float = 1.0


@dataclass
class GnssConfig:
    station: float  # antenna, m from the nose tip
    enabled: bool = True
    rate_hz: float = 10.0
    latency_s: float = 0.1
    horizontal_sigma_m: float = 1.5  # slowly varying position error (Gauss-Markov)
    vertical_sigma_m: float = 3.0
    correlation_time_s: float = 60.0
    white_noise_m: float = 0.3
    velocity_sigma_mps: float = 0.1
    max_accel_g: float = 4.0  # receiver loses lock above this acceleration
    reacquire_s: float = 1.0  # time below the limit before fixes are valid again


@dataclass
class SensorsConfig:
    seed: int = 0  # all random sensor errors (turn-on biases, noise) derive from this
    imu: list[ImuConfig] = field(default_factory=list)
    baro: BaroConfig | None = None
    gnss: GnssConfig | None = None


@dataclass
class ControllerConfig:
    type: str = "none"  # a name registered in rocket_dt.controllers.CONTROLLERS
    rate_hz: float = 100.0
    params: dict = field(default_factory=dict)  # fields of the controller's Params dataclass (checked strictly)


@dataclass
class SimulationConfig:
    dt: float = 0.002
    t_max: float = 300.0
    stop_at: str = "ground"  # "apogee" | "ground"
    log_dt: float = 0.01
    pad_time_s: float = 0.0  # time on the pad before ignition: sensors and controller run (calibration)


@dataclass
class MonteCarloConfig:
    n_runs: int = 100
    seed: int = 1
    processes: int | None = None
    dispersions: dict = field(default_factory=dict)  # dotted config path -> {dist, sigma|low,high}


@dataclass
class Config:
    rocket: RocketConfig
    motor: MotorConfig
    launch: LaunchConfig
    environment: EnvironmentConfig
    aero: AeroConfig = field(default_factory=AeroConfig)
    actuators: ActuatorsConfig = field(default_factory=ActuatorsConfig)
    sensors: SensorsConfig = field(default_factory=SensorsConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    disturbances: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    montecarlo: MonteCarloConfig | None = None
    # bookkeeping (not from YAML)
    source: Path | None = None
    raw: dict = field(default_factory=dict, repr=False)


_NOT_FROM_YAML = {"source", "raw"}


def _convert(tp, value, where: str, base: Path):
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        return _convert(non_none[0], value, where, base)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where, base)
    if tp is Path:
        p = Path(value)
        return p if p.is_absolute() else (base / p).resolve()
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        return [_convert(args[0], v, f"{where}[{i}]", base) for i, v in enumerate(value)]
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    if origin is dict or tp is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        return value
    return value


def build_dataclass(cls, data, where: str, base: Path = Path(".")):
    """Build a (nested) dataclass from a mapping with the same strict checks as the config."""
    return _build(cls, data, where, base)


def _build(cls, data, where: str, base: Path):
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {data!r}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)} - _NOT_FROM_YAML
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(names)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in _NOT_FROM_YAML:
            continue
        if f.name in data:
            kwargs[f.name] = _convert(hints[f.name], data[f.name], f"{where}.{f.name}", base)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{where}: missing required key '{f.name}'")
    return cls(**kwargs)


def parse_config(raw: dict, base_dir: Path, source: Path | None = None) -> Config:
    cfg = _build(Config, raw, "config", base_dir)
    cfg.source = source
    cfg.raw = copy.deepcopy(raw)
    _validate(cfg)
    return cfg


def _deep_merge(base: dict, child: dict) -> dict:
    """Child values win; mappings merge recursively; lists are replaced wholesale."""
    out = copy.deepcopy(base)
    for k, v in child.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _load_raw(path: Path, seen: tuple = ()) -> dict:
    """Load YAML, resolving ``extends: other.yaml`` (relative to this file).

    Relative file paths in each file are made absolute relative to *that* file, so
    a child config can extend a parent in another directory.
    """
    if path in seen:
        raise ConfigError(f"Circular 'extends': {' -> '.join(map(str, seen + (path,)))}")
    raw = yaml.safe_load(path.read_text()) or {}
    for section, keys in (("rocket", ("ork",)), ("motor", ("burn_data", "grains", "eng"))):
        for k in keys:
            v = raw.get(section, {}).get(k) if isinstance(raw.get(section), dict) else None
            if isinstance(v, str) and not Path(v).is_absolute():
                raw[section][k] = str((path.parent / v).resolve())
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_load_raw((path.parent / parent).resolve(), seen + (path,)), raw)


def load_raw(path: str | Path) -> dict:
    """The fully merged raw dict of a config file (``extends`` resolved, file paths absolute)."""
    return _load_raw(Path(path).resolve())


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    return parse_config(_load_raw(path), path.parent, path)


def _validate(cfg: Config) -> None:
    if cfg.simulation.stop_at not in ("apogee", "ground"):
        raise ConfigError("simulation.stop_at must be 'apogee' or 'ground'")
    if cfg.environment.wind.profile not in ("constant", "power_law"):
        raise ConfigError("environment.wind.profile must be 'constant' or 'power_law'")
    if cfg.environment.wind.turbulence.model not in ("none", "dryden"):
        raise ConfigError("environment.wind.turbulence.model must be 'none' or 'dryden'")
    cv = cfg.aero.canard_vortex
    if cv.model not in ("none", "vortex"):
        raise ConfigError("aero.canard_vortex.model must be 'none' or 'vortex'")
    if cv.strength_factor < 0 or cv.core_radius <= 0:
        raise ConfigError("aero.canard_vortex: strength_factor must be >= 0 and core_radius > 0")
    if not 0.0 <= cfg.aero.canard_body_carryover <= 1.0:
        raise ConfigError("aero.canard_body_carryover must be in [0, 1]")
    act = cfg.actuators.canards
    if act.model not in ("ideal", "servo"):
        raise ConfigError("actuators.canards.model must be 'ideal' or 'servo'")
    if min(act.natural_frequency_hz, act.damping_ratio, act.stiffness, act.stall_torque, act.max_rate_dps,
           act.limit_deg) <= 0 or min(act.deadband_deg, act.resolution_deg, act.backlash_deg) < 0:
        raise ConfigError("actuators.canards: dynamics parameters must be > 0, deadband/resolution/backlash >= 0")
    sensors = [*cfg.sensors.imu, *(x for x in (cfg.sensors.baro, cfg.sensors.gnss) if x is not None)]
    names = [i.name for i in cfg.sensors.imu]
    if len(set(names)) != len(names) or {"baro", "gnss"} & set(names):
        raise ConfigError("sensors.imu: names must be unique (and not 'baro' or 'gnss')")
    for sc in sensors:
        if not (0 < sc.rate_hz <= 1.0 / cfg.simulation.dt + 1e-9):
            raise ConfigError(f"sensor rate {sc.rate_hz} Hz must be in (0, 1/simulation.dt]")
        if sc.latency_s < 0:
            raise ConfigError("sensor latency must be >= 0")
    for imu in cfg.sensors.imu:
        if len(imu.offset_yz_m) != 2 or len(imu.orientation_deg) != 3:
            raise ConfigError(f"sensors.imu '{imu.name}': offset_yz_m needs 2 values, orientation_deg 3")
    if not (0 < cfg.simulation.dt <= 0.01):
        raise ConfigError("simulation.dt must be in (0, 0.01] s")
    if cfg.simulation.pad_time_s < 0:
        raise ConfigError("simulation.pad_time_s must be >= 0")
    if not (0 < cfg.controller.rate_hz <= 1.0 / cfg.simulation.dt + 1e-9):
        raise ConfigError("controller.rate_hz must be in (0, 1/simulation.dt]")
    if cfg.simulation.log_dt < cfg.simulation.dt:
        raise ConfigError("simulation.log_dt must be >= simulation.dt")
    for p in (cfg.rocket.ork, cfg.motor.burn_data, cfg.motor.grains, cfg.motor.eng):
        if not p.exists():
            raise ConfigError(f"File not found: {p}")
    for name in ("thrust_misalignment_deg", "thrust_offset_m", "cg_offset_m"):
        if len(getattr(cfg.disturbances, name)) != 2:
            raise ConfigError(f"disturbances.{name} must have 2 entries")


def set_by_path(raw: dict, dotted: str, value) -> None:
    """Set a value in a raw config dict by dotted path ('a.b.0.c').

    Keys may contain '/' (ork component paths) but not '.'.
    """
    keys = dotted.split(".")
    node = raw
    for k in keys[:-1]:
        node = node[int(k)] if isinstance(node, list) else node.setdefault(k, {})
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
