"""Controllers that can be selected from a config (``controller.type``).

To add one: write a module here with a ``Params`` dataclass (its fields are the
config's ``controller.params``) and a ``Controller`` subclass whose constructor takes
``(rocket, cfg, params)``, then add it to ``CONTROLLERS``.
"""

from __future__ import annotations

from rocket_dt.config import Config, ConfigError, build_dataclass
from rocket_dt.control import Controller, NoControl
from rocket_dt.controllers import vertical_hold


CONTROLLERS = {
    "none": (None, lambda rocket, cfg, params: NoControl(rocket.aero.n_canards)),
    "vertical_hold": (vertical_hold.Params, vertical_hold.VerticalHold),
}


def params_class(name: str):
    if name not in CONTROLLERS:
        raise ConfigError(f"controller.type '{name}' is not registered; available: {sorted(CONTROLLERS)}")
    return CONTROLLERS[name][0]


def make_controller(cfg: Config, rocket) -> Controller:
    c = cfg.controller
    params_cls = params_class(c.type)
    if params_cls is None:
        if c.params:
            raise ConfigError(f"controller '{c.type}' takes no params")
        params = None
    else:
        params = build_dataclass(params_cls, c.params, "controller.params")
    ctrl = CONTROLLERS[c.type][1](rocket, cfg, params)
    ctrl.rate_hz = c.rate_hz
    return ctrl
