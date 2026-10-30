"""Controllers that can be selected from a config (``controller.type``).

Every module in this package that defines ``CONTROLLER`` is registered under its module
name: ``rocket_dt/controllers/my_controller.py`` becomes ``controller.type: my_controller``.
The module needs

* ``Params``: a dataclass whose fields are the config's ``controller.params``
* ``CONTROLLER``: a ``Controller`` subclass whose constructor takes ``(rocket, cfg, params)``

so adding a controller never touches a shared file. Modules without ``CONTROLLER``
(helpers such as ``estimation.py``) are skipped.
"""

from __future__ import annotations

import importlib
import pkgutil

from rocket_dt.config import Config, ConfigError, build_dataclass
from rocket_dt.control import Controller, NoControl


def _discover() -> dict:
    found = {"none": (None, lambda rocket, cfg, params: NoControl(rocket.aero.n_canards))}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{info.name}")
        cls = getattr(mod, "CONTROLLER", None)
        if cls is None:
            continue
        if not (isinstance(cls, type) and issubclass(cls, Controller)):
            raise TypeError(f"{mod.__name__}.CONTROLLER must be a Controller subclass")
        found[info.name] = (getattr(mod, "Params", None), cls)
    return found


CONTROLLERS = _discover()


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
