"""rocket_dt: 6-DOF digital twin of a solid-propellant rocket.

Typical use::

    from rocket_dt import load_config, build_simulation
    sim = build_simulation(load_config("configs/baseline.yaml"))
    result = sim.run()
    print(result.summary())
"""


def __getattr__(name):
    if name == "load_config":
        from rocket_dt.config import load_config
        return load_config
    if name in ("build_rocket", "build_simulation"):
        from rocket_dt import builder
        return getattr(builder, name)
    raise AttributeError(name)


__all__ = ["load_config", "build_rocket", "build_simulation"]
