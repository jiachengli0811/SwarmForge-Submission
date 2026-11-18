from __future__ import annotations

from typing import Dict, Optional

from .model import (load, save, save_record, from_dict,
                    from_record, is_record, unset_params)
from .evaluate import Controller, Scene
from .environment import Environment
from .environment import load as load_environment
from ..classify import AlgorithmIR

TARGETS = {"boids": "boids", "adaptive": "adaptive",
           "vasarhelyi": "vasarhelyi", "gradient": "gradient"}

KEY_TO_TARGET = {"pyglet-boids": "boids", "target-boids": "boids",
                 "adaptive-swarm": "adaptive",
                 "swarmlab-vasarhelyi": "vasarhelyi",
                 "vasarhelyi": "vasarhelyi",
                 "swarmalgorithms-gradient": "gradient",
                 "gradient": "gradient"}


def adapter(target: str):
    import importlib
    if target not in TARGETS:
        raise KeyError("unknown target %r (have: %s)"
                       % (target, ", ".join(sorted(TARGETS))))
    return importlib.import_module(".adapters." + TARGETS[target], __package__)


def target_for(algo: AlgorithmIR) -> str:
    try:
        return KEY_TO_TARGET[algo.target_key]
    except KeyError:
        raise KeyError("cannot infer the target from target_key=%r; pass one of %s"
                       % (algo.target_key, ", ".join(sorted(TARGETS))))


def run_target(target: str, algo: Optional[AlgorithmIR] = None,
               env=None, **kw) -> Dict[str, float]:
    mod = adapter(target)
    if isinstance(env, str):
        env = load_environment(env)
    if env is not None:
        kw["env"] = env
    ctrl = None if algo is None else Controller(algo, dims=mod.DIMS)
    return mod.run(ctrl, **kw)
