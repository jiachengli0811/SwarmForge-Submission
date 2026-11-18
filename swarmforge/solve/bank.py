from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .state import State

BANK_VERSION = 2

_ARRAYS = ("positions", "velocities", "obstacles", "obstacle_velocities",
           "obstacle_radii", "dynamic", "target", "route", "obstacle_axes")


@dataclass
class Bank:
    state: State
    meta: Dict[str, Any]
    path: Path
    created: bool


def describe(model, samples: int, seed: int) -> Dict[str, Any]:
    config = model.config
    physics = config.physics
    description = {
        "bank_version": BANK_VERSION,
        "family": config.name,
        "model": model.name,
        "spec": {key: value for key, value in
                 dataclasses.asdict(model.spec).items() if value is not None},
        "focus": config.sampling.focus,
        "samples": int(samples),
        "seed": int(seed),
        "chunk": config.sampling.chunk,
        "physics": {
            "dt": physics.dt,
            "agent_radius": physics.agent_radius,
            "agent_min_gap": physics.agent_min_gap,
            "obstacle_min_gap": physics.obstacle_min_gap,
            "plant": dataclasses.asdict(physics.plant),
            "targets": dataclasses.asdict(physics.targets),
        },
        "state": _state(config.state),
        "reaches": {key: value for key, value in
                    dataclasses.asdict(model.reach.reaches).items()
                    if key != "walls" or value is not None},
    }
    window = model.reach.describe(model.spec)
    if window:
        description["window"] = window
    if physics.square_bodies:
        description["contact"] = "square"
    return description


def _state(state) -> Dict[str, Any]:
    out = dataclasses.asdict(state)
    for kind in ("static", "dynamic"):
        for option in ("wall_share", "standoff"):
            if out.get(kind) and not out[kind].get(option):
                out[kind].pop(option, None)
        if out.get(kind) and not out[kind].get("wall_share"):
            for option in ("wall_extent", "wall_length", "wall_thickness"):
                out[kind].pop(option, None)
    if out["window"].get("walls") is None:
        out["window"].pop("walls", None)
    if not out.get("route_final_share"):
        out.pop("route_final_share", None)
    return out


def fingerprint(description: Dict[str, Any]) -> str:
    text = json.dumps(description, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def bank_path(model, description: Dict[str, Any]) -> Path:
    return (Path(model.config.sampling.bank_dir) / model.config.name
            / ("%s-%s.npz" % (model.name, fingerprint(description))))


def save_bank(state: State, meta: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(state, name) for name in _ARRAYS}
    arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npz")
    os.close(handle)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_bank(path: Path) -> Bank:
    from ..paths import relocate_path

    path = relocate_path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        state = State(**{name: np.array(data[name]) for name in _ARRAYS
                         if name in data.files})
    return Bank(state=state, meta=meta, path=Path(path), created=False)


def bank_for(model, samples: Optional[int] = None,
             seed: Optional[int] = None) -> Bank:
    sampling = model.config.sampling
    samples = sampling.samples if samples is None else int(samples)
    seed = sampling.seed if seed is None else int(seed)
    description = describe(model, samples, seed)
    path = bank_path(model, description)
    if path.is_file():
        bank = load_bank(path)
        if bank.meta.get("fingerprint") != fingerprint(description):
            raise ValueError("%s holds a sampling drawn from something else; "
                             "remove it" % path)
        return bank
    started = time.perf_counter()
    state, info = model.sample(samples, seed)
    meta = {"fingerprint": fingerprint(description),
            "description": description, "sampler": info,
            "seconds": time.perf_counter() - started,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_bank(state, meta, path)
    return Bank(state=state, meta=meta, path=path, created=True)


__all__ = ["BANK_VERSION", "Bank", "bank_for", "bank_path", "describe",
           "fingerprint", "load_bank", "save_bank"]
