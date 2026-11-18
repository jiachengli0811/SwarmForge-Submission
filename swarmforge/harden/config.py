from __future__ import annotations

import dataclasses

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import yaml

from .calibrate import Calibrate, read_calibrate
from .scoring import Scoring, mission_files, read_scoring
from .weakness import Objective

BLOCK = "harden"
REQUIRED = ("objective", "max_evaluations", "seed", "workers")

PENDING = "pending"


@dataclass(frozen=True)
class Component:
    name: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.options, name=self.name)


@dataclass(frozen=True)
class HardenConfig:
    objective: Objective
    max_evaluations: int
    seed: int
    workers: int
    scoring: Scoring = field(default_factory=Scoring)
    policy: Component = field(default_factory=lambda: Component(PENDING))
    strategy: Component = field(default_factory=lambda: Component(PENDING))
    calibrate: Optional[Calibrate] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {"objective": self.objective.to_dict(),
               "max_evaluations": self.max_evaluations, "seed": self.seed,
               "workers": self.workers,
               "scoring": self.scoring.to_dict(),
               "policy": self.policy.to_dict(),
               "strategy": self.strategy.to_dict()}
        if self.calibrate is not None:
            out["calibrate"] = self.calibrate.to_dict()
        return out


def _component(raw: Any, label: str) -> Component:
    if raw is None:
        return Component(PENDING)
    if isinstance(raw, str):
        return Component(raw)
    if not isinstance(raw, Mapping):
        raise ValueError("%s.%s must be a name or a mapping" % (BLOCK, label))
    options = {str(k): v for k, v in raw.items() if k != "name"}
    return Component(str(raw.get("name") or PENDING), options)


Reader = Callable[[Any], Any]


def number(kind: type = float, lo: Optional[float] = None,
           hi: Optional[float] = None, *, above: bool = False) -> Reader:
    def read(value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("not a number")
        out = kind(value)
        if out != value:
            raise ValueError("not an integer")
        if lo is not None and (out < lo or (above and out == lo)):
            raise ValueError("out of range")
        if hi is not None and out > hi:
            raise ValueError("out of range")
        return out
    read.rule = "%s %s%s, %s]" % (
        "an integer" if kind is int else "a number",
        "(" if above else "[", "-inf" if lo is None else "%g" % lo,
        "inf" if hi is None else "%g" % hi)
    return read


def flag() -> Reader:
    def read(value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError("not true or false")
        return value
    read.rule = "true or false"
    return read


def read_options(where: str, given: Mapping[str, Any],
                 readers: Mapping[str, Reader]) -> Dict[str, Any]:
    unknown = sorted(set(given) - set(readers))
    if unknown:
        raise ValueError("%s does not read %s (it reads: %s)"
                         % (where, ", ".join(unknown), ", ".join(readers)))
    missing = [key for key in readers if given.get(key) is None]
    if missing:
        raise ValueError("%s needs %s -- none has a default"
                         % (where, ", ".join(missing)))
    out: Dict[str, Any] = {}
    for key, read in readers.items():
        try:
            out[key] = read(given[key])
        except (TypeError, ValueError):
            raise ValueError("%s: %s must be %s, not %r"
                             % (where, key, getattr(read, "rule", "valid"),
                                given[key])) from None
    return out


def read_harden(path, overrides: Optional[Mapping[str, Any]] = None
                ) -> HardenConfig:
    path = Path(path)
    document = yaml.safe_load(path.read_text()) or {}
    raw = document.get(BLOCK)
    if raw is None:
        raise ValueError("%s has no `%s:` block" % (path, BLOCK))
    if not isinstance(raw, Mapping):
        raise ValueError("%s: `%s` must be a mapping" % (path, BLOCK))
    merged: Dict[str, Any] = dict(raw)
    overrides = dict(overrides or {})
    missions = overrides.pop("missions", None)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    scoring = read_scoring(merged.get("scoring"), path.parent)
    if missions:
        if not scoring.region_reach:
            raise ValueError("missions are the reach-aware region "
                             "scoring's: %s scores by %s"
                             % (path, scoring.name))
        scoring = dataclasses.replace(
            scoring, missions=mission_files(missions, Path.cwd()))
    if scoring.region_reach and not scoring.missions:
        raise ValueError(
            "%s: %s, and none are given: write harden.scoring.missions or "
            "pass --missions"
            % (path, "the flow reach (harden.scoring.reach) flies the "
               "family's representative missions" if scoring.flow_reach else
               "the reach (harden.scoring.reach) starts from the original's "
               "failures on given missions"))
    missing =["%s.%s" % (BLOCK, key) for key in REQUIRED
               if merged.get(key) is None]
    if missing:
        raise ValueError("%s cannot run a hardening: it declares no %s. Write "
                         "the key%s or pass the flag -- none has a default"
                         % (path, ", ".join(missing),
                            "" if len(missing) == 1 else "s"))
    weights = merged["objective"]
    if not isinstance(weights, Mapping):
        weights = {str(name): 1.0 for name in weights}
    objective = Objective({str(k): float(v) for k, v in weights.items()})
    config = HardenConfig(
        objective=objective,
        max_evaluations=int(merged["max_evaluations"]),
        seed=int(merged["seed"]), workers=int(merged["workers"]),
        scoring=scoring,
        policy=_component(merged.get("policy"), "policy"),
        strategy=_component(merged.get("strategy"), "strategy"),
        calibrate=read_calibrate(merged.get("calibrate")))
    if config.calibrate is not None and not scoring.region:
        raise ValueError("%s: harden.calibrate is region scoring's; this run "
                         "scores by %s" % (path, scoring.name))
    if config.max_evaluations < 0:
        raise ValueError("%s.max_evaluations must be nonnegative" % BLOCK)
    if config.workers < 1:
        raise ValueError("%s.workers must be at least one" % BLOCK)
    return config


__all__ = ["BLOCK", "Component", "HardenConfig", "PENDING", "REQUIRED",
           "Reader", "flag", "number", "read_harden", "read_options"]
