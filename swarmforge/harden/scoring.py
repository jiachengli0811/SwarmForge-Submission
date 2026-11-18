from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ..solve.bank import Bank
from ..solve.effort import Sources, effort_from, plant_step
from ..solve.models import build_model
from ..solve.record import Record
from ..solve.verdict import CHARGED

_CHARGED = np.array([int(v) for v in CHARGED], dtype=np.int8)

REGION_MODELS = ("obstacle_collision", "agent_collision", "timeout")

_LO, _HI = 1e-3, 128.0


@dataclass(frozen=True)
class Scoring:
    name: str = "bank"
    shares: Tuple[float, ...] = ()
    candidates: int = 0
    coarse: int = 0
    segments: int = 0
    missions: Tuple[str, ...] = ()
    cells: int = 0
    pool: int = 0
    draws: int = 0
    tilt: float = 0.0
    regime: float = 0.0
    per_cell: int = 0
    flow: int = 0
    blend: float = 0.0
    effort: str = "response"

    @property
    def region(self) -> bool:
        return self.name == "region"

    @property
    def flow_reach(self) -> bool:
        return self.region_reach and (bool(self.flow)
                                      or self.effort != "response")

    @property
    def region_reach(self) -> bool:
        return self.region and bool(self.shares)

    def space_settings(self):
        from .space import SpaceSettings
        return SpaceSettings(self.cells, self.pool, self.draws, self.regime)

    def to_dict(self) -> Dict[str, Any]:
        if not self.region:
            return {"name": "bank"}
        out = {"name": "region", "cells": self.cells, "pool": self.pool,
               "draws": self.draws, "tilt": self.tilt}
        if self.regime:
            out["regime"] = self.regime
        if self.flow_reach:
            out["reach"] = dict(
                {"flow": self.flow} if self.effort == "response"
                else {"effort": self.effort},
                shares=list(self.shares), per_cell=self.per_cell,
                candidates=self.candidates)
            if self.blend:
                out["reach"]["blend"] = self.blend
        elif self.region_reach:
            out["reach"] = {"shares": list(self.shares),
                            "per_cell": self.per_cell,
                            "candidates": self.candidates,
                            "coarse": self.coarse,
                            "segments": self.segments}
        if self.region_reach and self.missions:
            out["missions"] = list(self.missions)
        return out


def read_scoring(raw: Any, base: Optional[Path] = None) -> Scoring:
    from .config import number, read_options
    if raw is None or raw == "bank":
        return Scoring("bank")
    if not isinstance(raw, Mapping):
        raise ValueError("harden.scoring must be bank, or a mapping with a "
                         "name")
    name = str(raw.get("name") or "")
    given = {str(k): v for k, v in raw.items() if k != "name"}
    if name == "bank":
        if given:
            raise ValueError("harden.scoring (bank) reads no options, not "
                             "%s" % ", ".join(sorted(given)))
        return Scoring("bank")
    if name != "region":
        raise ValueError("harden.scoring: no scoring %r (have: bank, region)"
                         % name)
    reach = given.pop("reach", None)
    missions = mission_files(given.pop("missions", None), base)
    regime = given.pop("regime", None)
    options = read_options("harden.scoring (region)", given, {
        "cells": number(int, 2),
        "pool": number(int, 1),
        "draws": number(int, 1),
        "tilt": number(float, 0.0, 1.0)})
    if options["draws"] < options["pool"]:
        raise ValueError("harden.scoring (region): draws must be at "
                         "least one pool")
    if regime is not None:
        options.update(read_options("harden.scoring (region)",
                                    {"regime": regime},
                                    {"regime": number(float, 0.0, 1.0)}))
    if reach is not None:
        if not isinstance(reach, Mapping):
            raise ValueError("harden.scoring (region): reach must be a "
                             "mapping of its options")
        flows = "flow" in reach or "effort" in reach
        readers = (dict({"effort": _effort} if "effort" in reach
                        else {"flow": number(int, 1)},
                        shares=_shares, per_cell=number(int, 1),
                        candidates=number(int, 1),
                        **({"blend": number(float, 0.0, 1.0)}
                           if "blend" in reach else {}))
                   if flows else
                   {"shares": _shares,
                    "per_cell": number(int, 1),
                    "candidates": number(int, 1),
                    "coarse": number(int, 1),
                    "segments": number(int, 1)})
        options.update(read_options("harden.scoring (region).reach", reach,
                                    readers))
    if missions and reach is None:
        raise ValueError("harden.scoring (region): missions are the "
                         "reach-aware region scoring's: give its reach")
    if (options.get("flow") or options.get("effort")) and len(
            options["shares"]) != 1:
        raise ValueError("harden.scoring (region).reach: a flow reach "
                         "reads one share, not %d" % len(options["shares"]))
    return Scoring("region", missions=missions, **options)


def mission_files(value: Any, base: Optional[Path]) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Path)):
        value = [value]
    if not isinstance(value, (list, tuple)) or not all(
            isinstance(v, (str, Path)) and str(v) for v in value):
        raise ValueError("harden.scoring: missions must be a list of case "
                         "files, not %r" % (value,))
    out = []
    for v in value:
        path = Path(v).expanduser()
        if not path.is_absolute():
            path = (base or Path.cwd()) / path
        if not path.is_file():
            raise ValueError("harden.scoring: no mission file %s" % path)
        out.append(str(path.resolve()))
    return tuple(out)


def _effort(value: Any) -> str:
    from ..mission.flow import EFFORTS, RESPONSE
    if value == RESPONSE or value not in EFFORTS:
        raise ValueError("not a baseline")
    return value


_effort.rule = "distance or direction (the laws' response is flow: K)"


def _shares(value: Any) -> Tuple[float, ...]:
    from .config import number
    if isinstance(value, (int, float)):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("not a list of shares")
    out = tuple(number(float, 0.0, 1.0, above=True)(v) for v in value)
    if any(b <= a for a, b in zip(out, out[1:])):
        raise ValueError("shares must increase")
    return out


def temper(e: np.ndarray, step: float, share: float) -> float:
    e = e[np.isfinite(e)]
    if e.size == 0:
        return _HI * step
    e = e - e.min()
    want = share * e.size

    def ess(c):
        w = np.exp(-e / (c * step) ** 2)
        return w.sum() ** 2 / (w * w).sum()

    lo, hi = _LO, _HI
    if ess(lo) >= want:
        return lo * step
    if ess(hi) < want:
        return hi * step
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        lo, hi = (lo, mid) if ess(mid) >= want else (mid, hi)
    return float(hi * step)


def _charged(verdict) -> np.ndarray:
    return np.isin(np.asarray(verdict, dtype=np.int8), _CHARGED)


@lru_cache(maxsize=4)
def _read_missions(paths: Tuple[str, ...]):
    from ..mission.sources import read_missions
    return read_missions(paths)


def _efforts(record: Record, config, bank: Bank, index: np.ndarray,
             scoring: Scoring, sources: Sources) -> np.ndarray:
    return effort_from(record, config, bank.state.take(index), sources,
                       candidates=scoring.candidates, coarse=scoring.coarse,
                       segments=scoring.segments)


def region_contributions(verdict: np.ndarray, weights: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, float]:
    charged = np.flatnonzero(_charged(verdict))
    values = np.asarray(weights, dtype=np.float64)[charged]
    return charged.astype(np.int32), values, float(values.sum())


@dataclass(eq=False)
class RegionReference:
    starts: Sources
    weights: np.ndarray
    shares: Tuple[float, ...]
    effective: Tuple[float, ...]
    decided: Tuple[Tuple[int, str], ...]
    original_score: float
    original_region: float
    tau: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"shares": list(self.shares),
                "effective_cells": [round(e, 3) for e in self.effective],
                "decided": [list(d) for d in self.decided],
                "sources": self.starts.size,
                "original_score": self.original_score,
                "original_region_score": self.original_region,
                **({"tau": self.tau} if self.tau else {})}


@dataclass
class Regions:
    scoring: Scoring
    spaces: Mapping[str, Any]
    references: Mapping[str, RegionReference] = field(default_factory=dict)

    def weights(self, name: str) -> np.ndarray:
        if name in self.references:
            return self.references[name].weights
        return self.spaces[name].state_weights(self.scoring.tilt)


def cell_sample(space, per_cell: int) -> np.ndarray:
    from .space import cell_rank
    return np.flatnonzero(cell_rank(space.cell) < per_cell)


def cell_efforts(record: Record, config, bank: Bank, space,
                 scoring: Scoring, sources: Sources) -> np.ndarray:
    index = cell_sample(space, scoring.per_cell)
    e = _efforts(record, config, bank, index, scoring, sources)
    out = np.full(space.cells, np.inf)
    np.minimum.at(out, space.cell[index], np.where(np.isfinite(e), e, np.inf))
    return out


def cell_levels(efforts: np.ndarray, step: float,
                shares: Tuple[float, ...]) -> np.ndarray:
    finite = np.isfinite(efforts)
    out = np.ones((len(shares), efforts.size))
    for k, share in enumerate(shares):
        if share >= 1.0 or not finite.any():
            continue
        sigma = temper(efforts, step, share)
        out[k] = np.where(finite, np.exp(-(efforts - efforts[finite].min())
                                         / sigma ** 2), 0.0)
    return out


def _region_levels(charged: np.ndarray, space, tilt: float,
                   levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = (np.bincount(space.cell, weights=charged.astype(float),
                     minlength=space.cells) / np.maximum(space.count, 1))
    base = space.cell_weights(tilt)
    mu = base[None, :] * levels
    total = mu.sum(axis=1, keepdims=True)
    mu = np.where(total > 0, mu / np.maximum(total, 1e-300), base[None, :])
    return mu @ v, mu


def reach_weights(record: Record, config, bank: Bank, space,
                  scoring: Scoring, sources: Sources) -> np.ndarray:
    e = cell_efforts(record, config, bank, space, scoring, sources)
    levels = cell_levels(e, plant_step(config), scoring.shares)
    base = space.cell_weights(scoring.tilt)
    mu = base[None, :] * levels
    total = mu.sum(axis=1, keepdims=True)
    mu = np.where(total > 0, mu / np.maximum(total, 1e-300), base[None, :])
    return (mu.mean(axis=0) / np.maximum(space.count, 1))[space.cell]


def region_reference(record: Record, config, banks: Mapping[str, Bank],
                     regions: "Regions") -> Dict[str, RegionReference]:
    from ..mission.sources import failure_sources
    scoring = regions.scoring
    if not scoring.missions:
        raise ValueError("reach-aware region scoring reads %s: give missions"
                         % ("the representative missions its flow reach "
                            "flies" if scoring.flow_reach
                            else "the original's failures on given missions"))
    if scoring.flow_reach:
        return _flow_reference(record, config, banks, regions)
    step = plant_step(config)
    sources, decided = failure_sources(
        config, record, _read_missions(scoring.missions),
        {n: build_model(n, config) for n in regions.spaces})
    out = {}
    for name, space in regions.spaces.items():
        levels = cell_levels(cell_efforts(record, config, banks[name], space,
                                          scoring, sources[name]),
                             step, scoring.shares)
        effective = tuple(float(w.sum() ** 2 / max(np.sum(w * w), 1e-300))
                          for w in levels)
        weights = reach_weights(record, config, banks[name], space, scoring,
                                sources[name])
        charged = _charged(build_model(name, config, record)
                           .evaluate(banks[name].state).verdict)
        plain = space.state_weights(scoring.tilt)
        out[name] = RegionReference(
            sources[name], weights, tuple(scoring.shares), effective,
            tuple(decided), float(weights[charged].sum()),
            float(plain[charged].sum()))
    return out


def flow_temperature(e: np.ndarray, base: np.ndarray, share: float) -> float:
    e = np.where(np.isfinite(e), e, np.inf)
    full = base.sum() ** 2 / np.sum(base * base)
    want = share * e.size

    def ess(t):
        w = base * np.exp(-e / t)
        return w.sum() ** 2 / max(np.sum(w * w), 1e-300) / full * e.size
    lo, hi = 1e-12, 1e12
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if ess(mid) < want else (lo, mid)
    return hi


def _flow_base(space, index: np.ndarray, tilt: float) -> np.ndarray:
    cell = space.cell[index]
    count = np.bincount(cell, minlength=space.cells)
    return (space.cell_weights(tilt) / np.maximum(count, 1))[cell]


def flow_efforts_of(record: Record, config, banks: Mapping[str, Bank],
                    regions: "Regions"):
    from ..mission.flow import DIRECTION, RESPONSE, Flow, flow_efforts
    from ..solve.effort import pin_kind
    scoring = regions.scoring
    flow = Flow(record, config, _read_missions(scoring.missions),
                scoring.flow, scoring.effort)
    windows = {}
    out = {}
    for name, space in regions.spaces.items():
        model = build_model(name, config)
        key = (pin_kind(model), int(model.spec.drones))
        if key not in windows:
            src, at = flow.windows(*key)
            windows[key] = (
                src, flow.gramians(key[0], src, at)
                if scoring.effort == RESPONSE else None,
                flow.updates(src, at) if scoring.effort == DIRECTION
                else None)
        src, W, update = windows[key]
        index = cell_sample(space, scoring.per_cell)
        out[name] = (index, flow_efforts(config, banks[name].state.take(index),
                                         src, W, update=update,
                                         candidates=scoring.candidates))
    return out, tuple(flow.decided)


def flow_weights(record: Record, config, banks: Mapping[str, Bank],
                 regions: "Regions") -> Dict[str, np.ndarray]:
    efforts, _ = flow_efforts_of(record, config, banks, regions)
    out = {}
    for name, (index, e) in efforts.items():
        space = regions.spaces[name]
        tau = regions.references[name].tau
        w = _flow_base(space, index, regions.scoring.tilt) * np.exp(
            -(np.where(np.isfinite(e), e, np.inf) - np.min(e)) / tau)
        out[name] = _blended(space, index, w / w.sum(), regions.scoring)
    return out


def _blended(space, index: np.ndarray, w: np.ndarray, scoring: Scoring
             ) -> np.ndarray:
    full = np.zeros(space.cell.size)
    full[index] = (1.0 - scoring.blend) * w
    if scoring.blend:
        full += scoring.blend * space.state_weights(scoring.tilt)
    return full


def _flow_reference(record: Record, config, banks: Mapping[str, Bank],
                    regions: "Regions") -> Dict[str, RegionReference]:
    from ..solve.effort import pin_kind
    scoring = regions.scoring
    efforts, decided = flow_efforts_of(record, config, banks, regions)
    out = {}
    for name, (index, e) in efforts.items():
        space = regions.spaces[name]
        base = _flow_base(space, index, scoring.tilt)
        tau = flow_temperature(e, base, scoring.shares[0])
        w = base * np.exp(-(np.where(np.isfinite(e), e, np.inf) - np.min(e))
                          / tau)
        weights = _blended(space, index, w / w.sum(), scoring)
        charged = _charged(build_model(name, config, record)
                           .evaluate(banks[name].state).verdict)
        plain = space.state_weights(scoring.tilt)
        effective = (float(w.sum() ** 2 / np.sum(w * w)),)
        out[name] = RegionReference(
            Sources(pin_kind(build_model(name, config)), np.zeros((0, 1, 2)),
                    np.zeros((0, 1, 2)),
                    np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2)),
                    np.zeros(0, int), np.zeros(0, int)),
            weights, tuple(scoring.shares), effective, decided,
            float(weights[charged].sum()), float(plain[charged].sum()), tau)
    return out


def paired_variance(a_index: np.ndarray, a_values: np.ndarray,
                    b_index: np.ndarray, b_values: np.ndarray) -> float:
    union = np.union1d(a_index, b_index)
    va = np.zeros(union.size)
    vb = np.zeros(union.size)
    va[np.searchsorted(union, a_index)] = a_values
    vb[np.searchsorted(union, b_index)] = b_values
    return float(np.sum((va - vb) ** 2))


__all__ = ["REGION_MODELS", "RegionReference", "Regions", "Scoring",
           "cell_efforts", "cell_levels", "cell_sample", "flow_efforts_of",
           "flow_temperature", "flow_weights", "mission_files",
           "paired_variance", "reach_weights", "read_scoring",
           "region_contributions", "region_reference", "temper"]
