from __future__ import annotations

import dataclasses
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..solve.bank import Bank, load_bank
from ..solve.config import FamilyConfig, Sampling, read_config
from ..solve.models import MODELS, build_model
from ..solve.record import Record, parse_record
from ..solve.verdict import CHARGED, Verdict
from .scoring import (REGION_MODELS, Regions, Scoring, flow_weights,
                      reach_weights, region_contributions, region_reference)

VERDICTS: Tuple[str, ...] = tuple(v.name for v in Verdict)
WEIGHTED = ("region",)
_CHARGED = np.array([int(v) for v in CHARGED], dtype=np.int8)


@dataclass(frozen=True)
class ModelReading:
    model: str
    fractions: Mapping[str, float] = field(default_factory=dict)
    score: float = 0.0
    bank_score: float = 0.0
    scoring: str = "bank"
    samples: int = 0
    bank: str = ""
    fingerprint: str = ""
    focus: str = ""
    seconds: float = 0.0
    error: str = ""
    codes: bytes = field(default=b"", repr=False, compare=False)
    reach_index: bytes = field(default=b"", repr=False, compare=False)
    reach_values: bytes = field(default=b"", repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return not self.error

    def verdicts(self) -> np.ndarray:
        return np.frombuffer(self.codes, dtype=np.int8)

    def reach(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.scoring not in WEIGHTED:
            return None
        return (np.frombuffer(self.reach_index, dtype=np.int32),
                np.frombuffer(self.reach_values, dtype=np.float64))

    def to_dict(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        if self.scoring in WEIGHTED:
            _, values = self.reach()
            extra = {"reach_states": int(values.size)}
        return {"model": self.model,
                "fractions": {k: round(v, 8) for k, v in
                              self.fractions.items()},
                "score": round(self.score, 8),
                "bank_score": round(self.bank_score, 8),
                "scoring": self.scoring, **extra, "samples": self.samples,
                "bank": self.bank, "fingerprint": self.fingerprint,
                "focus": self.focus, "seconds": round(self.seconds, 3),
                "error": self.error}


@dataclass(frozen=True)
class WeaknessReading:
    models: Mapping[str, ModelReading] = field(default_factory=dict)
    seconds: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def ok(self) -> bool:
        return bool(self.models) and all(r.ok for r in self.models.values())

    def score(self, model: str) -> Optional[float]:
        reading = self.models.get(model)
        return None if reading is None or not reading.ok else reading.score

    def errors(self) -> Dict[str, str]:
        return {name: r.error for name, r in self.models.items() if r.error}

    def to_dict(self) -> Dict[str, Any]:
        return {"seconds": round(self.seconds, 3),
                "started_at": (None if self.started_at is None
                               else round(self.started_at, 3)),
                "finished_at": (None if self.finished_at is None
                                else round(self.finished_at, 3)),
                "models": {k: v.to_dict() for k, v in self.models.items()}}


def transitions(before: ModelReading, after: ModelReading) -> Dict[str, int]:
    if before.fingerprint != after.fingerprint:
        raise ValueError("%s was read on sampling %s and %s, which are not the "
                         "same states" % (before.model, before.fingerprint,
                                          after.fingerprint))
    was = np.isin(before.verdicts(), _CHARGED)
    now = np.isin(after.verdicts(), _CHARGED)
    return {"repaired": int(np.sum(was & ~now)),
            "broken": int(np.sum(~was & now)),
            "still": int(np.sum(was & now))}


@dataclass(frozen=True)
class Objective:
    weights: Mapping[str, float]

    def __post_init__(self):
        if not self.weights:
            raise ValueError("an objective needs at least one model")
        for name, weight in self.weights.items():
            if name not in MODELS:
                raise ValueError("no model %r (have: %s)"
                                 % (name, ", ".join(MODELS)))
            if not weight >= 0.0:
                raise ValueError("the weight of %s must be nonnegative" % name)

    @property
    def models(self) -> Tuple[str, ...]:
        return tuple(self.weights)

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.weights.items()}


@dataclass(frozen=True)
class WeaknessScore:
    scores: Mapping[str, float] = field(default_factory=dict)
    weights: Mapping[str, float] = field(default_factory=dict)
    complete: bool = False

    @property
    def total(self) -> float:
        if not self.complete:
            return float("inf")
        return sum(self.weights[name] * value
                   for name, value in self.scores.items())

    def of(self, model: str) -> Optional[float]:
        return self.scores.get(model)

    def to_dict(self) -> Dict[str, Any]:
        return {"scores": {k: round(v, 8) for k, v in self.scores.items()},
                "weights": dict(self.weights), "complete": self.complete,
                "total": None if not self.complete else round(self.total, 8)}


def score_of(reading: WeaknessReading, objective: Objective
             ) -> WeaknessScore:
    scores: Dict[str, float] = {}
    complete = True
    for name in objective.models:
        value = reading.score(name)
        if value is None:
            complete = False
            continue
        scores[name] = value
    return WeaknessScore(scores, dict(objective.weights), complete)


def deltas(child: WeaknessScore, parent: WeaknessScore) -> Dict[str, float]:
    out = {name: child.scores[name] - parent.scores[name]
           for name in child.scores if name in parent.scores}
    if child.complete and parent.complete:
        out["total"] = child.total - parent.total
    return out


def measure_record(record: Record, config: FamilyConfig,
                   banks: Mapping[str, Bank],
                   regions: Optional[Regions] = None) -> WeaknessReading:
    started = time.time()
    out: Dict[str, ModelReading] = {}
    flowing = None
    try:
        if (regions is not None and regions.scoring.flow_reach
                and regions.references):
            flowing = flow_weights(record, config, banks, regions)
    except (ValueError, ArithmeticError, TypeError, AssertionError,
            KeyError) as error:
        now = time.time()
        return WeaknessReading(
            {name: ModelReading(model=name, error="%s: %s" % (
                type(error).__name__, error)) for name in banks},
            now - started, started, now)
    for name, bank in banks.items():
        began = time.perf_counter()
        meta = bank.meta
        common = dict(model=name, samples=int(bank.state.size),
                      bank=str(bank.path),
                      fingerprint=str(meta.get("fingerprint", "")),
                      focus=str(meta["description"]["focus"]))
        try:
            model = build_model(name, config, record)
            verdict = np.asarray(model.evaluate(bank.state).verdict,
                                 dtype=np.int8)
            weighted = None
            if regions is not None and name in regions.spaces:
                ref = regions.references.get(name)
                if ref is None:
                    weights = regions.weights(name)
                elif flowing is not None:
                    weights = flowing[name]
                else:
                    weights = reach_weights(record, config, bank,
                                            regions.spaces[name],
                                            regions.scoring, ref.starts)
                weighted = region_contributions(verdict, weights)
        except (ValueError, ArithmeticError, TypeError, AssertionError,
                KeyError) as error:
            out[name] = ModelReading(error="%s: %s" % (type(error).__name__,
                                                       error),
                                     seconds=time.perf_counter() - began,
                                     **common)
            continue
        out[name] = model_reading(verdict, weighted, "region",
                                  seconds=time.perf_counter() - began,
                                  **common)
    finished = time.time()
    return WeaknessReading(out, finished - started, started, finished)


def model_reading(verdict: np.ndarray,
                  weighted: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                  scoring: str = "region", **common: Any) -> ModelReading:
    verdict = np.asarray(verdict, dtype=np.int8)
    n = max(1, verdict.size)
    fractions = {v.name: float(np.sum(verdict == int(v))) / n
                 for v in Verdict}
    share = float(np.sum(np.isin(verdict, _CHARGED))) / n
    if weighted is None:
        return ModelReading(fractions=fractions, score=share,
                            bank_score=share, codes=verdict.tobytes(),
                            **common)
    index = np.asarray(weighted[0], dtype=np.int32)
    values = np.asarray(weighted[1], dtype=np.float64)
    return ModelReading(fractions=fractions, score=float(weighted[2]),
                        bank_score=share, scoring=scoring,
                        codes=verdict.tobytes(),
                        reach_index=index.tobytes(),
                        reach_values=values.tobytes(), **common)


def _parse(document: Mapping[str, Any], label: str,
           config: FamilyConfig) -> Record:
    return parse_record(document, Path("%s.json" % label),
                        config.physics.record_params)


_WORKER: Dict[str, Any] = {}


def _worker_init(config_path: str, sampling: Sampling,
                 banks: Mapping[str, str],
                 regions: Optional[Regions] = None) -> None:
    config = read_config(config_path)
    _WORKER["config"] = dataclasses.replace(config, sampling=sampling)
    _WORKER["banks"] = {name: load_bank(Path(path))
                        for name, path in banks.items()}
    _WORKER["regions"] = regions


def _worker_measure(item: Tuple[int, str, Mapping[str, Any]]
                    ) -> Tuple[int, WeaknessReading]:
    index, label, document = item
    config = _WORKER["config"]
    try:
        record = _parse(document, label, config)
    except (ValueError, KeyError, TypeError) as error:
        return index, _failed(_WORKER["banks"], "the record does not read: "
                              "%s" % error)
    return index, measure_record(record, config, _WORKER["banks"],
                                 _WORKER.get("regions"))


def _failed(banks: Mapping[str, Bank], error: str) -> WeaknessReading:
    now = time.time()
    return WeaknessReading({name: ModelReading(model=name, error=error)
                            for name in banks}, 0.0, now, now)


class Evaluator:
    def __init__(self, config: FamilyConfig, objective: Objective,
                 workers: int = 1, scoring: Optional[Scoring] = None,
                 original: Optional[Record] = None):
        self.config = config
        self.objective = objective
        self.workers = max(1, int(workers))
        self.scoring = scoring or Scoring("bank")
        self.banks: Dict[str, Bank] = {}
        self.spaces: Dict[str, Any] = {}
        for name in objective.models:
            config.model(name)
            model = build_model(name, config)
            if self.scoring.region and name in REGION_MODELS:
                from .space import space_bank
                self.banks[name], self.spaces[name] = space_bank(
                    model, self.scoring.space_settings(),
                    workers=self.workers)
            else:
                self.banks[name] = model.bank()
        self.regions: Optional[Regions] = (
            Regions(self.scoring, self.spaces) if self.scoring.region
            else None)
        if self.scoring.region_reach:
            if original is None:
                raise ValueError("reach-aware region scoring needs the "
                                 "original record to set its scale")
            self.regions.references = region_reference(
                original, config, self.banks, self.regions)
        self._pool: Optional[ProcessPoolExecutor] = None

    @property
    def state_weights(self) -> Optional[Mapping[str, np.ndarray]]:
        if self.regions is None:
            return None
        return {name: self.regions.weights(name) for name in self.spaces}

    def describe(self) -> Dict[str, Any]:
        return {name: {"path": str(bank.path),
                       "fingerprint": bank.meta.get("fingerprint"),
                       "samples": int(bank.state.size),
                       "focus": bank.meta["description"]["focus"],
                       "seed": bank.meta["description"]["seed"],
                       "contested_share": bank.meta["sampler"].get(
                           "contested_share"),
                       "scoring": self.scoring.to_dict(),
                       **({"region_reach": self.regions.references[
                           name].to_dict()}
                          if self.regions is not None
                          and name in self.regions.references else {}),
                       **({"space": {
                           "cells": self.spaces[name].cells,
                           "features": list(self.spaces[name].names),
                           "least_count": int(
                               self.spaces[name].count.min()),
                           "draws": bank.meta["sampler"].get("draws")}}
                          if name in self.spaces else {})}
                for name, bank in self.banks.items()}

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers, initializer=_worker_init,
                initargs=(str(self.config.path), self.config.sampling,
                          {name: str(bank.path)
                           for name, bank in self.banks.items()},
                          self.regions))
        return self._pool

    def measure(self, items: Sequence[Tuple[str, Mapping[str, Any]]],
                records: Optional[Sequence[Optional[Record]]] = None
                ) -> List[WeaknessReading]:
        if not items:
            return []
        if self.workers == 1 or len(items) == 1:
            out = []
            for i, (label, document) in enumerate(items):
                record = records[i] if records else None
                if record is None:
                    try:
                        record = _parse(document, label, self.config)
                    except (ValueError, KeyError, TypeError) as error:
                        out.append(_failed(self.banks, "the record does not "
                                           "read: %s" % error))
                        continue
                out.append(measure_record(record, self.config, self.banks,
                                          self.regions))
            return out
        results: List[Optional[WeaknessReading]] = [None] * len(items)
        tasks = [(i, label, dict(document))
                 for i, (label, document) in enumerate(items)]
        for index, reading in self._executor().map(_worker_measure, tasks):
            results[index] = reading
        return [r for r in results if r is not None]

    def map(self, function: Any, tasks: Sequence[Any]) -> List[Any]:
        return list(self._executor().map(function, tasks))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


__all__ = ["Evaluator", "ModelReading", "Objective", "VERDICTS", "WEIGHTED",
           "WeaknessReading", "WeaknessScore", "deltas", "measure_record",
           "model_reading", "score_of", "transitions"]
