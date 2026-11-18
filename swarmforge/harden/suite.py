from __future__ import annotations

import collections
import csv
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Mapping,
                    Optional, Sequence, Set, Tuple)

from ..solve.config import FamilyConfig, read_config
from ..solve.record import Record, parse_record, release_sympy_caches

if TYPE_CHECKING:
    from ..mission.campaign import Case


CHUNK = 5
PARSED = 16
SUITE_DIR = "suite"


@dataclass(frozen=True)
class Flown:
    index: int
    verdict: str
    tick: Optional[int] = None
    crash_kind: str = ""
    seconds: float = 0.0
    error: str = ""

    @property
    def failed(self) -> bool:
        return self.verdict != "success"

    @property
    def kind(self) -> str:
        if not self.failed:
            return ""
        return self.crash_kind if self.verdict == "crash" else self.verdict

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "verdict": self.verdict,
                "tick": self.tick, "crash_kind": self.crash_kind,
                "seconds": round(self.seconds, 3), "error": self.error}


KINDS = ("agent", "obstacle", "both", "timeout", "error")


@dataclass
class SuiteReading:
    cases: int
    flights: Dict[int, Flown] = field(default_factory=dict)
    raced: bool = False
    seconds: float = 0.0

    @property
    def complete(self) -> bool:
        return len(self.flights) == self.cases

    @property
    def failures(self) -> int:
        return sum(1 for f in self.flights.values() if f.failed)

    def failed_cases(self) -> Set[int]:
        return {i for i, f in self.flights.items() if f.failed}

    def kinds(self) -> Dict[str, int]:
        counts = collections.Counter(f.kind for f in self.flights.values()
                                     if f.failed)
        return {kind: counts.get(kind, 0) for kind in KINDS}

    def flight_seconds(self) -> float:
        return sum(f.seconds for f in self.flights.values())

    def to_dict(self) -> Dict[str, Any]:
        return {"cases": self.cases, "flown": len(self.flights),
                "failures": self.failures, "complete": self.complete,
                "raced": self.raced, "kinds": self.kinds(),
                "flight_seconds": round(self.flight_seconds(), 3),
                "seconds": round(self.seconds, 3)}


def flips(parent: SuiteReading, child: SuiteReading) -> Dict[str, int]:
    both = set(parent.flights) & set(child.flights)
    was, now = parent.failed_cases() & both, child.failed_cases() & both
    return {"repaired": len(was - now), "broken": len(now - was),
            "still": len(was & now)}


_WORKER: Dict[str, Any] = {}


def _state(config_path: str, cases: Sequence[Case]) -> Dict[str, Any]:
    from ..mission.verdict import Oracle

    config = read_config(config_path)
    return dict(config=config, oracle=Oracle.for_config(config),
                cases={case.meta.index: case for case in cases},
                records=collections.OrderedDict())


def _start(config_path: str, cases: Sequence[Case]) -> None:
    _WORKER.update(_state(config_path, cases))


def _record(state: Dict[str, Any], key: str,
            document: Mapping[str, Any]) -> Record:
    records = state["records"]
    found = records.get(key)
    if found is None:
        config = state["config"]
        try:
            found = parse_record(document, Path("%s.json" % key),
                                 config.physics.record_params)
        finally:
            release_sympy_caches()
        records[key] = found
        while len(records) > PARSED:
            records.popitem(last=False)
    else:
        records.move_to_end(key)
    return found


def _fly(key: str, document: Mapping[str, Any], indices: Sequence[int],
         state: Optional[Dict[str, Any]] = None) -> List[Flown]:
    from ..mission.campaign import fly_case

    state = _WORKER if state is None else state
    try:
        record = _record(state, key, document)
    except Exception as error:
        return [Flown(i, "error", error="the record does not read: %s: %s"
                      % (type(error).__name__, error)) for i in indices]
    out = []
    for index in indices:
        began = time.time()
        try:
            result = fly_case(state["config"], record, state["cases"][index],
                              name=key, oracle=state["oracle"])
        except Exception as error:
            out.append(Flown(index, "error", seconds=time.time() - began,
                             error="%s: %s" % (type(error).__name__, error)))
            continue
        out.append(Flown(index, result.verdict, result.verdict_tick,
                         result.crash_kind or "", float(result.seconds),
                         result.error or ""))
    return out


def _task(key: str, document: Mapping[str, Any],
          indices: Sequence[int]) -> Tuple[List[Flown], float]:
    began = time.process_time()
    flights = _fly(key, document, indices)
    return flights, time.process_time() - began


class Flyer:
    def __init__(self, config_path, cases: Sequence[Case], workers: int = 1):
        self.config_path = str(config_path)
        self.cases = tuple(cases)
        self.indices = tuple(case.meta.index for case in self.cases)
        self.workers = max(1, int(workers))
        self._pool: Optional[ProcessPoolExecutor] = None
        self.cpu = 0.0
        self._state = (_state(self.config_path, self.cases)
                       if self.workers == 1 else None)

    def __enter__(self) -> "Flyer":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers, initializer=_start,
                initargs=(self.config_path, self.cases))
        return self._pool

    def fly(self, controllers: Sequence[Tuple[str, Mapping[str, Any]]], *,
            indices: Optional[Sequence[int]] = None,
            cuts: Optional[Mapping[str, Optional[int]]] = None,
            order: Optional[Mapping[str, Sequence[int]]] = None
            ) -> Dict[str, SuiteReading]:
        indices = tuple(self.indices if indices is None else indices)
        cuts = dict(cuts or {})
        readings = {key: SuiteReading(len(indices)) for key, _ in controllers}
        documents = dict(controllers)
        chunks: Dict[str, List[Tuple[int, ...]]] = {}
        for key in documents:
            first = [i for i in (order or {}).get(key, ()) if i in indices]
            seen = set(first)
            ordered = first + [i for i in indices if i not in seen]
            chunks[key] = [tuple(ordered[k:k + CHUNK])
                           for k in range(0, len(ordered), CHUNK)]
        tasks = [(key, chunk) for k in range(max(map(len, chunks.values()),
                                                 default=0))
                 for key in documents if k < len(chunks[key])
                 for chunk in (chunks[key][k],)]
        started = {key: None for key in documents}

        def take(key: str, flown: Iterable[Flown]) -> bool:
            reading = readings[key]
            for flight in flown:
                reading.flights[flight.index] = flight
            cut = cuts.get(key)
            if cut is not None and reading.failures > cut:
                reading.raced = True
            return reading.raced

        now = time.time
        if self.workers == 1:
            for key, chunk in tasks:
                if readings[key].raced:
                    continue
                began = now()
                started[key] = started[key] or began
                take(key, _fly(key, documents[key], chunk, self._state))
                readings[key].seconds = now() - started[key]
            return readings
        pool = self._executor()
        pending = {}
        for key, chunk in tasks:
            started[key] = started[key] or now()
            pending[pool.submit(_task, key, documents[key], chunk)] = key
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for future in done:
                key = pending.pop(future)
                if future.cancelled():
                    continue
                flights, cpu = future.result()
                self.cpu += cpu
                if take(key, flights):
                    for other, owner in list(pending.items()):
                        if owner == key and other.cancel():
                            pending.pop(other)
                readings[key].seconds = now() - started[key]
        return readings


@dataclass
class Suite:
    seed: int
    cases: Tuple[Case, ...]
    original: SuiteReading
    flown: int
    campaign: Dict[str, int] = field(default_factory=dict)
    cpu: float = 0.0
    bands: Optional[Tuple[int, ...]] = None

    @property
    def indices(self) -> Tuple[int, ...]:
        return tuple(case.meta.index for case in self.cases)

    def to_dict(self) -> Dict[str, Any]:
        bands = collections.Counter(case.meta.band for case in self.cases)
        return {"seed": self.seed, "cases": len(self.cases),
                "flown": self.flown, "campaign": dict(self.campaign),
                "kinds": self.original.kinds(),
                "bands": {str(b): n for b, n in sorted(bands.items())},
                "from_bands": (None if self.bands is None
                               else list(self.bands)),
                "indices": list(self.indices)}


def collect(config_path, document: Mapping[str, Any], *, failures: int,
            seed: int, missions: int, workers: int = 1,
            bands: Optional[Sequence[int]] = None,
            out_dir=None, progress: bool = False) -> Suite:
    from ..mission.campaign import random_cases

    config = read_config(config_path)
    plan = random_cases(config, int(missions), int(seed))
    if bands is not None:
        keep = {int(b) for b in bands}
        plan = [case for case in plan if case.meta.band in keep]
    step = int(workers) * CHUNK
    flown: Dict[int, Flown] = {}
    cpu = 0.0
    started = time.time()
    with Flyer(config_path, plan, workers) as flyer:
        start = 0
        while start < len(plan):
            failed = sum(1 for f in flown.values() if f.failed)
            if failed >= failures:
                break
            rate = failed / len(flown) if flown else 1.0
            need = int((failures - failed) / max(rate, 0.05) * 1.1) + 1
            chunk = plan[start:start + max(step, need)]
            reading = flyer.fly([("original", dict(document))],
                                indices=[c.meta.index for c in chunk])
            flown.update(reading["original"].flights)
            cpu = flyer.cpu
            start += len(chunk)
            if progress:
                print("  suite: flew %d missions, %d failed (%.0fs)"
                      % (len(flown), sum(1 for f in flown.values()
                                         if f.failed), time.time() - started),
                      flush=True)
    failing = sorted(i for i, f in flown.items() if f.failed)
    if len(failing) < failures:
        raise ValueError("the original failed only %d of %d missions of "
                         "campaign %d%s: a suite of %d needs a larger cap "
                         "(suite_missions)"
                         % (len(failing), len(flown), seed,
                            "" if bands is None else " in bands %s"
                            % sorted(bands), failures))
    chosen = failing[:failures]
    last = chosen[-1]
    within = [f for i, f in flown.items() if i <= last]
    by_index = {case.meta.index: case for case in plan}
    suite = Suite(
        seed=int(seed), cases=tuple(by_index[i] for i in chosen),
        original=SuiteReading(len(chosen), {i: flown[i] for i in chosen}),
        flown=len(within),
        campaign=dict(collections.Counter(f.kind or "success"
                                          for f in within)), cpu=cpu,
        bands=None if bands is None else tuple(sorted(int(b) for b in bands)))
    if out_dir is not None:
        write(out_dir, suite, config,
              [case for case in plan if case.meta.index <= last], flown)
    return suite


def write(out_dir, suite: Suite, config: FamilyConfig,
          campaign: Sequence[Case], flown: Mapping[int, Flown]) -> Path:
    from ..mission.store import write_case, write_index

    folder = Path(out_dir)
    missions = folder / "missions"
    for case in suite.cases:
        write_case(str(missions), case.document, case.meta)
    write_index(str(missions), [case.meta for case in suite.cases],
                header=dict(suite.to_dict(), family=config.name,
                            config=str(Path(config.path).resolve())))
    taken = set(suite.indices)
    with open(folder / "original.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "index", "band", "rho", "verdict", "tick", "crash_kind", "taken",
            "seconds", "error"))
        writer.writeheader()
        for case in campaign:
            f = flown.get(case.meta.index)
            if f is None:
                continue
            writer.writerow({"index": f.index, "band": case.meta.band,
                             "rho": case.meta.rho, "verdict": f.verdict,
                             "tick": "" if f.tick is None else f.tick,
                             "crash_kind": f.crash_kind,
                             "taken": int(f.index in taken),
                             "seconds": round(f.seconds, 3),
                             "error": f.error})
    return folder


FLIGHT_FIELDS = ("id", "index", "verdict", "tick", "crash_kind", "seconds",
                 "error")


def write_flights(path, readings: Mapping[str, SuiteReading]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLIGHT_FIELDS)
        writer.writeheader()
        for key, reading in readings.items():
            for index in sorted(reading.flights):
                row = reading.flights[index].to_dict()
                row["id"] = key
                writer.writerow({k: ("" if row.get(k) is None else row[k])
                                 for k in FLIGHT_FIELDS})
    os.replace(partial, path)
    return str(path)


__all__ = ["CHUNK", "FLIGHT_FIELDS", "Flown", "Flyer", "KINDS", "PARSED",
           "SUITE_DIR", "Suite", "SuiteReading", "collect", "flips", "write",
           "write_flights"]
