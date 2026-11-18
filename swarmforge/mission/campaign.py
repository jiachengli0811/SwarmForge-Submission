from __future__ import annotations

import collections
import csv
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..solve.config import FamilyConfig, read_config
from ..solve.record import Record, read_record
from .align import alignment_rows, text as alignment_text, write_alignment
from .behaviors import Settings as BehaviorSettings, measure as measure_behaviors
from .field import Field, mission_defaults, moving_goal, random_field
from .fly import fly
from .measure import Space, spaces_of
from .store import (CaseMeta, Result, case_name, digest, write_case,
                    write_index, write_results)
from .test_range import TestRange
from .verdict import Oracle, decisive, judge


@dataclass(frozen=True)
class Case:
    meta: CaseMeta
    document: Field


def range_of(config: FamilyConfig) -> TestRange:
    if config.test_range is None:
        raise ValueError("%s declares no test_range block" % config.path)
    return TestRange.of(config.test_range)


def random_cases(config: FamilyConfig, count: int, seed: int = 0
                 ) -> List[Case]:
    test_range = range_of(config)
    knobs = mission_defaults(config)
    out: List[Case] = []
    for index, slot in enumerate(test_range.schedule(count,
                                                     "%d:random" % seed)):
        field_seed = random.Random("%d:field:%d"
                                   % (seed, index)).randrange(1 << 30)
        document = random_field(
            field_seed, density=slot.rho,
            min_spacing=max(slot.spacing, 2.0 * knobs["radius"]), **knobs)
        document = moving_goal(document, knobs["goal_speed"],
                               knobs["goal_lead"])
        spawn_seed = random.Random("%d:spawn:%s" % (
            seed, digest(moving_goal(document, None)))).randrange(1 << 30)
        out.append(Case(CaseMeta(
            index=index, digest=digest(document), spawn_seed=spawn_seed,
            field_seed=field_seed, band=slot.band,
            target_success=round(slot.target_success, 6),
            rho=float("%.6g" % slot.rho),
            spacing=round(max(slot.spacing, 2.0 * knobs["radius"]), 6),
            ticks=int(document["ticks"])), document))
    return out


def fly_case(config: FamilyConfig, record: Record, case: Case, *,
             name: str = "", every: int = 4,
             oracle: Optional[Oracle] = None,
             spaces: Optional[Mapping[str, Space]] = None,
             probe: Optional[Any] = None) -> Result:
    oracle = oracle or Oracle.for_config(config)
    goal = (float(case.document["goal"][0]), float(case.document["goal"][1]))
    started = time.time()
    flight = fly(case.document, config, record,
                 spawn_seed=case.meta.spawn_seed,
                 every=1 if spaces else every,
                 decides=lambda snapshot: decisive(snapshot, goal, oracle),
                 probe=probe)
    if probe is not None:
        probe.snapshots = flight.snapshots
    outcome = judge(flight.snapshots, goal=goal, oracle=oracle,
                    completed=flight.completed)
    reach = {}
    if spaces and not outcome.errored:
        for model, space in spaces.items():
            reach[model] = space.read(
                flight.snapshots, case.document,
                settle=oracle.settle_ticks, verdict=outcome.verdict.value,
                verdict_tick=outcome.tick,
                crash_kind=outcome.crash_kind).to_dict()
    behaviors = ({} if outcome.errored else
                 measure_behaviors(flight.snapshots, case.document,
                                   BehaviorSettings.for_config(config)))
    meta = case.meta
    return Result(record=name, case=case_name(meta.index), index=meta.index,
                  band=meta.band, target_success=meta.target_success,
                  rho=meta.rho, spacing=meta.spacing,
                  spawn_seed=meta.spawn_seed, verdict=outcome.verdict.value,
                  failed=outcome.failed, verdict_tick=outcome.tick,
                  crash_kind=outcome.crash_kind, ticks_run=flight.ticks_run,
                  seconds=round(time.time() - started, 3),
                  error=flight.error or ("" if not outcome.errored else
                                         str(outcome.detail.get("note", ""))),
                  reach=reach, arm=meta.arm, behaviors=behaviors)


_WORKER: Dict[str, Any] = {}


def _start_worker(config_path: str, records: Mapping[str, str],
                  cases: Sequence[Case], models: Sequence[str] = ()) -> None:
    config = read_config(config_path)
    _WORKER.update(
        config=config, oracle=Oracle.for_config(config),
        records={name: read_record(Path(path), config.physics.record_params)
                 for name, path in records.items()},
        cases={case.meta.index: case for case in cases},
        models=tuple(models), spaces={})


def _spaces(name: str) -> Dict[str, Space]:
    built = _WORKER["spaces"]
    if name not in built:
        built[name] = spaces_of(_WORKER["config"], _WORKER["records"][name],
                                _WORKER["models"])
    return built[name]


def _fly_task(name: str, index: int, every: int) -> Result:
    return fly_case(_WORKER["config"], _WORKER["records"][name],
                    _WORKER["cases"][index], name=name, every=every,
                    oracle=_WORKER["oracle"],
                    spaces=_spaces(name) if _WORKER["models"] else None)


def fly_all(config_path: str, records: Mapping[str, str],
            cases: Sequence[Case], *, every: int = 4, workers: int = 1,
            models: Sequence[str] = (),
            on_result: Optional[Callable[[Result], None]] = None
            ) -> Dict[str, List[Result]]:
    tasks = [(name, case.meta.index) for name in records for case in cases]
    out: Dict[str, List[Result]] = {name: [] for name in records}
    if workers <= 1:
        _start_worker(str(config_path), records, cases, models)
        for name, index in tasks:
            result = _fly_task(name, index, every)
            out[name].append(result)
            if on_result is not None:
                on_result(result)
    else:
        with ProcessPoolExecutor(
                max_workers=int(workers), initializer=_start_worker,
                initargs=(str(config_path), dict(records), list(cases),
                          tuple(models))) as pool:
            futures = [pool.submit(_fly_task, name, index, every)
                       for name, index in tasks]
            for future in as_completed(futures):
                result = future.result()
                out[result.record].append(result)
                if on_result is not None:
                    on_result(result)
    for name in out:
        out[name].sort(key=lambda r: r.index)
    return out


SUMMARY_FIELDS = ("record", "band", "cases", "flown", "success", "failed",
                  "fail_rate", "crash", "agent_crash", "obstacle_crash",
                  "both_crash", "timeout", "errors", "mean_success_tick")


def tally(results: Sequence[Result]) -> Dict[str, Any]:
    flown = [r for r in results if r.verdict != "error"]
    kinds = collections.Counter(r.crash_kind for r in flown
                                if r.verdict == "crash")
    ticks = [r.verdict_tick for r in flown
             if r.verdict == "success" and r.verdict_tick is not None]
    failed = sum(1 for r in flown if r.failed)
    return {"cases": len(results), "flown": len(flown),
            "success": sum(1 for r in flown if r.verdict == "success"),
            "failed": failed,
            "fail_rate": round(failed / len(flown), 4) if flown else "",
            "crash": sum(kinds.values()), "agent_crash": kinds["agent"],
            "obstacle_crash": kinds["obstacle"], "both_crash": kinds["both"],
            "timeout": sum(1 for r in flown if r.verdict == "timeout"),
            "errors": len(results) - len(flown),
            "mean_success_tick": (round(sum(ticks) / len(ticks), 1)
                                  if ticks else "")}


def summary_rows(results: Mapping[str, Sequence[Result]]
                 ) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, flown in results.items():
        rows.append(dict(tally(flown), record=name, band="all"))
        bands = sorted({r.band for r in flown if r.band is not None})
        for band in bands:
            rows.append(dict(tally([r for r in flown if r.band == band]),
                             record=name, band=band))
    return rows


def write_summary(directory: str, results: Mapping[str, Sequence[Result]]
                  ) -> str:
    path = os.path.join(directory, "summary.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows(results):
            writer.writerow(row)
    return path


def run(config_path: str, records: Mapping[str, str], *, count: int,
        seed: int = 0, out_dir: Optional[str] = None, every: int = 4,
        workers: int = 1, models: Sequence[str] = (), progress: bool = True
        ) -> Dict[str, List[Result]]:
    config = read_config(config_path)
    cases = random_cases(config, count, seed)
    test_range = range_of(config)
    header = {"family": config.name,
              "config": str(Path(config_path).resolve()),
              "seed": int(seed), "count": int(count),
              "mission_defaults": {k: v for k, v in
                                   mission_defaults(config).items()},
              "oracle": Oracle.for_config(config).to_dict(),
              "test_range": test_range.to_dict()}
    if out_dir:
        missions = os.path.join(out_dir, "missions")
        for case in cases:
            write_case(missions, case.document, case.meta)
        write_index(missions, [case.meta for case in cases], header=header)

    started = time.time()
    done = [0]
    total = len(cases) * len(records)

    def report(result: Result) -> None:
        done[0] += 1
        if progress:
            print("  [%d/%d] %-12s %s b%s %-8s @%-5s %-8s %5.1fs %s"
                  % (done[0], total, result.record, result.case,
                     result.band, result.verdict,
                     "-" if result.verdict_tick is None
                     else result.verdict_tick, result.crash_kind or "-",
                     result.seconds, result.error), flush=True)

    spaces = {name: {model: space.to_dict() for model, space in spaces_of(
        config, read_record(Path(path), config.physics.record_params),
        models).items()} for name, path in records.items()}
    results = fly_all(config_path, records, cases, every=every,
                      workers=workers, models=models, on_result=report)
    rows = alignment_rows(results, spaces) if models else []
    if out_dir:
        for name, flown in results.items():
            write_results(os.path.join(out_dir, name), flown, header=dict(
                header, record=name,
                record_path=str(Path(records[name]).resolve()),
                every=1 if models else int(every), spaces=spaces[name]))
        write_summary(out_dir, results)
        if rows:
            write_alignment(out_dir, rows)
    if progress:
        print("flew %d missions x %d records in %.0fs"
              % (len(cases), len(records), time.time() - started))
        if rows:
            print(alignment_text(rows))
    return results


__all__ = ["Case", "SUMMARY_FIELDS", "fly_all", "fly_case", "random_cases",
           "run", "summary_rows", "tally", "range_of", "write_summary"]
