from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .candidate import ORIGINAL

MATCH: Dict[str, Tuple[str, ...]] = {
    "obstacle_collision": ("obstacle", "both"),
    "agent_collision": ("agent", "both"),
    "timeout": ("timeout",),
}


@dataclass
class Entry:
    id: str
    parent: str
    retained: bool
    total: Optional[float]
    scores: Dict[str, float]
    properties: str
    file: Path
    row: Dict[str, str] = field(repr=False, default_factory=dict)


def _number(text: str) -> Optional[float]:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_run(run: Path, models: Sequence[str]) -> Dict[str, Entry]:
    out: Dict[str, Entry] = {}
    with open(Path(run) / "candidates.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            total = _number(row["total"])
            scores = {m: _number(row.get(m, "")) for m in models}
            out[row["id"]] = Entry(
                id=row["id"], parent=row["parent"],
                retained=row["retained"] == "1",
                total=total if row["stage"] == "measured" else None,
                scores={m: v for m, v in scores.items() if v is not None},
                properties=row.get("properties_changed", ""),
                file=Path(run) / row["file"], row=row)
    if ORIGINAL not in out:
        raise ValueError("%s holds no original" % run)
    return out


def lineage(entries: Mapping[str, Entry], cid: str) -> List[str]:
    chain = [cid]
    while entries[chain[-1]].parent:
        chain.append(entries[chain[-1]].parent)
    return chain[::-1]


@dataclass(frozen=True)
class Pick:
    id: str
    role: str
    level: Optional[float] = None


def sample(entries: Mapping[str, Entry], count: int,
           models: Sequence[str]) -> List[Pick]:
    pool = [e for e in entries.values()
            if e.retained and e.total is not None and e.id != ORIGINAL]
    original = entries[ORIGINAL]
    picks = [Pick(ORIGINAL, "original", original.total)]
    if not pool:
        return picks
    best = min(pool, key=lambda e: (e.total, e.id))
    trace = set(lineage(entries, best.id))
    top = float(original.total)
    step = (top - best.total) / max(count, 1)
    taken = {ORIGINAL}
    for i in range(1, count + 1):
        level = top - step * i
        free = [e for e in pool if e.id not in taken]
        if not free:
            break
        nearest = min(free, key=lambda e: (abs(e.total - level), e.id))
        on_trace = [e for e in free if e.id in trace]
        if on_trace:
            close = min(on_trace, key=lambda e: (abs(e.total - level), e.id))
            if abs(close.total - level) <= abs(nearest.total - level) \
                    + 0.5 * step:
                nearest = close
        taken.add(nearest.id)
        picks.append(Pick(nearest.id, "level-%d" % i, round(level, 6)))
    for model in models:
        scored = [e for e in pool if model in e.scores]
        if not scored:
            continue
        low = min(scored, key=lambda e: (e.scores[model], e.total, e.id))
        if low.id not in taken:
            taken.add(low.id)
            picks.append(Pick(low.id, "low-%s" % model))
    return picks


def mission_rates(results: Sequence[Any]) -> Dict[str, Any]:
    flown = [r for r in results if r.verdict != "error"]
    n = len(flown)
    exposure = sum(int(r.verdict_tick if r.verdict_tick is not None
                       else r.ticks_run) for r in flown)
    crashes = [r for r in flown if r.verdict == "crash"]
    survived = [r for r in flown if r.verdict != "crash"]
    out: Dict[str, Any] = {
        "flown": n, "errors": len(results) - n,
        "success": sum(1 for r in flown if r.verdict == "success"),
        "fail_rate": (sum(1 for r in flown if r.failed) / n) if n else None,
        "exposure": exposure}
    for model, kinds in MATCH.items():
        if kinds == ("timeout",):
            hit = sum(1 for r in flown if r.verdict == "timeout")
            out["%s_count" % model] = hit
            out["%s_share" % model] = hit / n if n else None
            out["%s_rate" % model] = (hit / len(survived)
                                      if survived else None)
        else:
            hit = sum(1 for r in crashes if r.crash_kind in kinds)
            out["%s_count" % model] = hit
            out["%s_share" % model] = hit / n if n else None
            out["%s_rate" % model] = (1000.0 * hit / exposure
                                      if exposure else None)
    return out


def fly(config_path: Path, records: Mapping[str, Path], *, missions: int,
        seed: int, out_dir: Path, workers: int, progress: bool = False
        ) -> Dict[str, List[Any]]:
    from ..mission.campaign import run as campaign

    return campaign(str(config_path), {k: str(v) for k, v in records.items()},
                    count=missions, seed=seed, out_dir=str(out_dir),
                    workers=workers, progress=progress)


def flown(out_dir: Path, names: Sequence[str]) -> Dict[str, List[Any]]:
    from ..mission.store import read_results

    return {name: read_results(str(Path(out_dir) / name))[1]
            for name in names}


THROUGH_FIELDS = ("record", "index", "band", "agent", "obstacle", "arrived",
                  "ticks", "error")
THROUGH_DIR = "through"


def fly_through_case(config: Any, record: Any, case: Any,
                     oracle: Any) -> Dict[str, Any]:
    from ..mission.fly import fly as fly_mission
    from ..mission.verdict import contact_kind

    goal = (float(case.document["goal"][0]), float(case.document["goal"][1]))
    first: Dict[str, Optional[int]] = {"agent": None, "obstacle": None}

    def decides(snapshot) -> bool:
        kind = contact_kind(snapshot, oracle)
        for k in first:
            if first[k] is None and kind in (k, "both"):
                first[k] = snapshot.tick
        need = max(1, math.ceil(oracle.arrive_fraction
                                * len(snapshot.positions)))
        inside = sum(1 for p in snapshot.positions
                     if math.dist(p, goal) <= oracle.goal_radius)
        return inside >= need

    flight = fly_mission(case.document, config, record,
                         spawn_seed=case.meta.spawn_seed, every=1 << 30,
                         decides=decides)
    return {"index": case.meta.index, "band": case.meta.band,
            "agent": first["agent"], "obstacle": first["obstacle"],
            "arrived": flight.ticks_run if flight.decided else None,
            "ticks": flight.ticks_run,
            "error": "" if flight.completed else (flight.error
                                                  or "incomplete")}


_THROUGH: Dict[str, Any] = {}


def _through_start(config_path: str, records: Mapping[str, str],
                   cases: Sequence[Any]) -> None:
    from ..mission.verdict import Oracle
    from ..solve.config import read_config
    from ..solve.record import read_record

    config = read_config(config_path)
    _THROUGH.update(
        config=config, oracle=Oracle.for_config(config),
        records={name: read_record(Path(path), config.physics.record_params)
                 for name, path in records.items()},
        cases={case.meta.index: case for case in cases})


def _through_task(name: str, index: int) -> Dict[str, Any]:
    out = fly_through_case(_THROUGH["config"], _THROUGH["records"][name],
                           _THROUGH["cases"][index], _THROUGH["oracle"])
    out["record"] = name
    return out


def fly_through(config_path: Path, records: Mapping[str, Path], *,
                missions: int, seed: int, out_dir: Path, workers: int
                ) -> Dict[str, List[Dict[str, Any]]]:
    from concurrent.futures import ProcessPoolExecutor

    from ..mission.campaign import random_cases
    from ..solve.config import read_config

    cases = random_cases(read_config(config_path), missions, seed)
    paths = {name: str(path) for name, path in records.items()}
    names = [name for name in records for _case in cases]
    indices = [case.meta.index for _name in records for case in cases]
    out: Dict[str, List[Dict[str, Any]]] = {name: [] for name in records}
    if workers <= 1:
        _through_start(str(config_path), paths, cases)
        results = [_through_task(n, i) for n, i in zip(names, indices)]
    else:
        with ProcessPoolExecutor(max_workers=int(workers),
                                 initializer=_through_start,
                                 initargs=(str(config_path), paths,
                                           list(cases))) as pool:
            results = list(pool.map(_through_task, names, indices,
                                    chunksize=4))
    for result in results:
        out[result["record"]].append(result)
    folder = Path(out_dir) / THROUGH_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for name, flights in out.items():
        flights.sort(key=lambda f: f["index"])
        with open(folder / ("%s.csv" % name), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=THROUGH_FIELDS)
            writer.writeheader()
            for flight in flights:
                writer.writerow({k: ("" if flight.get(k) is None
                                     else flight[k]) for k in THROUGH_FIELDS})
    return out


def flown_through(out_dir: Path, names: Sequence[str]
                  ) -> Dict[str, List[Dict[str, Any]]]:
    def value(text: str) -> Optional[int]:
        return int(text) if text not in ("", None) else None

    out = {}
    for name in names:
        with open(Path(out_dir) / THROUGH_DIR / ("%s.csv" % name),
                  newline="") as handle:
            out[name] = [dict(row, index=int(row["index"]),
                              agent=value(row["agent"]),
                              obstacle=value(row["obstacle"]),
                              arrived=value(row["arrived"]),
                              ticks=int(row["ticks"]))
                         for row in csv.DictReader(handle)]
    return out


def through_rates(flights: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    flown_ = [f for f in flights if not f["error"]]
    n = len(flown_)
    if not n:
        return {"through_flown": 0}
    return {"through_flown": n,
            "obstacle_collision_through":
                sum(f["obstacle"] is not None for f in flown_) / n,
            "agent_collision_through":
                sum(f["agent"] is not None for f in flown_) / n,
            "timeout_through": sum(f["arrived"] is None for f in flown_) / n}


def _ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[order[j + 1]] == \
                values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    ra, rb = (_ranks([p[k] for p in pairs]) for k in (0, 1))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def rows(entries: Mapping[str, Entry], picks: Sequence[Pick],
         results: Mapping[str, Sequence[Any]],
         models: Sequence[str],
         through: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None
         ) -> List[Dict[str, Any]]:
    out = []
    for pick in picks:
        entry = entries[pick.id]
        row: Dict[str, Any] = {
            "id": pick.id, "role": pick.role, "level": pick.level,
            "parent": entry.parent,
            "generation": int(entry.row.get("generation") or 0),
            "operator": entry.row.get("operator", ""),
            "total": entry.total,
            "properties_changed": entry.properties}
        for model in models:
            row[model] = entry.scores.get(model)
        row.update(mission_rates(results[pick.id]))
        if through is not None and pick.id in through:
            row.update(through_rates(through[pick.id]))
        out.append(row)
    return out


def correlations(table: Sequence[Mapping[str, Any]],
                 models: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for model in models:
        if model not in MATCH:
            continue
        scores = [r[model] for r in table]
        out[model] = {
            "share": spearman(scores, [r["%s_share" % model] for r in table]),
            "rate": spearman(scores, [r["%s_rate" % model] for r in table]),
            "through": spearman(scores, [r.get("%s_through" % model)
                                         for r in table])}
    out["total"] = {"fail_rate": spearman([r["total"] for r in table],
                                          [r["fail_rate"] for r in table])}
    return out


COLUMNS = ("id", "role", "level", "parent", "generation", "operator",
           "total")


def write(out_dir: Path, table: Sequence[Mapping[str, Any]],
          trend: Mapping[str, Any], header: Mapping[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = list(table[0]) if table else list(COLUMNS)
    with open(out_dir / "trend.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in table:
            writer.writerow({k: ("" if row.get(k) is None else row[k])
                             for k in columns})
    with open(out_dir / "trend.json", "w") as handle:
        json.dump({"schema": "swarmforge.harden.trend.v1",
                   "header": dict(header), "trend": dict(trend),
                   "rows": list(table)}, handle, indent=1, default=str)


def text(table: Sequence[Mapping[str, Any]], trend: Mapping[str, Any],
         models: Sequence[str]) -> str:
    def f(value, spec):
        return "-" if value is None else spec % value

    short = {m: m.split("_")[0][:5] for m in models}
    through = any("%s_through" % m in r for r in table for m in models)
    lines = ["%-10s %-14s %8s  %s  %5s  %s%s" % (
        "record", "role", "total",
        " ".join("%7s" % short[m] for m in models), "fail",
        " ".join("%11s" % ("%s n/rate" % short[m]) for m in models),
        ("  through: " + " ".join("%5s" % short[m] for m in models))
        if through else "")]
    for r in table:
        lines.append("%-10s %-14s %8s  %s  %5s  %s%s" % (
            r["id"], r["role"][:14], f(r["total"], "%.4f"),
            " ".join("%7s" % f(r[m], "%.4f") for m in models),
            f(r["fail_rate"], "%.3f"),
            " ".join("%11s" % ("%d/%s" % (r["%s_count" % m],
                                          f(r["%s_rate" % m], "%.3f")))
                     for m in models),
            ("           " + " ".join("%5s" % f(r.get("%s_through" % m),
                                               "%.3f") for m in models))
            if through else ""))
    lines.append("Spearman, score vs its failures (share; rate%s): "
                 % ("; through" if through else "") + "; ".join(
                     "%s %s; %s%s" % (
                         short[m], f(trend[m]["share"], "%+.2f"),
                         f(trend[m]["rate"], "%+.2f"),
                         ("; " + f(trend[m].get("through"), "%+.2f"))
                         if through else "")
                     for m in models if m in trend)
                 + "; total vs fail rate %s"
                 % f(trend["total"]["fail_rate"], "%+.2f"))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ..solve.config import read_config
    from .config import read_harden

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("config", type=Path)
    parser.add_argument("--run", type=Path, required=True,
                        help="a hardening run's folder")
    parser.add_argument("--sample", type=int, default=8,
                        help="weakness levels from the original to the best")
    parser.add_argument("--missions", type=int, default=400)
    parser.add_argument("--seed", type=int, required=True,
                        help="the corpus: campaign seed of the missions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--through", action="store_true",
                        help="also fly every mission on past contact: each "
                             "failure kind judged on every mission")
    parser.add_argument("--no-fly", action="store_true",
                        help="read the flights already under --out")
    args = parser.parse_args(argv)

    config = read_config(args.config)
    models = tuple(read_harden(args.config).objective.models)
    entries = read_run(args.run, models)
    picks = sample(entries, args.sample, models)
    records = {p.id: entries[p.id].file for p in picks}
    through = None
    if args.no_fly:
        results = flown(args.out, list(records))
        if (Path(args.out) / THROUGH_DIR).is_dir():
            through = flown_through(args.out, list(records))
    else:
        results = fly(args.config, records, missions=args.missions,
                      seed=args.seed, out_dir=args.out,
                      workers=args.workers)
        if args.through:
            through = fly_through(args.config, records,
                                  missions=args.missions, seed=args.seed,
                                  out_dir=args.out, workers=args.workers)
    table = rows(entries, picks, results, models, through)
    trend = correlations(table, models)
    write(args.out, table, trend, {
        "config": str(Path(args.config).resolve()), "family": config.name,
        "run": str(Path(args.run).resolve()), "sample": args.sample,
        "missions": args.missions, "seed": args.seed,
        "picks": [p.__dict__ for p in picks]})
    print(text(table, trend, models))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["Entry", "MATCH", "Pick", "THROUGH_FIELDS", "correlations", "fly",
           "fly_through", "fly_through_case", "flown", "flown_through",
           "lineage", "mission_rates", "read_run", "rows", "sample",
           "spearman", "text", "through_rates", "write"]
