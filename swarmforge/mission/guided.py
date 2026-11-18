from __future__ import annotations

import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ..solve.config import read_config
from ..solve.models import MISSION_MODELS
from ..solve.record import read_record
from . import aim as A
from . import movers as M
from .campaign import Case, fly_case, random_cases
from .measure import Approach, Space, failure_of, spaces_of
from .store import CaseMeta, Result, digest, write_case, write_index, \
    write_results
from .verdict import Oracle

PRIOR = 2.0
EXPOSED = 0.3
NEW_WEIGHT = 2.0
PRIOR_FAIL = (1.0, 1.0)
PRIOR_NEW = (1.0, 2.0)
TRIES = 4
LINEAGE = 0.5
REUSE = ("transplant", "mutate", "steer", "climb")


@dataclass
class Entry:
    index: int
    document: Dict[str, Any]
    spawn: int
    root: int = -1
    tactic: str = ""


@dataclass
class Touch:
    entry: Entry
    where: Tuple[float, float]
    heading: Tuple[float, float]
    body: str
    point: np.ndarray
    distance: float
    failed: bool = False
    charged: int = 0
    target: Optional[np.ndarray] = None

    def rank(self) -> Tuple[int, int, float]:
        return (int(self.failed), self.charged, -self.distance)


class Board:
    def __init__(self, space: Space, prior: float = PRIOR):
        self.space = space
        self.prior = float(prior)
        r = space.size
        self.share = np.asarray(space.share, dtype=float)
        self.reached = np.zeros(r)
        self.charged = np.zeros(r)
        self.failed = np.zeros(r)
        self.attempts = np.zeros(r)
        self.elite: Dict[int, Touch] = {}
        self.nearest: Dict[int, Touch] = {}
        self.weak: Dict[int, Touch] = {}
        self.weak_min: Dict[int, float] = {}
        self.evident: Set[int] = set()
        centres = space.cells.centres
        from scipy.spatial import cKDTree
        self.scale = float(np.median(cKDTree(centres).query(centres,
                                                             k=2)[0][:, 1]))

    def risk(self) -> np.ndarray:
        seen = self.reached + self.prior
        return np.divide(self.charged + self.prior * self.share, seen,
                         out=np.zeros_like(self.share), where=seen > 0)

    def value(self) -> np.ndarray:
        novelty = np.where(self.failed > 0, EXPOSED, 1.0)
        spread = 1.0 / (1.0 + self.attempts)
        near = np.ones_like(self.share)
        for j in range(self.share.size):
            if self.reached[j] == 0:
                touch = self.nearest.get(j)
                near[j] = (math.exp(-touch.distance / self.scale)
                           if touch is not None else 0.0)
        return self.risk() * novelty * spread * near

    def climb_value(self) -> np.ndarray:
        novelty = np.where(self.failed > 0, EXPOSED, 1.0)
        spread = 1.0 / (1.0 + self.attempts)
        near = np.zeros_like(self.share)
        for j, touch in self.weak.items():
            near[j] = math.exp(-touch.distance / self.scale)
        return near * novelty * spread

    def update(self, entry: Entry, reading: Mapping[str, Any],
               approach: Optional[Approach], failed: bool) -> None:
        cells = {int(k) for k in reading.get("cells", {})}
        charged = {int(k): v for k, v in
                   reading.get("charged_cells", {}).items()}
        through = reading.get("failure_cell") if failed else None
        if charged or failed:
            self.evident.add(entry.index)
        for j in cells:
            self.reached[j] += 1
        for j in charged:
            self.charged[j] += 1
        if through is not None:
            self.failed[int(through)] += 1
        if approach is None:
            return
        if approach.weak_distance is not None:
            finite = approach.weak_distance[np.isfinite(
                approach.weak_distance)]
            if finite.size:
                self.weak_min[entry.index] = float(finite.min())
            for j in np.flatnonzero(np.isfinite(approach.weak_distance)):
                d = float(approach.weak_distance[j])
                best = self.weak.get(int(j))
                if best is None or d < best.distance:
                    self.weak[int(j)] = Touch(
                        entry, tuple(map(float, approach.weak_where[j])),
                        tuple(map(float, approach.weak_heading[j])),
                        approach.weak_body[j],
                        np.array(approach.weak_point[j]), d,
                        target=np.array(approach.weak_target[j]))
        for j in range(self.share.size):
            d = float(approach.distance[j])
            if not np.isfinite(d):
                continue
            touch = Touch(entry, tuple(map(float, approach.where[j])),
                          tuple(map(float, approach.heading[j])),
                          approach.body[j], np.array(approach.point[j]), d,
                          failed=through is not None and int(through) == j,
                          charged=int(charged.get(j, 0)))
            if j in cells:
                best = self.elite.get(j)
                if best is None or touch.rank() > best.rank():
                    self.elite[j] = touch
            best = self.nearest.get(j)
            if best is None or d < best.distance:
                self.nearest[j] = touch


class Operators:
    def __init__(self, names: Sequence[str], rng: np.random.Generator,
                 uniform: bool = False):
        self.names = list(names)
        self.rng = rng
        self.uniform = uniform
        self.fail = {n: list(PRIOR_FAIL) for n in names}
        self.new = {n: list(PRIOR_NEW) for n in names}
        self.pulls = {n: 0 for n in names}

    def order(self) -> List[str]:
        if self.uniform:
            names = list(self.names)
            self.rng.shuffle(names)
            return names
        score = {n: self.rng.beta(*self.fail[n])
                 + NEW_WEIGHT * self.rng.beta(*self.new[n])
                 for n in self.names}
        return sorted(self.names, key=lambda n: -score[n])

    def update(self, name: str, failed: bool, new: bool) -> None:
        self.pulls[name] += 1
        self.fail[name][0 if failed else 1] += 1.0
        self.new[name][0 if new else 1] += 1.0

    def table(self) -> List[Dict[str, Any]]:
        return [{"operator": n, "pulls": self.pulls[n],
                 "p_fail": round(self.fail[n][0] / sum(self.fail[n]), 4),
                 "p_new": round(self.new[n][0] / sum(self.new[n]), 4)}
                for n in self.names]


@dataclass
class Pick:
    operator: str
    model: str = ""
    cell: int = -1
    parent: int = -1
    detail: str = ""
    value: float = 0.0
    before: Optional[float] = None
    root: int = -1
    mover: str = ""


class Planner:
    def __init__(self, course: A.Course, boards: Mapping[str, Board],
                 seed: int, operators: Sequence[str] = A.OPERATORS,
                 uniform: bool = False, greedy_steer: bool = True,
                 lineage: bool = False, evidence: bool = False):
        self.course = course
        self.boards = dict(boards)
        self.seed = int(seed)
        self.rng = np.random.default_rng([int(seed), 0x6E1D])
        self.ops = Operators(operators, self.rng, uniform=uniform)
        self.greedy_steer = greedy_steer
        self.lineage = lineage
        self.reuses: Dict[int, int] = {}
        self.evidence = evidence
        self.exposed: Set[Tuple[str, int]] = set()
        self.entries: Dict[int, Entry] = {}
        if course.mover_draw == "learned":
            table = M.levers(course.config.name)
            if table is not None:
                course.sampler = M.TacticDraw(table, self.rng)

    def _touch(self, operator: str, board: "Board", j: int
               ) -> Optional[Touch]:
        touch = None
        if operator in ("mutate", "transplant"):
            touch = board.elite.get(j)
        elif operator == "steer":
            touch = board.nearest.get(j) if board.reached[j] == 0 else None
        elif operator == "climb":
            touch = board.weak.get(j)
        if (touch is not None and self.evidence
                and touch.entry.index not in board.evident):
            return None
        return touch

    def _targets(self, operator: str, slot: Optional[A.Slot] = None
                 ) -> List[Tuple[float, str, int]]:
        out = []
        for model, board in self.boards.items():
            value = (board.climb_value() if operator == "climb"
                     else board.value())
            for j in np.flatnonzero(value > 0.0):
                v = float(value[j])
                if operator in REUSE:
                    touch = self._touch(operator, board, int(j))
                    if touch is None:
                        continue
                    if (slot is not None and slot.movers < 1
                            and touch.entry.tactic not in ("", M.RANDOM)):
                        continue
                    if self.lineage:
                        v /= (1.0 + self.reuses.get(touch.entry.root,
                                                    0)) ** LINEAGE
                out.append((v, model, int(j)))
        return out

    def _draw(self, targets, tried: Set[Tuple[str, int]]):
        pool = [t for t in targets if (t[1], t[2]) not in tried]
        if not pool:
            return None
        weights = np.array([t[0] for t in pool])
        k = int(self.rng.choice(len(pool), p=weights / weights.sum()))
        return pool[k]

    def build(self, slot: A.Slot, index: int, name: str
              ) -> Tuple[Dict[str, Any], int, Pick]:
        for operator in self.ops.order():
            if operator in ("random", "corridor"):
                seed = self._seed(index, operator, 0)
                build = (A.random_mission if operator == "random"
                         else A.corridor_mission)
                self.course.drawn = []
                document = build(seed, self.course, slot, name)
                if document is not None:
                    return document, -1, Pick(operator,
                                              mover=self._mover(document))
                continue
            targets = self._targets(operator, slot)
            tried: Set[Tuple[str, int]] = set()
            for attempt in range(TRIES):
                target = self._draw(targets, tried)
                if target is None:
                    break
                value, model, cell = target
                tried.add((model, cell))
                self.course.drawn = []
                built = self._make(operator, model, cell, slot, index, name,
                                   attempt)
                if built is not None:
                    document, spawn, pick = built
                    pick.value = value
                    pick.mover = self._mover(document)
                    self.boards[model].attempts[cell] += 1
                    if operator in REUSE:
                        root = self._touch(operator, self.boards[model],
                                           cell).entry.root
                        pick.root = root
                        self.reuses[root] = self.reuses.get(root, 0) + 1
                    return document, spawn, pick
        seed = self._seed(index, "random", 1)
        self.course.drawn = []
        document = A.random_mission(seed, self.course, slot, name)
        if document is None:
            raise RuntimeError("slot %d builds no mission" % index)
        return document, -1, Pick("random", detail="fallback",
                                  mover=self._mover(document))

    def _mover(self, document: Mapping[str, Any]) -> str:
        if not document.get("dynamic") or not self.course.drawn:
            return ""
        return self.course.drawn[-1]

    def _seed(self, index: int, operator: str, attempt: int) -> int:
        return random.Random("%d:guided:%d:%s:%d" % (
            self.seed, index, operator, attempt)).randrange(1 << 30)

    def _make(self, operator, model, cell, slot, index, name, attempt):
        board = self.boards[model]
        seed = self._seed(index, operator, attempt)
        course = self.course
        if operator == "aim":
            space = board.space
            centre = space.cells.centres[cell]
            target = space.raw(centre)
            quantile = dict(zip(space.names, map(float, centre)))
            document, account = A.aim(seed, course, slot, model, target,
                                      quantile, space.unit, space.goal_cap,
                                      name)
            if document is None:
                return None
            return document, -1, Pick("aim", model, cell, detail=account)
        if operator in ("mutate", "transplant"):
            touch = board.elite[cell]
            if operator == "mutate":
                document = A.mutate(seed, course, slot, touch.entry.document,
                                    touch.where, touch.body, name)
                spawn = touch.entry.spawn
            else:
                document = A.transplant(seed, course, slot,
                                        touch.entry.document, touch.where,
                                        touch.body, name)
                spawn = -1
            if document is None:
                return None
            return document, spawn, Pick(operator, model, cell,
                                         parent=touch.entry.index,
                                         detail=touch.body)
        if operator == "climb":
            touch = board.weak[cell]
            delta = {n: float(t - p) for n, t, p in
                     zip(board.space.names, touch.target, touch.point)
                     if np.isfinite(p) and np.isfinite(t)}
            document, edit = A.steer(seed, course, slot, touch.entry.document,
                                     touch.where, touch.body, touch.heading,
                                     delta, name, greedy=self.greedy_steer,
                                     model=model)
            if document is None:
                return None
            return document, touch.entry.spawn, Pick(
                "climb", model, cell, parent=touch.entry.index, detail=edit,
                before=board.weak_min.get(touch.entry.index))
        if operator == "steer":
            touch = board.nearest[cell]
            centre = board.space.cells.centres[cell]
            delta = {n: float(c - p) for n, c, p in
                     zip(board.space.names, centre, touch.point)
                     if np.isfinite(p)}
            document, edit = A.steer(seed, course, slot, touch.entry.document,
                                     touch.where, touch.body, touch.heading,
                                     delta, name, greedy=self.greedy_steer,
                                     model=model)
            if document is None:
                return None
            return document, touch.entry.spawn, Pick(
                "steer", model, cell, parent=touch.entry.index, detail=edit,
                before=touch.distance)
        raise ValueError("no operator %r" % operator)

    def update(self, pick: Pick, entry: Entry, result: Result,
               approaches: Mapping[str, Approach]) -> Dict[str, Any]:
        entry.tactic = pick.mover
        if not pick.mover and entry.document.get("dynamic") \
                and pick.parent in self.entries:
            entry.tactic = self.entries[pick.parent].tactic
        self.entries[entry.index] = entry
        through = []
        for model, board in self.boards.items():
            reading = result.reach.get(model, {})
            failed = failure_of(model, result.verdict, result.crash_kind)
            board.update(entry, reading, approaches.get(model), failed)
            if failed and reading.get("failure_cell") is not None:
                through.append((model, int(reading["failure_cell"])))
        new = [t for t in through if t not in self.exposed]
        self.exposed |= set(through)
        if pick.operator in self.ops.names:
            self.ops.update(pick.operator, result.failed, bool(new))
        if pick.mover and self.course.sampler is not None:
            self.course.sampler.update(pick.mover, result.failed)
        out: Dict[str, Any] = {"exposed": ["%s:%d" % t for t in through],
                               "new": ["%s:%d" % t for t in new]}
        if pick.model:
            reading = result.reach.get(pick.model, {})
            approach = approaches.get(pick.model)
            out["reached_target"] = str(pick.cell) in reading.get("cells", {})
            out["charged_target"] = (str(pick.cell)
                                     in reading.get("charged_cells", {}))
            if pick.operator == "climb":
                d = (np.nanmin(approach.weak_distance)
                     if approach is not None
                     and approach.weak_distance is not None else np.inf)
            else:
                d = (approach.distance[pick.cell] if approach is not None
                     else np.inf)
            out["target_distance"] = (round(float(d), 6) if np.isfinite(d)
                                      else None)
        return out


_WORKER: Dict[str, Any] = {}


def _start_worker(config_path: str, name: str, record_path: str,
                  models: Sequence[str]) -> None:
    config = read_config(config_path)
    record = read_record(Path(record_path), config.physics.record_params)
    _WORKER.update(config=config, name=name, record=record,
                   oracle=Oracle.for_config(config),
                   spaces=spaces_of(config, record, models))


def _fly(case: Case) -> Tuple[Result, Dict[str, Approach]]:
    spaces = _WORKER["spaces"]
    result = fly_case(_WORKER["config"], _WORKER["record"], case,
                      name=_WORKER["name"], oracle=_WORKER["oracle"],
                      spaces=spaces)
    approaches = ({m: s.last.approach for m, s in spaces.items()
                   if s.last is not None} if result.verdict != "error"
                  and result.reach else {})
    return result, approaches


def slots_of(config, count: int, seed: int) -> List[A.Slot]:
    out = []
    for case in random_cases(config, count, seed):
        meta = case.meta
        out.append(A.Slot(index=meta.index, band=meta.band,
                          target_success=meta.target_success, rho=meta.rho,
                          spacing=meta.spacing,
                          statics=len(case.document.get("static", [])),
                          movers=len(case.document.get("dynamic", []))))
    return out


def _input_seed(config_path, name, record_path, models, path, count):
    from .store import read_case
    document, meta = read_case(str(path))
    _start_worker(str(config_path), name, str(record_path), tuple(models))
    case = Case(CaseMeta(index=-1, digest=digest(document),
                         spawn_seed=int(meta.spawn_seed),
                         field_seed=int(document.get("seed", 0)),
                         band=meta.band, target_success=meta.target_success,
                         rho=meta.rho, spacing=meta.spacing,
                         ticks=int(document["ticks"])), document)
    result, _ = _fly(case)
    where, body, model = None, "", ""
    for m in models:
        reading = result.reach.get(m, {})
        if failure_of(m, result.verdict, result.crash_kind) and \
                reading.get("failure_where") is not None:
            where, body, model = reading["failure_where"], \
                reading.get("failure_body") or "", m
            break
    if where is None:
        raise ValueError("%s: the record does not fail the input mission %s "
                         "(%s %s), so there is no place to mutate round"
                         % (name, path, result.verdict, result.crash_kind))
    return {"path": str(Path(path).resolve()), "count": int(count),
            "verdict": result.verdict, "crash_kind": result.crash_kind,
            "verdict_tick": result.verdict_tick, "model": model,
            "where": [float(v) for v in where], "body": body,
            "spawn_seed": int(meta.spawn_seed), "band": meta.band,
            "rho": meta.rho, "spacing": meta.spacing,
            "document": document, "meta": meta}


def _input_mutant(planner, course, seeded, index, name):
    meta = seeded["meta"]
    for attempt in range(A.ATTEMPTS):
        seed = planner._seed(index, "input", attempt)
        course.drawn = []
        document = A.mutate_input(seed, course, seeded["document"],
                                  tuple(seeded["where"]), seeded["body"], name,
                                  rho=meta.rho, spacing=meta.spacing)
        if document is not None:
            break
    else:
        raise RuntimeError("input mutant %d builds no mission" % index)
    slot = A.Slot(index=index, band=meta.band,
                  target_success=meta.target_success, rho=meta.rho,
                  spacing=meta.spacing,
                  statics=len(document.get("static", [])),
                  movers=len(document.get("dynamic", [])))
    pick = Pick("input", detail=seeded["body"], root=0)
    return document, int(seeded["spawn_seed"]), pick, slot


def run_guided(config_path: str, name: str, record_path: str, *, count: int,
               seed: int = 0, out_dir: Optional[str] = None, workers: int = 1,
               batch: Optional[int] = None, models: Sequence[str] = None,
               progress: bool = True,
               operators: Sequence[str] = A.OPERATORS,
               uniform: bool = False, greedy_steer: bool = True,
               dense: bool = True, lineage: bool = True,
               prior: float = PRIOR, evidence: bool = True,
               tactics: bool = False, tactic_blind: bool = False,
               mover_draw: str = "random",
               input_mission: Optional[str] = None, input_count: int = 0
               ) -> Tuple[List[Result], Dict[str, Any]]:
    models = tuple(models or MISSION_MODELS)
    config = read_config(config_path)
    record = read_record(Path(record_path), config.physics.record_params)
    course = A.Course(config, dense=dense, tactics=tactics,
                      tactic_blind=tactic_blind, mover_draw=mover_draw)
    spaces = spaces_of(config, record, models)
    boards = {m: Board(s, prior) for m, s in spaces.items()}
    planner = Planner(course, boards, seed, operators=operators,
                      uniform=uniform, greedy_steer=greedy_steer,
                      lineage=lineage, evidence=evidence)
    slots = slots_of(config, count, seed)
    batch = max(1, int(batch or workers))
    seeded = _input_seed(config_path, name, record_path, models, input_mission,
                         input_count) if input_mission else None
    results: List[Result] = []
    log: List[Dict[str, Any]] = []
    cases: List[Case] = []
    started = time.time()
    pool = (ProcessPoolExecutor(max_workers=int(workers),
                                initializer=_start_worker,
                                initargs=(str(config_path), name,
                                          str(record_path), models))
            if workers > 1 else None)
    if pool is None:
        _start_worker(str(config_path), name, str(record_path), models)
    try:
        for first in range(0, count, batch):
            picked: List[Tuple[Case, Pick]] = []
            for index in range(first, min(count, first + batch)):
                slot = slots[index]
                if seeded is not None and index < int(input_count):
                    document, spawn, pick, slot = _input_mutant(
                        planner, course, seeded, index, "guided-%04d" % index)
                else:
                    document, spawn, pick = planner.build(
                        slot, index, "guided-%04d" % index)
                if spawn < 0:
                    spawn = random.Random("%d:spawn:%s" % (
                        seed, digest(document))).randrange(1 << 30)
                meta = CaseMeta(
                    index=index, digest=digest(document), spawn_seed=spawn,
                    field_seed=int(document.get("seed", 0)), band=slot.band,
                    target_success=round(slot.target_success, 6),
                    rho=float("%.6g" % slot.rho),
                    spacing=round(course.spacing(slot), 6),
                    ticks=int(document["ticks"]), arm=pick.operator,
                    region=("%s:%d" % (pick.model, pick.cell)
                            if pick.model else ""),
                    plan={"parent": pick.parent, "detail": pick.detail})
                picked.append((Case(meta, document), pick))
            batch_cases = [c for c, _ in picked]
            flown = (list(pool.map(_fly, batch_cases)) if pool is not None
                     else [_fly(c) for c in batch_cases])
            for (case, pick), (result, approaches) in zip(picked, flown):
                row = {"index": case.meta.index, "band": case.meta.band,
                       "operator": pick.operator, "model": pick.model,
                       "cell": pick.cell, "parent": pick.parent,
                       "detail": pick.detail,
                       "value": round(pick.value, 6),
                       "parent_distance": (None if pick.before is None
                                           else round(pick.before, 6)),
                       "verdict": result.verdict,
                       "crash_kind": result.crash_kind,
                       "mover": pick.mover}
                row["root"] = pick.root if pick.root >= 0 else \
                    case.meta.index
                if result.verdict != "error":
                    entry = Entry(case.meta.index, case.document,
                                  case.meta.spawn_seed, root=row["root"])
                    row.update(planner.update(pick, entry, result,
                                              approaches))
                results.append(result)
                cases.append(case)
                log.append(row)
                if progress:
                    print("  [%d/%d] %s b%s %-8s %-8s %s%s%s" % (
                        len(results), count, name, case.meta.band,
                        result.verdict, result.crash_kind or "-",
                        pick.operator,
                        (" -> %s:%d %s" % (pick.model, pick.cell,
                                           "HIT" if row.get("reached_target")
                                           else "miss"))
                        if pick.model else "",
                        (" NEW " + ",".join(row["new"]))
                        if row.get("new") else ""), flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    summary = {"record": name, "record_path": str(Path(record_path)
                                                  .resolve()),
               "seed": int(seed), "count": int(count), "batch": batch,
               "operators": list(operators), "uniform": bool(uniform),
               "greedy_steer": bool(greedy_steer),
               "dense": bool(dense), "lineage": bool(lineage),
               "tactics": bool(tactics), "tactic_blind": bool(tactic_blind),
               "mover_draw": mover_draw,
               "input": ({k: v for k, v in seeded.items()
                          if k not in ("document", "meta")}
                         if seeded is not None else None),
               "prior": float(prior), "evidence": bool(evidence),
               "seconds": round(time.time() - started, 1),
               "exposed": sorted("%s:%d" % t for t in planner.exposed),
               "table": planner.ops.table(), "log": log,
               "tactics": (course.sampler.table()
                           if course.sampler is not None else []),
               "spaces": {m: s.to_dict() for m, s in spaces.items()}}
    if out_dir:
        missions = os.path.join(out_dir, "missions-%s" % name)
        for case in cases:
            write_case(missions, case.document, case.meta)
        write_index(missions, [c.meta for c in cases],
                    header={"family": config.name, "seed": int(seed),
                            "count": int(count), "guided": name})
        write_results(os.path.join(out_dir, name), results, header={
            "family": config.name, "record": name,
            "record_path": summary["record_path"], "seed": int(seed),
            "guided": True, "every": 1, "spaces": summary["spaces"]})
        with open(os.path.join(out_dir, name, "guided.json"), "w") as out:
            json.dump(summary, out, indent=1)
    return results, summary


__all__ = ["Board", "EXPOSED", "Entry", "NEW_WEIGHT", "Operators", "PRIOR",
           "Pick", "Planner", "Touch", "run_guided", "slots_of"]
