from __future__ import annotations

import os
import pickle
import random
import resource
import shutil
import tempfile
import time

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..solve.config import FamilyConfig
from ..solve.record import Record, release_sympy_caches
from .archive import Archive
from .candidate import ORIGINAL, Candidate
from .config import HardenConfig
from .document import (Document, canonical, digest, fixed_changes,
                       interface_change, load)
from .domain import Domain, domains_for
from .operators import Mutation, OperatorContext, apply
from .probe import ProbeReading, ProbeSet, draw as draw_probe
from .probe import read as read_probe
from .properties import (PROPERTIES, CommandScale, Profiler, Property,
                         PropertyProfile, compare)
from .screening import (Screen, screen as screen_variant, worker_probe,
                        worker_screen)
from .units import expressions
from .validate import check
from .weakness import Evaluator, deltas, score_of, transitions

CHECKPOINT_SECONDS = 1800.0


def tree_cpu() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    gone = resource.getrusage(resource.RUSAGE_CHILDREN)
    total = own.ru_utime + own.ru_stime + gone.ru_utime + gone.ru_stime
    try:
        parents = {}
        for entry in os.listdir("/proc"):
            if entry.isdigit():
                try:
                    with open("/proc/%s/stat" % entry) as handle:
                        fields = handle.read().rsplit(")", 1)[1].split()
                except OSError:
                    continue
                parents[int(entry)] = (int(fields[1]), fields)
    except OSError:
        return total
    tick = float(os.sysconf("SC_CLK_TCK"))
    below, frontier = set(), [os.getpid()]
    while frontier:
        pid = frontier.pop()
        for child, (ppid, _fields) in parents.items():
            if ppid == pid and child not in below:
                below.add(child)
                frontier.append(child)
    for child in below:
        fields = parents[child][1]
        total += sum(int(v) for v in fields[11:15]) / tick
    return total


@dataclass
class Budget:
    max_evaluations: int
    used: int = 0
    probed: float = 0.0

    @property
    def left(self) -> int:
        return max(0, self.max_evaluations - self.used
                   - int(self.probed + 1e-9))

    @property
    def exhausted(self) -> bool:
        return self.left == 0

    def to_dict(self) -> Dict[str, Any]:
        return {"max_evaluations": self.max_evaluations, "used": self.used,
                "probed": round(self.probed, 3)}


@dataclass
class Round:
    index: int
    label: str = ""
    stages: Dict[str, int] = field(default_factory=dict)
    retained: int = 0
    best_total: Optional[float] = None
    best_scores: Dict[str, float] = field(default_factory=dict)
    evaluations: int = 0
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"round": self.index, "label": self.label,
                "stages": dict(self.stages), "retained": self.retained,
                "best_total": self.best_total,
                "best_scores": dict(self.best_scores),
                "evaluations": self.evaluations,
                "seconds": round(self.seconds, 3)}


class Session:
    def __init__(self, config: FamilyConfig, harden: HardenConfig, *,
                 record_path=None, out_dir=None,
                 progress: bool = True, resume=None):
        self.started_at = time.time()
        self.cpu_started = tree_cpu()
        self.config = config
        self.harden = harden
        self.rng_seed: Any = int(harden.seed)
        self.rng = random.Random(self.rng_seed)
        self.out_dir = None if out_dir is None else Path(out_dir)
        if resume is not None:
            from .resume import check_folders
            check_folders(resume, self.out_dir)
        self.progress = progress
        self.record_path = Path(record_path or config.record_path).resolve()
        self.closed = False
        self.seconds = 0.0

        self.original: Document = canonical(load(self.record_path))
        from ..solve.record import parse_record
        record = parse_record(self.original, self.record_path,
                              config.physics.record_params)
        self.domains: Dict[str, Domain] = domains_for(record, config)
        self.properties: Tuple[Property, ...] = tuple(PROPERTIES)
        self.scale = CommandScale.of(record, config)
        self.profiler = Profiler(self.domains, self.properties, self.scale)
        validity = check(self.original, config=config, domains=self.domains,
                         label=ORIGINAL)
        if not validity.ok:
            raise ValueError("the original controller %s fails the validity "
                             "screen (%s): %s" % (self.record_path,
                                                  validity.check,
                                                  validity.reason))
        self.reference: PropertyProfile = self.profiler.profile(
            validity.record, ORIGINAL)
        self.context = OperatorContext(config, self.domains, self.original,
                                       self.reference, self.scale)
        self.archive = Archive()
        self.budget = Budget(int(harden.max_evaluations))
        self.rounds: List[Round] = []
        self._round_from = 0
        self._round_started = time.time()
        self._round_used = 0
        self._next = 0
        self._screened: Dict[Tuple[str, Tuple[Any, ...]], Candidate] = {}
        self._probes: Dict[Tuple[str, int, int], ProbeSet] = {}
        self._prescreened: Dict[str, Screen] = {}
        self._applied: Dict[Tuple[str, Tuple[Any, ...]], Any] = {}
        self._probe_hits: Dict[int, Tuple[Record, ProbeSet, Any]] = {}
        self._probe_files: Dict[int, Tuple[ProbeSet, str]] = {}
        self._scratch: Optional[str] = None
        self._checkpoint_at = time.time()
        self.evaluator = Evaluator(config, harden.objective, harden.workers,
                                   scoring=harden.scoring, original=record)

        baseline = Candidate(
            id=ORIGINAL, digest=digest(self.original), stage="screened",
            validity=validity, profile=self.reference,
            comparison=compare(self.reference, self.reference,
                               self.properties),
            expressions=expressions(validity.units),
            created_at=self.started_at)
        self.archive.add(baseline, self.original)
        self._measure([baseline], charge=False)
        if baseline.total is None:
            raise ValueError("the original controller could not be measured "
                             "on every model of the objective: %s"
                             % baseline.weakness.errors())
        self.calibration: Optional[Dict[str, Any]] = None
        if harden.calibrate is not None:
            from .calibrate import calibrate
            try:
                self.harden, self.calibration = calibrate(
                    config, self.record_path, harden,
                    {m: baseline.weakness.score(m)
                     for m in harden.objective.models}, self.out_dir)
            except BaseException:
                self.evaluator.close()
                raise
            baseline.score = score_of(baseline.weakness, self.harden.objective)
            if self.progress:
                print("  calibrated on %d missions (seed %d): weights %s" % (
                    self.calibration["flown"], harden.calibrate.seed,
                    ", ".join("%s %.4g" % kv for kv in
                              self.calibration["weights"].items())))
        self.baseline = baseline
        if self.progress:
            print("  %-9s %s" % (baseline.id, self.line(baseline)))
        self.resumed: Optional[Dict[str, Any]] = None
        if resume is not None:
            from .resume import carry
            try:
                self.resumed = carry(self, resume)
            except BaseException:
                self.evaluator.close()
                raise
            if self.progress:
                best = self.archive.best()
                print("  resumed %s: %d candidates (%s), pool %d, best %s; "
                      "%d evaluations and %.1f of probes spent of %d"
                      % (resume, self.resumed["candidates"], ", ".join(
                          "%s %d" % kv for kv in
                          sorted(self.resumed["stages"].items())),
                         self.resumed["pool"],
                         "-" if best is None else "%.6f" % best.total,
                         self.budget.used, self.budget.probed,
                         self.budget.max_evaluations))


    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.evaluator.close()
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None
        self.write()

    def write(self) -> None:
        self.seconds = time.time() - self.started_at
        self._checkpoint_at = time.time()
        if self.out_dir is not None:
            from . import store
            store.write_all(self.out_dir, self)


    def _id(self) -> str:
        self._next += 1
        return "cand-%04d" % self._next

    def document(self, candidate: Candidate) -> Document:
        return self.archive.document(candidate)

    def consider(self, document: Mapping[str, Any], *,
                 parent: Optional[Candidate] = None,
                 mutation: Optional[Mutation] = None, source: str = "",
                 label: Optional[str] = None,
                 fixed: bool = True) -> Candidate:
        parent = parent or self.baseline
        document = (canonical(document) if mutation is None
                    else dict(document))
        candidate = Candidate(
            id=label or self._id(), parent=parent.id,
            parent_digest=parent.digest, generation=parent.generation + 1,
            mutation=mutation, source=source, round=len(self.rounds) + 1)
        problem = interface_change(self.original, document, fixed=fixed)
        if problem is not None:
            candidate.stage, candidate.note = "refused", ("interface "
                                                          "changed: %s"
                                                          % problem)
            return self._file(candidate, document)
        if not fixed:
            candidate.note = "; ".join(fixed_changes(self.original, document))
        candidate.digest = digest(document)
        twin = self.archive.holding(candidate.digest)
        if twin is not None:
            candidate.stage, candidate.same_as = "duplicate", twin.id
            candidate.note = "the same controller as %s" % twin.id
            return self._file(candidate, document)
        found = self._prescreened.pop(candidate.digest, None)
        if found is None or found.label != candidate.id:
            found = screen_variant(
                document, label=candidate.id, config=self.config,
                domains=self.domains, profiler=self.profiler,
                reference=self.reference, properties=self.properties)
        elif found.probe is not None and found.validity is not None:
            self._probe_hits[id(found.validity.record)] = (
                found.validity.record, found.probe[0], found.probe[1])
        if found.failed is not None and found.failed[0] == "validity":
            return self._unreadable(candidate, document, *found.failed)
        validity = found.validity
        candidate.validity = validity
        if found.expressions is not None:
            candidate.expressions = found.expressions
        if not validity.ok:
            candidate.stage = "invalid"
            candidate.note = "; ".join(filter(None, (
                "%s: %s" % (validity.check, validity.reason),
                candidate.note)))
            return self._file(candidate, document)
        if found.failed is not None and found.failed[0] == "properties":
            candidate.profile = None
            return self._unreadable(candidate, document, *found.failed)
        candidate.profile = found.profile
        candidate.comparison = found.comparison
        candidate.stage = "screened"
        return self._file(candidate, document)

    def _unreadable(self, candidate: Candidate, document: Document,
                    step: str, kind: str, message: str) -> Candidate:
        candidate.stage = "invalid"
        candidate.note = "; ".join(filter(None, (
            "%s failed: %s: %s" % (step, kind, message),
            candidate.note)))
        return self._file(candidate, document)

    def propose(self, parent: Candidate, mutation: Mutation) -> Candidate:
        key = (parent.digest, mutation.key())
        known = self._screened.get(key)
        if known is not None:
            return known
        applied = self._applied.pop(key, None)
        mutated, reason = (applied if applied is not None else
                           apply(mutation, self.document(parent),
                                 self.context))
        if mutated is None:
            candidate = Candidate(
                id=self._id(), parent=parent.id, parent_digest=parent.digest,
                generation=parent.generation + 1, mutation=mutation,
                stage="refused", note=reason, round=len(self.rounds) + 1)
            self._file(candidate, None)
        else:
            candidate = self.consider(mutated, parent=parent,
                                      mutation=mutation)
        self._screened[key] = candidate
        return candidate

    def prescreen(self, parent: Candidate, mutations: Sequence[Mutation],
                  probes: Optional[ProbeSet] = None) -> int:
        self._prescreened.clear()
        self._applied.clear()
        self._probe_hits.clear()
        if self.evaluator.workers <= 1 \
                or os.environ.get("SWARMFORGE_PRESCREEN", "1") == "0":
            return 0
        document = self.document(parent)
        tasks: List[Tuple[str, str, Document]] = []
        seen, digests, taken = set(), set(), self._next
        for mutation in mutations:
            key = (parent.digest, mutation.key())
            if key in self._screened or key in seen:
                continue
            seen.add(key)
            taken += 1
            applied = self._applied[key] = apply(mutation, document,
                                                 self.context)
            if applied[0] is None:
                continue
            variant = dict(applied[0])
            if interface_change(self.original, variant, fixed=True):
                continue
            found = digest(variant)
            if found in digests or self.archive.holding(found) is not None:
                continue
            digests.add(found)
            tasks.append((found, "cand-%04d" % taken, variant))
        if not tasks:
            return 0
        spec = {"record_path": str(self.record_path)}
        path = None if probes is None else self._probe_file(probes)
        try:
            screens = self.evaluator.map(
                worker_screen, [(spec, label, variant, path)
                                for _digest, label, variant in tasks])
        except Exception:
            return 0
        for (found, _label, _variant), result in zip(tasks, screens):
            if result.probe is not None:
                result.probe = (probes, result.probe)
            self._prescreened[found] = result
        return len(tasks)

    def read_ahead(self, records: Sequence[Record], probes: ProbeSet) -> int:
        if not records or self.evaluator.workers <= 1 \
                or os.environ.get("SWARMFORGE_PRESCREEN", "1") == "0":
            return 0
        path = self._probe_file(probes)
        try:
            readings = self.evaluator.map(worker_probe,
                                          [(record, path)
                                           for record in records])
        except Exception:
            return 0
        for record, reading in zip(records, readings):
            self._probe_hits[id(record)] = (record, probes, reading)
        return len(records)

    def _probe_file(self, probes: ProbeSet) -> str:
        held = self._probe_files.get(id(probes))
        if held is not None and held[0] is probes:
            return held[1]
        if self._scratch is None:
            self._scratch = tempfile.mkdtemp(prefix="swarmforge-probes-")
        for _probes, old in self._probe_files.values():
            try:
                os.remove(old)
            except OSError:
                pass
        self._probe_files.clear()
        self._probe_count = getattr(self, "_probe_count", 0) + 1
        path = os.path.join(self._scratch,
                            "probes-%d.pkl" % self._probe_count)
        with open(path, "wb") as handle:
            pickle.dump(probes, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._probe_files[id(probes)] = (probes, path)
        return path

    def _file(self, candidate: Candidate,
              document: Optional[Document]) -> Candidate:
        release_sympy_caches()
        self.archive.add(candidate, document)
        if self.out_dir is not None and document is not None:
            from . import store
            candidate.path = store.write_candidate(self.out_dir, document,
                                                   candidate)
        return candidate


    def measure(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        todo = list({c.id: c for c in candidates if c.stage == "screened"
                     and c.id != ORIGINAL}.values())
        todo = todo[:self.budget.left]
        return self._measure(todo, charge=True)

    def _measure(self, todo: List[Candidate], *,
                 charge: bool) -> List[Candidate]:
        if not todo:
            return []
        readings = self.evaluator.measure(
            [(c.id, self.document(c)) for c in todo],
            [c.validity.record if c.validity else None for c in todo])
        for candidate, reading in zip(todo, readings):
            candidate.weakness = reading
            candidate.score = score_of(reading, self.harden.objective)
            candidate.stage = "measured"
            if not candidate.score.complete:
                candidate.note = "; ".join("%s: %s" % item for item in
                                           reading.errors().items())
            if charge:
                self.budget.used += 1
            if self.out_dir is not None:
                from . import store
                candidate.path = store.write_candidate(
                    self.out_dir, self.document(candidate), candidate)
            if self.progress and candidate.id != ORIGINAL:
                print("  %-9s %s" % (candidate.id, self.line(candidate)))
        return todo


    def probe_set(self, parent: Candidate, charged: int,
                  safe: int) -> ProbeSet:
        key = (parent.id, int(charged), int(safe))
        found = self._probes.get(key)
        if found is None:
            rng = np.random.default_rng(
                [int(self.harden.seed), int(parent.digest or "0", 16)])
            found = self._probes[key] = draw_probe(
                parent, self.evaluator.banks, self.harden.objective.weights,
                charged, safe, rng, self.evaluator.state_weights)
        return found

    def probe(self, record: Record, probes: ProbeSet) -> ProbeReading:
        hit = self._probe_hits.pop(id(record), None)
        if hit is not None and hit[0] is record and hit[1] is probes:
            reading = hit[2]
        else:
            reading = read_probe(probes, record, self.config)
        banks = sum(int(b.state.size) for b in self.evaluator.banks.values())
        self.budget.probed += probes.size / max(banks, 1)
        return reading


    def retain(self, candidate: Candidate, retained: bool = True) -> None:
        candidate.retained = bool(retained)

    def annotate(self, candidate: Candidate, **values: Any) -> None:
        candidate.annotations.update(values)

    def mark_round(self, label: str = "") -> Round:
        made = self.archive.candidates[self._round_from:]
        stages: Dict[str, int] = {}
        for candidate in made:
            if candidate.id == ORIGINAL:
                continue
            stages[candidate.stage] = stages.get(candidate.stage, 0) + 1
        best = self.archive.best()
        summary = Round(
            index=len(self.rounds) + 1, label=label, stages=stages,
            retained=sum(1 for c in made if c.retained),
            best_total=None if best is None else best.total,
            best_scores=dict(self.archive.best_scores()),
            evaluations=self.budget.used - self._round_used,
            seconds=time.time() - self._round_started)
        self.rounds.append(summary)
        self._round_from = len(self.archive.candidates)
        self._round_started = time.time()
        self._round_used = self.budget.used
        if (self.out_dir is not None and time.time() - self._checkpoint_at
                >= CHECKPOINT_SECONDS):
            self.write()
        return summary


    def versus(self, candidate: Candidate,
               other: Optional[Candidate] = None) -> Dict[str, float]:
        other = other or (self.archive.get(candidate.parent)
                          if candidate.parent else None)
        if other is None or candidate.score is None or other.score is None:
            return {}
        return deltas(candidate.score, other.score)

    def transitions(self, candidate: Candidate, model: str,
                    other: Optional[Candidate] = None) -> Dict[str, int]:
        other = other or self.baseline
        return transitions(other.weakness.models[model],
                           candidate.weakness.models[model])

    def line(self, candidate: Candidate) -> str:
        parts = []
        for name in self.harden.objective.models:
            value = candidate.score.of(name) if candidate.score else None
            parts.append("%s=%s" % (name[:8], "-" if value is None
                                    else "%.5f" % value))
        label = (candidate.mutation.label() if candidate.mutation
                 else candidate.source or "(original)")
        changed = (candidate.comparison.summary()
                   if candidate.comparison else "")
        return "%-44s %s  total %s%s" % (
            label[:44], " ".join(parts),
            "-" if candidate.total is None else "%.6f" % candidate.total,
            "  [%s]" % changed[:80] if changed else "")


__all__ = ["Budget", "Round", "Session"]
