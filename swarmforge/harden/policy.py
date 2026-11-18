from __future__ import annotations

import dataclasses
from array import array
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Dict, List, Mapping, Optional,
                    Sequence, Set, Tuple, Type)

from .candidate import Candidate
from .config import flag, number, read_options
from .operators import (PATTERN, Mutation, mutation_laws, names, operator,
                        recipes as all_recipes)

if TYPE_CHECKING:
    from .search import Session

PENDING = ("the property-aware mutation policy is not designed yet "
           "(swarmforge.harden.policy)")


@dataclass(frozen=True)
class Admission:
    admissible: bool = True
    weight: float = 1.0
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"admissible": self.admissible, "weight": self.weight,
                "reasons": list(self.reasons)}


class MutationPolicy:
    name = ""
    oversample = 1

    def __init__(self, options: Mapping[str, Any] = ()):
        self.options = dict(options)

    def propose(self, parent: Candidate, session: "Session",
                count: int) -> List[Mutation]:
        raise NotImplementedError

    def admit(self, candidate: Candidate, session: "Session") -> Admission:
        raise NotImplementedError

    def prefetch(self, parent: Candidate, session: "Session"):
        return None

    def rank(self, candidates: Sequence[Candidate], session: "Session",
             count: int) -> List[Candidate]:
        return list(candidates)[:max(count, 0)]

    def observe(self, candidate: Candidate) -> None:
        pass

    def exhausted(self, parent: Candidate, session: "Session") -> bool:
        return False

    def resume(self, session: "Session") -> None:
        for candidate in session.archive:
            self.observe(candidate)

    @staticmethod
    def _tried_on(session: "Session") -> Dict[str, Set[Tuple[Any, ...]]]:
        tried: Dict[str, Set[Tuple[Any, ...]]] = {}
        for candidate in session.archive:
            if candidate.mutation is not None and candidate.parent:
                tried.setdefault(candidate.parent, set()).add(
                    candidate.mutation.key())
        return tried


class PendingPolicy(MutationPolicy):
    name = "pending"

    def propose(self, parent, session, count):
        raise NotImplementedError(PENDING)

    def admit(self, candidate, session):
        raise NotImplementedError(PENDING)


class RandomPolicy(MutationPolicy):
    name = "random"

    def __init__(self, options: Mapping[str, Any] = ()):
        super().__init__(options)
        settings = read_options("harden.policy (random)", self.options,
                                {"pattern_share": number(float, 0.0, 1.0)})
        self.pattern_share: float = settings["pattern_share"]
        self._untried: Dict[str, Dict[str, array]] = {}
        self._counts: Dict[str, Dict[str, int]] = {}
        self._recipes: Tuple[str, Dict[str, List[Mutation]]] = ("", {})
        self._done: Dict[str, Set[Tuple[Any, ...]]] = {}

    def resume(self, session):
        super().resume(session)
        self._done = self._tried_on(session)

    def _enumerated(self, parent: Candidate, session: "Session"
                    ) -> Dict[str, List[Mutation]]:
        if self._recipes[0] != parent.id:
            document = session.document(parent)
            listed = {name: operator(name).recipes(document, session.context)
                      for name in names()}
            listed = {k: v for k, v in listed.items() if v}
            counts = {k: len(v) for k, v in listed.items()}
            known = self._counts.setdefault(parent.id, counts)
            if known != counts:
                raise RuntimeError("the recipes on %s changed between two "
                                   "enumerations: %s, now %s"
                                   % (parent.id, known, counts))
            self._recipes = (parent.id, listed)
        return self._recipes[1]

    def _pending(self, parent: Candidate, session: "Session"
                 ) -> Dict[str, array]:
        found = self._untried.get(parent.id)
        if found is None:
            done = self._done.get(parent.id, ())
            found = {name: array("I", (i for i, recipe in enumerate(recipes)
                                       if not done
                                       or recipe.key() not in done))
                     for name, recipes in self._enumerated(parent,
                                                           session).items()}
            self._untried[parent.id] = found
        return found

    def remaining(self, parent: Candidate, session: "Session") -> int:
        return sum(len(left) for left in self._pending(parent,
                                                       session).values())

    def exhausted(self, parent, session):
        found = self._untried.get(parent.id)
        return found is not None and not any(found.values())

    def propose(self, parent, session, count):
        pending = self._pending(parent, session)
        rng = session.rng
        out: List[Mutation] = []
        while len(out) < count:
            left = sorted(name for name, indices in pending.items() if indices)
            if not left:
                break
            patterns = [n for n in left if operator(n).family == PATTERN]
            others = [n for n in left if operator(n).family != PATTERN]
            if patterns and others:
                group = (patterns if rng.random() < self.pattern_share
                         else others)
            else:
                group = patterns or others
            name = group[rng.randrange(len(group))]
            indices = pending[name]
            k = rng.randrange(len(indices))
            indices[k], indices[-1] = indices[-1], indices[k]
            out.append(self._enumerated(parent, session)[name][indices.pop()])
        return out

    def admit(self, candidate, session):
        return Admission(True, 1.0, ("random policy: every valid variant",))


SPEED_PATTERNS = ("speed_gain", "anticipate")


def scaled(record: Any, law: str, factor: float) -> Any:
    return dataclasses.replace(record, terms=tuple(
        dataclasses.replace(term, weight=term.weight * factor)
        if term.name == law else term for term in record.terms))


@dataclass(frozen=True)
class LawNeed:
    stronger: float
    weaker: float
    direction: int
    priority: float

    def to_dict(self) -> Dict[str, Any]:
        return {"stronger": round(self.stronger, 8),
                "weaker": round(self.weaker, 8), "direction": self.direction,
                "priority": round(self.priority, 4)}


@dataclass(frozen=True)
class Diagnosis:
    parent: str
    laws: Mapping[str, LawNeed]
    velocity: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"laws": {law: need.to_dict()
                         for law, need in self.laws.items()},
                "velocity": self.velocity}


def _weighted(rng: Any, weights: Sequence[float]) -> int:
    total = float(sum(weights))
    if total <= 0.0:
        return rng.randrange(len(weights))
    point, running = rng.random() * total, 0.0
    for i, weight in enumerate(weights):
        running += weight
        if point < running:
            return i
    return len(weights) - 1


class GuidedPolicy(MutationPolicy):
    name = "guided"

    def __init__(self, options: Mapping[str, Any] = ()):
        super().__init__(options)
        settings = read_options("harden.policy (guided)", self.options, {
            "pattern_share": number(float, 0.0, 1.0),
            "explore": number(float, 0.0, 1.0),
            "screen": number(int, 1),
            "explore_admit": number(float, 0.0, 1.0),
            "probe_charged": number(int, 1),
            "probe_safe": number(int, 1),
            "learn": flag()})
        self.pattern_share: float = settings["pattern_share"]
        self.explore: float = settings["explore"]
        self.oversample: int = settings["screen"]
        self.explore_admit: float = settings["explore_admit"]
        self.probe_charged: int = settings["probe_charged"]
        self.probe_safe: int = settings["probe_safe"]
        self.learn: bool = settings["learn"]
        self._diagnoses: Dict[str, Diagnosis] = {}
        self._tried: Dict[str, Set[Tuple[Any, ...]]] = {}
        self._index: Tuple[str, Dict[Tuple[Any, ...], list], int] = (
            "", {}, 0)
        self._record: Dict[str, Tuple[int, int]] = {}


    def _probes(self, parent: Candidate, session: "Session"):
        return session.probe_set(parent, self.probe_charged, self.probe_safe)

    def prefetch(self, parent, session):
        if session.budget.left <= 0 or parent.weakness is None:
            return None
        return self._probes(parent, session)

    def diagnose(self, parent: Candidate, session: "Session") -> Diagnosis:
        found = self._diagnoses.get(parent.id)
        if found is not None:
            return found
        record = parent.validity.record if parent.validity else None
        if record is None:
            raise ValueError("%s has no parsed record to diagnose"
                             % parent.id)
        probes = self._probes(parent, session)
        laws = [term.name for term in record.terms]
        variants = {(law, factor): scaled(record, law, factor)
                    for factor in (2.0, 0.5) for law in laws}
        if session.budget.left > 0 and hasattr(session, "read_ahead"):
            session.read_ahead(list(variants.values()), probes)

        def gain(law: str, factor: float) -> float:
            if session.budget.left <= 0:
                return 0.0
            return session.probe(variants[(law, factor)], probes).gain

        stronger = {law: gain(law, 2.0) for law in laws}
        weaker = {law: gain(law, 0.5) for law in laws}
        worth = {law: max(stronger[law], weaker[law], 0.0) for law in laws}
        total = sum(worth.values())
        needs = {}
        for law in laws:
            share = worth[law] / total if total > 0.0 else 1.0 / len(laws)
            needs[law] = LawNeed(
                stronger[law], weaker[law],
                1 if stronger[law] >= weaker[law] else -1,
                (1.0 - self.explore) * share + self.explore / len(laws))
        diagnosis = Diagnosis(
            parent.id, needs,
            session.config.physics.plant.integration == "velocity")
        self._diagnoses[parent.id] = diagnosis
        session.annotate(parent, diagnosis=diagnosis.to_dict())
        return diagnosis

    def _recipes(self, parent: Candidate, session: "Session"
                 ) -> Tuple[Dict[Tuple[Any, ...], list], int]:
        if self._index[0] != parent.id:
            document = session.document(parent)
            index: Dict[Tuple[Any, ...], list] = {}
            every = all_recipes(document, session.context)
            for mutation in every:
                found = operator(mutation.operator)
                key = mutation.key()
                for law in mutation_laws(mutation, document):
                    kind = ((law, PATTERN, found.direction(mutation))
                            if found.family == PATTERN else (law, "syntax"))
                    index.setdefault(kind, []).append((key, mutation))
                index.setdefault(("any",), []).append((key, mutation))
            self._index = (parent.id, index, len(every))
        return self._index[1], self._index[2]


    def propose(self, parent, session, count):
        need = self.diagnose(parent, session)
        index, _total = self._recipes(parent, session)
        tried = self._tried.setdefault(parent.id, set())
        rng = session.rng
        laws = sorted(need.laws)
        priorities = [need.laws[law].priority for law in laws]
        out: List[Mutation] = []
        for _attempt in range(64 * max(count, 1)):
            if len(out) >= count:
                break
            law = laws[_weighted(rng, priorities)]
            if rng.random() < self.pattern_share:
                pool = [row for row in index.get(
                            (law, PATTERN, need.laws[law].direction), ())
                        if row[0] not in tried and (
                            need.velocity
                            or row[1].operator not in SPEED_PATTERNS)]
            else:
                pool = [row for row in index.get((law, "syntax"), ())
                        if row[0] not in tried]
            if pool:
                key, mutation = self._pick(pool, rng)
                tried.add(key)
                out.append(mutation)
        if len(out) < count:
            rest = [row for row in index.get(("any",), ())
                    if row[0] not in tried]
            while rest and len(out) < count:
                key, mutation = rest.pop(rng.randrange(len(rest)))
                tried.add(key)
                out.append(mutation)
        return out

    def _pick(self, pool: list, rng: Any) -> Tuple[Any, Mutation]:
        by_operator: Dict[str, list] = {}
        for row in pool:
            by_operator.setdefault(row[1].operator, []).append(row)
        operators = sorted(by_operator)
        weights = [self.success(name) for name in operators] if self.learn \
            else [1.0] * len(operators)
        rows = by_operator[operators[_weighted(rng, weights)]]
        return rows[rng.randrange(len(rows))]

    def success(self, name: str) -> float:
        kept, scored = self._record.get(name, (0, 0))
        return (kept + 1.0) / (scored + 2.0)

    def admit(self, candidate, session):
        if session.budget.left <= 0:
            return Admission(False, 0.0, ("no budget left to probe it",))
        parent = session.archive.get(candidate.parent)
        reading = session.probe(candidate.validity.record,
                                self._probes(parent, session))
        laws: Tuple[str, ...] = ()
        direction = 0
        if candidate.mutation is not None:
            laws = mutation_laws(candidate.mutation, session.document(parent))
            direction = operator(candidate.mutation.operator).direction(
                candidate.mutation)
        session.annotate(candidate, guide=dict(
            reading.to_dict(), laws=list(laws), direction=direction))
        return Admission(True, reading.gain,
                         ("predicted gain %+.6g" % reading.gain,))

    def rank(self, candidates, session, count):
        if count <= 0 or not candidates:
            return []

        def predicted(candidate: Candidate) -> float:
            return float(candidate.annotations.get("guide", {})
                         .get("gain", 0.0))

        ordered = sorted(candidates, key=lambda c: -predicted(c))
        explore = min(int(round(self.explore_admit * count)), count)
        best = [c for c in ordered if predicted(c) > 0.0][:count - explore]
        chosen = {c.id for c in best}
        rest = [c for c in ordered if c.id not in chosen]
        picks: List[Candidate] = []
        while rest and len(picks) < explore:
            picks.append(rest.pop(session.rng.randrange(len(rest))))
        for candidate in best:
            candidate.annotations["guide"]["chosen"] = "predicted"
        for candidate in picks:
            candidate.annotations["guide"]["chosen"] = "explore"
        return best + picks

    def observe(self, candidate):
        if not self.learn or candidate.mutation is None \
                or candidate.stage != "measured":
            return
        kept, scored = self._record.get(candidate.mutation.operator, (0, 0))
        self._record[candidate.mutation.operator] = (
            kept + int(bool(candidate.annotations.get("accepted"))),
            scored + 1)

    def exhausted(self, parent, session):
        if self._index[0] != parent.id:
            return False
        return len(self._tried.get(parent.id, ())) >= self._index[2]

    def resume(self, session):
        super().resume(session)
        self._tried = self._tried_on(session)
        if (session.resumed or {}).get("rescored"):
            return
        for candidate in session.archive:
            found = candidate.annotations.get("diagnosis")
            if not found or not candidate.measured:
                continue
            self._diagnoses[candidate.id] = Diagnosis(
                candidate.id,
                {law: LawNeed(float(need["stronger"]), float(need["weaker"]),
                              int(need["direction"]),
                              float(need["priority"]))
                 for law, need in found["laws"].items()},
                bool(found["velocity"]))


POLICIES: Dict[str, Type[MutationPolicy]] = {
    PendingPolicy.name: PendingPolicy, RandomPolicy.name: RandomPolicy,
    GuidedPolicy.name: GuidedPolicy}


def build_policy(name: str, options: Mapping[str, Any] = ()
                 ) -> MutationPolicy:
    found = POLICIES.get(name)
    if found is None:
        raise ValueError("no mutation policy %r (have: %s)"
                         % (name, ", ".join(sorted(POLICIES))))
    return found(options)


__all__ = ["Admission", "Diagnosis", "GuidedPolicy", "LawNeed",
           "MutationPolicy", "PENDING", "POLICIES", "PendingPolicy",
           "RandomPolicy", "SPEED_PATTERNS", "build_policy", "scaled"]
