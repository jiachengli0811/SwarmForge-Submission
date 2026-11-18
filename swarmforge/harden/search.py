from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Dict, List, Mapping, Optional, Sequence, Set,
                    Tuple, Type)

from ..solve.config import read_config
from .candidate import Candidate
from .config import Component, Reader, number, read_harden, read_options
from .draws import boltzmann, top_k
from .policy import MutationPolicy, build_policy
from .session import Session
from .scoring import paired_variance
from .weakness import transitions

PENDING = ("the hardening search loop is not designed yet "
           "(swarmforge.harden.search)")


class SearchStrategy:
    name = ""

    def __init__(self, policy: MutationPolicy,
                 options: Mapping[str, Any] = ()):
        self.policy = policy
        self.options = dict(options)

    def run(self, session: Session) -> None:
        raise NotImplementedError


class PendingStrategy(SearchStrategy):
    name = "pending"

    def run(self, session: Session) -> None:
        raise NotImplementedError(PENDING)


def property_changes(candidate: Candidate) -> int:
    comparison = candidate.comparison
    if comparison is None:
        return 0
    return sum(1 for _law, _facet, change in comparison.changes()
               if change.shifted or change.changed is None)


def property_drift(candidate: Candidate) -> float:
    comparison = candidate.comparison
    if comparison is None:
        return 0.0
    return sum(change.drift for _law, _facet, change in comparison.changes()
               if change.drift is not None and not change.shifted)


def property_energy(candidate: Candidate) -> float:
    drift = property_drift(candidate)
    return property_changes(candidate) + drift / (1.0 + drift)


def paired_noise(child: Candidate, parent: Candidate,
                 weights: Mapping[str, float]) -> Optional[float]:
    variance = 0.0
    for name, weight in weights.items():
        before = parent.weakness.models.get(name) if parent.weakness else None
        after = child.weakness.models.get(name) if child.weakness else None
        if before is None or after is None or not before.codes \
                or not after.codes or not before.samples:
            return None
        a, b = after.reach(), before.reach()
        if a is not None and b is not None:
            variance += weight ** 2 * paired_variance(a[0], a[1], b[0], b[1])
            continue
        moved = transitions(before, after)
        variance += ((weight / before.samples) ** 2
                     * (moved["repaired"] + moved["broken"]))
    return math.sqrt(variance)


def state_unit(session: Session) -> float:
    weights = session.harden.objective.weights
    units = [weights[name] / reading.samples
             for name, reading in session.baseline.weakness.models.items()
             if weights.get(name, 0.0) > 0.0 and reading.samples > 0]
    return min(units) if units else 0.0


BANDED_OPTIONS: Mapping[str, Reader] = {
    "children": number(int, 1),
    "proposals": number(int, 1),
    "weakness_temperature": number(float, 0.0),
    "band": number(float, 0.0),
    "floor": number(float, 0.0),
    "property_temperature": number(float, 0.0),
    "decay": number(float, 0.0, 1.0, above=True),
    "min_improvement": number(float, 0.0),
    "significance": number(float, 0.0),
}


@dataclass(frozen=True)
class BandedSettings:
    children: int
    proposals: int
    weakness_temperature: float
    band: float
    floor: float
    property_temperature: float
    decay: float
    min_improvement: float
    significance: float

    @classmethod
    def read(cls, options: Mapping[str, Any]) -> "BandedSettings":
        return cls(**read_options("harden.strategy (banded)", options,
                                  BANDED_OPTIONS))


def anchor_weights(totals: Sequence[float], idle: Sequence[int],
                   settings: BandedSettings, floor: float) -> List[float]:
    temperature = settings.weakness_temperature * min(totals) + floor
    return [weight * settings.decay ** quiet
            for weight, quiet in zip(boltzmann(totals, temperature), idle)]


def comparable(totals: Sequence[float], anchor: float,
               settings: BandedSettings, floor: float) -> List[int]:
    width = settings.band * anchor + floor
    return [i for i, total in enumerate(totals)
            if abs(total - anchor) <= width]


def parent_weights(energies: Sequence[float], idle: Sequence[int],
                   settings: BandedSettings) -> List[float]:
    return [weight * settings.decay ** quiet
            for weight, quiet in zip(boltzmann(energies,
                                               settings.property_temperature),
                                     idle)]


def verdict(child_total: Optional[float], child_changes: int,
            parent_total: Optional[float], parent_changes: int,
            epsilon: float, noise: Optional[float] = 0.0,
            significance: float = 0.0) -> str:
    if child_total is None or parent_total is None:
        return ""
    gain = parent_total - child_total
    bar = max(epsilon, significance * (noise or 0.0))
    if gain > 0.0 and gain >= bar * (1.0 - 1e-9):
        return "improves"
    if gain >= -1e-12 and child_changes < parent_changes:
        return "repairs"
    return ""


def expand(session: Session, policy: MutationPolicy, parent: Candidate,
           children: int, proposals: int
           ) -> Tuple[List[Candidate], List[Candidate], bool]:
    want = min(children, session.budget.left)
    target = want * max(int(policy.oversample), 1)
    cap = proposals * target
    made: List[Candidate] = []
    admitted: List[Candidate] = []
    exhausted = False
    while len(admitted) < target and len(made) < cap:
        asked = min(target - len(admitted), cap - len(made))
        recipes = policy.propose(parent, session, asked)
        session.prescreen(parent, recipes, policy.prefetch(parent, session))
        for recipe in recipes:
            child = session.propose(parent, recipe)
            made.append(child)
            if child.stage != "screened":
                continue
            admission = policy.admit(child, session)
            session.annotate(child, admission=admission.to_dict())
            if admission.admissible:
                admitted.append(child)
        if len(recipes) < asked:
            exhausted = True
            break
    want = min(want, session.budget.left)
    return made, policy.rank(admitted, session, want), exhausted


class BandedStrategy(SearchStrategy):
    name = "banded"

    def __init__(self, policy: MutationPolicy,
                 options: Mapping[str, Any] = ()):
        super().__init__(policy, options)
        self.settings = BandedSettings.read(self.options)

    def run(self, session: Session) -> None:
        _BandedRun(session, self.policy, self.settings).run()


class _BandedRun:
    def __init__(self, session: Session, policy: MutationPolicy,
                 settings: BandedSettings):
        self.session = session
        self.policy = policy
        self.settings = settings
        unit = state_unit(session)
        self.floor = settings.floor * unit
        self.epsilon = settings.min_improvement * unit
        self.pool: List[Candidate] = [session.baseline] + [
            c for c in session.archive
            if c.retained and c.measured and c.id != session.baseline.id]
        session.retain(session.baseline)
        session.annotate(session.baseline, accepted="original")
        self.idle: Dict[str, int] = {
            c.id: int(c.annotations["idle"]) for c in self.pool
            if "idle" in c.annotations}
        self.expansions: Dict[str, int] = {
            c.id: int(c.annotations["expansions"]) for c in self.pool
            if "expansions" in c.annotations}
        self.exhausted: Set[str] = set()

    def live(self) -> List[Candidate]:
        return [c for c in self.pool if c.id not in self.exhausted
                and not self.policy.exhausted(c, self.session)]

    def select(self) -> Optional[Tuple[Candidate, Dict[str, Any]]]:
        live = self.live()
        if not live:
            return None
        rng = self.session.rng
        totals = [c.total for c in live]
        idle = [self.idle.get(c.id, 0) for c in live]
        first = anchor_weights(totals, idle, self.settings, self.floor)
        a = top_k(list(range(len(live))), first, 1, rng)[0]
        close = comparable(totals, totals[a], self.settings, self.floor)
        energies = [property_energy(live[i]) for i in close]
        second = parent_weights(energies, [idle[i] for i in close],
                                self.settings)
        j = top_k(list(range(len(close))), second, 1, rng)[0]
        parent = live[close[j]]
        drawn = {
            "round": len(self.session.rounds) + 1, "anchor": live[a].id,
            "p_anchor": round(first[a] / sum(first), 6),
            "comparable": len(close),
            "p_parent": round(second[j] / sum(second), 6),
            "energy": round(energies[j], 6), "idle": idle[close[j]]}
        return parent, drawn

    def expand(self, parent: Candidate
               ) -> Tuple[List[Candidate], List[Candidate]]:
        made, chosen, exhausted = expand(self.session, self.policy, parent,
                                         self.settings.children,
                                         self.settings.proposals)
        if exhausted:
            self.exhausted.add(parent.id)
        return made, chosen

    def run(self) -> None:
        session = self.session
        while not session.budget.exhausted:
            picked = self.select()
            if picked is None:
                break
            parent, drawn = picked
            made, admitted = self.expand(parent)
            measured = session.measure(admitted)
            changes = property_changes(parent)
            kept = {"improves": 0, "repairs": 0}
            weights = session.harden.objective.weights
            for child in measured:
                child_changes = property_changes(child)
                noise = paired_noise(child, parent, weights)
                why = verdict(child.total, child_changes, parent.total,
                              changes, self.epsilon, noise,
                              self.settings.significance)
                session.annotate(
                    child, accepted=why, changes=child_changes,
                    delta=(None if child.total is None
                           else child.total - parent.total),
                    noise=noise)
                if why:
                    self.pool.append(child)
                    session.retain(child)
                    kept[why] += 1
            self.expansions[parent.id] = self.expansions.get(parent.id, 0) + 1
            self.idle[parent.id] = (0 if any(kept.values())
                                    else self.idle.get(parent.id, 0) + 1)
            drawn.update(made=len(made), measured=len(measured), **kept)
            parent.annotations.setdefault("draws", []).append(drawn)
            session.annotate(parent, expansions=self.expansions[parent.id],
                             idle=self.idle[parent.id])
            for child in made:
                self.policy.observe(child)
            round_ = session.mark_round("expand %s" % parent.id)
            if session.progress:
                print("round %d: %s (total %.6f, %d changed, drawn p=%.2f "
                      "of %d comparable) -> %d measured, %d improve, %d "
                      "repair; pool %d, best %.6f"
                      % (round_.index, parent.id, parent.total, changes,
                         drawn["p_parent"], drawn["comparable"],
                         len(measured), kept["improves"], kept["repairs"],
                         len(self.pool), round_.best_total
                         if round_.best_total is not None else float("nan")))


STRATEGIES: Dict[str, Type[SearchStrategy]] = {
    PendingStrategy.name: PendingStrategy,
    BandedStrategy.name: BandedStrategy}


def build_strategy(component: Component,
                   policy: MutationPolicy) -> SearchStrategy:
    found = STRATEGIES.get(component.name)
    if found is None:
        raise ValueError("no search strategy %r (have: %s)"
                         % (component.name, ", ".join(sorted(STRATEGIES))))
    return found(policy, component.options)


def run(config_path, *, record_path=None, out_dir=None,
        overrides: Optional[Mapping[str, Any]] = None,
        progress: bool = True, resume=None) -> Session:
    config = read_config(Path(config_path))
    harden = read_harden(Path(config_path), overrides)
    policy = build_policy(harden.policy.name, harden.policy.options)
    strategy = build_strategy(harden.strategy, policy)
    session = Session(config, harden, record_path=record_path,
                      out_dir=out_dir, progress=progress, resume=resume)
    try:
        if session.resumed is not None:
            policy.resume(session)
        strategy.run(session)
    finally:
        session.close()
    return session


__all__ = ["BANDED_OPTIONS", "BandedSettings", "BandedStrategy", "PENDING",
           "PendingStrategy", "STRATEGIES", "SearchStrategy",
           "anchor_weights", "build_strategy", "comparable", "expand",
           "paired_noise", "parent_weights", "property_changes",
           "property_drift", "property_energy", "run", "state_unit",
           "verdict"]
