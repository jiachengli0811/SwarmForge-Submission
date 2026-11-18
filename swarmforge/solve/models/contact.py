from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..bank import Bank, bank_for
from ..config import FamilyConfig, ModelSpec
from ..dynamics import Simulator, World
from ..reach import Reach
from ..record import Record, read_record
from ..state import Sampler, State, admissible
from ..verdict import CHARGED, Verdict


@dataclass
class Outcome:
    verdict: np.ndarray
    tick: np.ndarray
    drone: np.ndarray
    other: np.ndarray
    clearance: np.ndarray
    positions: Optional[np.ndarray] = None

    def count(self, verdict: Verdict) -> int:
        return int(np.sum(self.verdict == int(verdict)))

    def fractions(self) -> Dict[str, float]:
        n = max(1, self.verdict.size)
        return {v.name: self.count(v) / n for v in Verdict}

    def score(self) -> float:
        n = max(1, self.verdict.size)
        return sum(self.count(v) for v in CHARGED) / n


@dataclass
class Reading:
    model: str
    family: str
    record: str
    spec: ModelSpec
    focus: str
    samples: int
    seed: int
    fractions: Dict[str, float]
    score: float
    contested_share: float
    contested_share_se: float
    window_score: float
    bank: str
    bank_created: bool
    seconds: float
    witnesses: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["spec"] = dataclasses.asdict(self.spec)
        return out


class ContactModel:
    name = ""
    reach_type = Reach
    margin_gap: Optional[str] = None

    def __init__(self, config: FamilyConfig, record: Optional[Record] = None,
                 *, drones: Optional[int] = None,
                 static_obstacles: Optional[int] = None,
                 dynamic_obstacles: Optional[int] = None,
                 horizon: Optional[int] = None,
                 margin: Optional[float] = None):
        spec = config.model(self.name)
        changes = {key: value for key, value in (
            ("drones", drones), ("static_obstacles", static_obstacles),
            ("dynamic_obstacles", dynamic_obstacles), ("horizon", horizon),
            ("margin", margin)) if value is not None}
        if changes:
            spec = dataclasses.replace(spec, **changes)
        config = self.judged(config, spec)
        self.config = config
        self.spec = spec
        self.record = record or read_record(config.record_path,
                                            config.physics.record_params)
        if self.record.target_key != config.name:
            raise ValueError("the record targets %r but the config is for %r"
                             % (self.record.target_key, config.name))
        self.simulator = Simulator(self.record, config)
        self.reach = self.make_reach(config, spec)
        self.reach.check_record(self.simulator.step_reach(),
                                "record %s" % self.record.path.name)
        self.sampler = Sampler(config, spec)

    def make_reach(self, config: FamilyConfig, spec: ModelSpec) -> Reach:
        return self.reach_type(config, spec.horizon)

    @classmethod
    def judged(cls, config: FamilyConfig, spec: ModelSpec) -> FamilyConfig:
        if not spec.margin:
            return config
        if cls.margin_gap is None:
            raise ValueError("the %s question judges no contact: it takes no "
                             "margin" % cls.name)
        physics = config.physics
        return dataclasses.replace(config, physics=dataclasses.replace(
            physics, **{cls.margin_gap: getattr(physics, cls.margin_gap)
                        + float(spec.margin)}))

    @property
    def horizon(self) -> int:
        return self.spec.horizon


    def gaps(self, positions: np.ndarray, obstacles: np.ndarray,
             radii: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def pair_members(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def margins(self, world: World, state: State) -> np.ndarray:
        return self.gaps(world.positions, world.obstacles,
                         world.obstacle_radii)


    def sample(self, n: Optional[int] = None, seed: Optional[int] = None,
               regime: float = 0.0) -> Tuple[State, Dict[str, Any]]:
        sampling = self.config.sampling
        n = sampling.samples if n is None else int(n)
        seed = sampling.seed if seed is None else int(seed)
        return self.reach.sample(self.sampler, n, seed,
                                 contested=sampling.focus == "contested",
                                 regime=regime)

    def bank(self, samples: Optional[int] = None,
             seed: Optional[int] = None) -> Bank:
        return bank_for(self, samples, seed)


    def doomed(self, state: State) -> np.ndarray:
        return self.reach.split(state)[1]

    def evaluate(self, state: State) -> Outcome:
        b = state.size
        verdict = np.full(b, int(Verdict.SAFE), dtype=np.int8)
        tick = np.zeros(b, dtype=np.int32)
        drone = np.full(b, -1, dtype=np.int32)
        other = np.full(b, -1, dtype=np.int32)
        clearance = np.full(b, np.inf)
        infeasible = ~admissible(state, self.config)
        doomed = self.doomed(state) & ~infeasible
        first, second = self.pair_members(state)

        world = World.of(state)
        for t in range(1, self.horizon + 1):
            self.simulator.tick(world)
            fresh = world.singular & (tick == 0)
            verdict[fresh] = int(Verdict.SINGULAR)
            tick[fresh] = t
            if first.size == 0:
                continue
            gaps = self.margins(world, state)
            gaps = np.where(np.isnan(gaps), np.inf, gaps)
            worst = np.argmin(gaps, axis=1)
            closest = gaps[np.arange(b), worst]
            clearance = np.minimum(clearance, closest)
            hit = (closest < 0.0) & (tick == 0)
            verdict[hit] = int(Verdict.VIOLATES)
            tick[hit] = t
            drone[hit] = first[worst[hit]]
            other[hit] = second[worst[hit]]

        verdict[doomed] = int(Verdict.DOOMED)
        verdict[infeasible] = int(Verdict.INFEASIBLE)
        return Outcome(verdict, tick, drone, other, clearance,
                       world.positions)


    def measure(self, samples: Optional[int] = None,
                seed: Optional[int] = None, witnesses: int = 10,
                bank: Optional[Bank] = None
                ) -> Tuple[Reading, State, Outcome]:
        started = time.perf_counter()
        bank = bank or self.bank(samples, seed)
        outcome = self.evaluate(bank.state)
        seconds = time.perf_counter() - started
        info = bank.meta["sampler"]
        score = outcome.score()
        share = float(info["contested_share"])
        focus = bank.meta["description"]["focus"]
        violating = np.flatnonzero(outcome.verdict == int(Verdict.VIOLATES))
        reading = Reading(
            model=self.name, family=self.config.name,
            record=str(self.record.path), spec=self.spec, focus=focus,
            samples=bank.state.size,
            seed=int(bank.meta["description"]["seed"]),
            fractions=outcome.fractions(), score=score,
            contested_share=share,
            contested_share_se=float(info["contested_share_se"]),
            window_score=score * share if focus == "contested" else score,
            bank=str(bank.path), bank_created=bank.created, seconds=seconds,
            witnesses=[int(i) for i in violating[:witnesses]])
        return reading, bank.state, outcome


__all__ = ["ContactModel", "Outcome", "Reading"]
