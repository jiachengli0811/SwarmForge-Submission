from __future__ import annotations

import numpy as np

from ..config import FamilyConfig, ModelSpec
from ..dynamics import World
from ..reach import TimeoutReach
from ..state import State, admissible
from ..verdict import Verdict
from .contact import ContactModel, Outcome, Reading

MODEL_NAME = "timeout"


class Timeout(ContactModel):
    name = MODEL_NAME
    reach_type = TimeoutReach

    def make_reach(self, config: FamilyConfig, spec: ModelSpec
                   ) -> TimeoutReach:
        return TimeoutReach(config, spec.horizon, spec.pace)

    def pair_members(self, state: State):
        empty = np.zeros(0, dtype=int)
        return empty, empty

    def evaluate(self, state: State) -> Outcome:
        b = state.size
        verdict = np.full(b, int(Verdict.SAFE), dtype=np.int8)
        tick = np.zeros(b, dtype=np.int32)
        infeasible = ~admissible(state, self.config)
        doomed = self.doomed(state) & ~infeasible
        target = state.target.copy()
        world = World.of(state)
        for t in range(1, self.horizon + 1):
            self.simulator.tick(world)
            fresh = world.singular & (tick == 0)
            verdict[fresh] = int(Verdict.SINGULAR)
            tick[fresh] = t
        margin, arrived = self.reach.progress(state.positions,
                                              world.positions, target,
                                              state.route)
        short = ~arrived & (margin < 0.0) & (tick == 0)
        verdict[short] = int(Verdict.VIOLATES)
        tick[short] = self.horizon
        verdict[doomed] = int(Verdict.DOOMED)
        verdict[infeasible] = int(Verdict.INFEASIBLE)
        none = np.full(b, -1, dtype=np.int32)
        return Outcome(verdict, tick, none, none.copy(),
                       np.where(arrived, np.inf, margin), world.positions)


__all__ = ["MODEL_NAME", "Outcome", "Reading", "Timeout"]
