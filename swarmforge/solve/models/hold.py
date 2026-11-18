from __future__ import annotations

from typing import Tuple

import numpy as np

from ..config import FamilyConfig, ModelSpec
from ..dynamics import World
from ..hold import HoldReach
from ..state import State
from .contact import ContactModel


class HoldModel(ContactModel):
    reach_type = HoldReach
    thresholds: Tuple[str, ...] = ()

    def make_reach(self, config: FamilyConfig, spec: ModelSpec):
        return self.reach_type(config, spec.horizon,
                               *(getattr(spec, name)
                                 for name in self.thresholds))

    def pair_members(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        return self.reach.members(state.drones)

    def margins(self, world: World, state: State) -> np.ndarray:
        return np.where(self.reach.held(state), self.reach.margin(world),
                        np.inf)


__all__ = ["HoldModel"]
