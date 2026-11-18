from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..config import FamilyConfig
from ..laws import norm
from ..reach import AgentReach
from ..record import Record
from ..state import State
from .contact import ContactModel, Outcome, Reading

MODEL_NAME = "agent_collision"


class AgentCollision(ContactModel):
    name = MODEL_NAME
    reach_type = AgentReach
    margin_gap = "agent_min_gap"

    def __init__(self, config: FamilyConfig, record: Optional[Record] = None,
                 **counts):
        super().__init__(config, record, **counts)
        if self.spec.drones < 2:
            raise ValueError("%s needs at least two drones; models.%s.drones "
                             "is %d" % (MODEL_NAME, MODEL_NAME,
                                        self.spec.drones))

    def gaps(self, positions: np.ndarray, obstacles: np.ndarray,
             radii: np.ndarray) -> np.ndarray:
        first, second = AgentReach.pairs(positions.shape[1])
        distance = norm(positions[:, first] - positions[:, second])
        return distance - self.config.physics.agent_boundary

    def pair_members(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        return AgentReach.pairs(state.drones)


__all__ = ["AgentCollision", "MODEL_NAME", "Outcome", "Reading"]
