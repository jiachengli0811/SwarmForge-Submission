from __future__ import annotations

from typing import Optional

from ..config import FamilyConfig
from ..record import Record
from .agent_collision import AgentCollision
from .contact import ContactModel, Outcome, Reading
from .fragmentation import Fragmentation
from .hold import HoldModel
from .loss_target import LossTarget
from .misalignment import Misalignment
from .obstacle_collision import ObstacleCollision
from .off_path import OffPath
from .timeout import Timeout

MODELS = {model.name: model for model in (ObstacleCollision, AgentCollision,
                                          Timeout, Fragmentation,
                                          Misalignment, LossTarget, OffPath)}

MISSION_MODELS = ("obstacle_collision", "agent_collision", "timeout")


def build_model(name: str, config: FamilyConfig,
                record: Optional[Record] = None, **overrides):
    if name not in MODELS:
        raise ValueError("no model %r (have: %s)" % (name, ", ".join(MODELS)))
    return MODELS[name](config, record, **overrides)


__all__ = ["AgentCollision", "ContactModel", "Fragmentation", "HoldModel",
           "LossTarget", "MISSION_MODELS", "MODELS", "Misalignment",
           "ObstacleCollision", "OffPath", "Outcome", "Reading", "Timeout",
           "build_model"]
