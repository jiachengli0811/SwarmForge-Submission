from __future__ import annotations

from ..hold import RangeReach
from .contact import Outcome, Reading
from .hold import HoldModel

MODEL_NAME = "loss_target"


class LossTarget(HoldModel):
    name = MODEL_NAME
    reach_type = RangeReach
    thresholds = ("target_range",)


__all__ = ["LossTarget", "MODEL_NAME", "Outcome", "Reading"]
