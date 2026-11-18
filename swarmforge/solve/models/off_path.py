from __future__ import annotations

from ..hold import PathReach
from .contact import Outcome, Reading
from .hold import HoldModel

MODEL_NAME = "off_path"


class OffPath(HoldModel):
    name = MODEL_NAME
    reach_type = PathReach
    thresholds = ("path_width",)


__all__ = ["MODEL_NAME", "OffPath", "Outcome", "Reading"]
