from __future__ import annotations

from typing import Optional

from ..config import FamilyConfig
from ..hold import LinkReach
from ..record import Record
from .contact import Outcome, Reading
from .hold import HoldModel

MODEL_NAME = "fragmentation"


class Fragmentation(HoldModel):
    name = MODEL_NAME
    reach_type = LinkReach
    thresholds = ("sense_range",)

    def __init__(self, config: FamilyConfig, record: Optional[Record] = None,
                 **counts):
        super().__init__(config, record, **counts)
        if self.spec.drones < 2:
            raise ValueError("%s needs at least two drones; models.%s.drones "
                             "is %d" % (MODEL_NAME, MODEL_NAME,
                                        self.spec.drones))


__all__ = ["Fragmentation", "MODEL_NAME", "Outcome", "Reading"]
