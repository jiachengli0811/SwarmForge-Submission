from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .operators import Mutation
from .properties import PropertyComparison, PropertyProfile
from .validate import Validity
from .weakness import WeaknessReading, WeaknessScore

STAGES = ("refused", "duplicate", "invalid", "screened", "measured",
          "flown")

ORIGINAL = "baseline"


@dataclass
class Candidate:
    id: str
    parent: str = ""
    parent_digest: str = ""
    generation: int = 0
    mutation: Optional[Mutation] = None
    source: str = ""
    digest: str = ""
    stage: str = "screened"
    note: str = ""
    same_as: str = ""
    validity: Optional[Validity] = None
    profile: Optional[PropertyProfile] = field(default=None, repr=False)
    comparison: Optional[PropertyComparison] = None
    weakness: Optional[WeaknessReading] = field(default=None, repr=False)
    score: Optional[WeaknessScore] = None
    expressions: Dict[str, str] = field(default_factory=dict)
    retained: bool = False
    annotations: Dict[str, Any] = field(default_factory=dict)
    round: int = 0
    created_at: float = field(default_factory=time.time)
    path: str = ""

    @property
    def measured(self) -> bool:
        return self.stage == "measured"

    @property
    def total(self) -> Optional[float]:
        if self.score is None or not self.score.complete:
            return None
        return self.score.total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "parent": self.parent,
            "parent_digest": self.parent_digest,
            "generation": self.generation, "round": self.round,
            "mutation": (None if self.mutation is None
                         else self.mutation.to_dict()),
            "source": self.source, "digest": self.digest,
            "stage": self.stage, "note": self.note, "same_as": self.same_as,
            "validity": (None if self.validity is None
                         else self.validity.to_dict()),
            "comparison": (None if self.comparison is None
                           else self.comparison.to_dict()),
            "weakness": (None if self.weakness is None
                         else self.weakness.to_dict()),
            "score": None if self.score is None else self.score.to_dict(),
            "expressions": dict(self.expressions),
            "retained": self.retained,
            "annotations": dict(self.annotations),
            "created_at": round(self.created_at, 3),
        }


__all__ = ["Candidate", "ORIGINAL", "STAGES"]
