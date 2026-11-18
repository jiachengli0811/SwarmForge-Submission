from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Vec = Tuple[float, ...]


def vnorm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


class Scene:
    dims: int = 2

    def resolve(self, source: str, agent) -> List:
        raise NotImplementedError

    def param(self, name: Optional[str]) -> Optional[float]:
        return None

    def position(self, obj) -> Vec:
        return tuple(float(x) for x in obj.position)

    def velocity(self, obj) -> Vec:
        v = getattr(obj, "velocity", None)
        return (tuple(float(x) for x in v) if v is not None
                else tuple([0.0] * self.dims))

    def size(self, obj) -> float:
        return float(getattr(obj, "size", 0.0) or 0.0)

    def role(self, agent) -> str:
        return "agent"


__all__ = ["Scene", "Vec", "vnorm"]
