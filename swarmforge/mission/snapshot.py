from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

Vec = Tuple[float, float]


@dataclass
class Snapshot:
    tick: int
    positions: Tuple[Vec, ...]
    velocities: Tuple[Vec, ...]
    obstacles: Tuple[Tuple[Vec, float], ...]
    goal: Optional[Vec] = None
    movers: Tuple[Tuple[Vec, float, Vec], ...] = ()
    aim: Optional[Vec] = None
    route: Vec = (0.0, 0.0)


__all__ = ["Snapshot", "Vec"]
