from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from ..solve.config import TestRangeSpec


@dataclass(frozen=True)
class Slot:
    band: int
    target_success: float
    rho: float
    spacing: float

    def to_dict(self) -> Dict[str, float]:
        return {"band": self.band,
                "target_success": round(self.target_success, 6),
                "rho": float("%.6g" % self.rho),
                "spacing": float("%.6g" % self.spacing)}


@dataclass(frozen=True)
class TestRange:
    rho_easy: float
    rho_hard: float
    curve: Tuple[Tuple[float, float], ...]
    spacing_easy: float
    spacing_hard: float
    bands: int = 5
    weights: Tuple[int, ...] = ()

    @classmethod
    def of(cls, spec: TestRangeSpec) -> "TestRange":
        return cls(rho_easy=spec.rho_easy, rho_hard=spec.rho_hard,
                   curve=tuple(spec.curve), spacing_easy=spec.spacing_easy,
                   spacing_hard=spec.spacing_hard, bands=spec.bands,
                   weights=tuple(spec.weights))


    def success_at(self, rho: float) -> float:
        points = self.curve
        if rho <= points[0][0]:
            return points[0][1]
        for (x0, p0), (x1, p1) in zip(points, points[1:]):
            if rho <= x1:
                return p0 + _fraction(rho, x0, x1) * (p1 - p0)
        return points[-1][1]

    def rho_for(self, success: float) -> float:
        points = self.curve
        if success >= points[0][1]:
            return points[0][0]
        for (x0, p0), (x1, p1) in zip(points, points[1:]):
            if p0 >= success > p1:
                t = (p0 - success) / (p0 - p1)
                if x0 <= 0.0:
                    return x0 + t * (x1 - x0)
                return math.exp(math.log(x0) + t * (math.log(x1)
                                                    - math.log(x0)))
        return points[-1][0]

    @property
    def success_easy(self) -> float:
        return self.success_at(self.rho_easy)

    @property
    def success_hard(self) -> float:
        return self.success_at(self.rho_hard)

    def band_edges(self) -> List[float]:
        top, bottom = self.success_easy, self.success_hard
        return [top + (bottom - top) * k / self.bands
                for k in range(self.bands + 1)]

    @property
    def band_weights(self) -> Tuple[int, ...]:
        return tuple(self.weights) or (1,) * self.bands


    def slot(self, band: int, rng: random.Random) -> Slot:
        edges = self.band_edges()
        hi, lo = edges[band], edges[band + 1]
        target = rng.uniform(lo, hi)
        rho = min(max(self.rho_for(target), self.rho_easy), self.rho_hard)
        spacing = rng.uniform(min(self.spacing_hard, self.spacing_easy),
                              max(self.spacing_hard, self.spacing_easy))
        return Slot(band, target, rho, spacing)

    def schedule(self, count: int, name: str) -> List[Slot]:
        slots: List[Slot] = []
        block = 0
        while len(slots) < int(count):
            order = [band for band in range(self.bands)
                     for _ in range(self.band_weights[band])]
            random.Random("%s:block:%d" % (name, block)).shuffle(order)
            for band in order:
                if len(slots) >= int(count):
                    break
                slots.append(self.slot(band, random.Random(
                    "%s:slot:%d" % (name, len(slots)))))
            block += 1
        return slots

    def to_dict(self) -> Dict[str, Any]:
        return {"bands": self.bands,
                "weights": list(self.band_weights),
                "rho": {"easy": self.rho_easy, "hard": self.rho_hard,
                        "success_easy": round(self.success_easy, 4),
                        "success_hard": round(self.success_hard, 4),
                        "curve": [list(point) for point in self.curve]},
                "min_spacing": {"easy": self.spacing_easy,
                                "hard": self.spacing_hard},
                "band_edges": [round(edge, 4) for edge in self.band_edges()]}


def _fraction(x: float, x0: float, x1: float) -> float:
    if x0 <= 0.0:
        return (x - x0) / (x1 - x0)
    return (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))


def ceiling(rho: float, bounds: Sequence[float]) -> int:
    return int(round(float(rho) * float(bounds[0]) * float(bounds[1])))


__all__ = ["Slot", "TestRange", "ceiling"]
