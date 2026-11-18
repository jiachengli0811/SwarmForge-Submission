from __future__ import annotations

import collections
import math
import random
from typing import Dict, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def gumbel(rng: random.Random) -> float:
    return -math.log(-math.log(max(rng.random(), 1e-12)))


def top_k(items: Sequence[T], weights: Sequence[float], k: int,
          rng: random.Random) -> List[T]:
    if len(items) != len(weights):
        raise ValueError("%d items and %d weights" % (len(items),
                                                      len(weights)))
    if k <= 0 or not items:
        return []
    keyed = [(math.log(w) + gumbel(rng) if w > 0.0 else -math.inf, i)
             for i, w in enumerate(weights)]
    keyed.sort(key=lambda pair: -pair[0])
    return [items[i] for _key, i in keyed[:int(k)]]


def boltzmann(values: Sequence[float], temperature: float) -> List[float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [0.0] * len(values)
    best = min(finite)
    if temperature <= 0.0:
        return [1.0 if v == best else 0.0 for v in values]
    return [math.exp(-(v - best) / temperature) if math.isfinite(v) else 0.0
            for v in values]


def choices(items: Sequence[T], weights: Sequence[float], k: int,
            rng: random.Random) -> List[Tuple[T, int]]:
    if k <= 0 or not items or not any(w > 0.0 for w in weights):
        return []
    drawn: Dict[int, int] = collections.Counter(
        rng.choices(range(len(items)), weights=weights, k=int(k)))
    return [(items[i], drawn[i]) for i in range(len(items)) if i in drawn]


__all__ = ["boltzmann", "choices", "gumbel", "top_k"]
