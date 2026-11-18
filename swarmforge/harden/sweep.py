from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import sympy as sp

from ..solve.laws import Expression
from .domain import Domain

SAMPLES = 257
GRID = 17


def grid_axes(inputs: Sequence[str], domain: Domain
              ) -> Tuple[Tuple[str, ...], Tuple[Tuple[float, float], ...]]:
    spans = []
    for name in inputs:
        span = domain.span(name)
        if span is None:
            raise ValueError("%s reads %s, which law %r's domain does not span"
                             % ("a unit", name, domain.law))
        spans.append(span)
    return tuple(inputs), tuple(spans)


def grids_for(spans: Sequence[Tuple[float, float]]
              ) -> Tuple[Tuple[float, ...], ...]:
    count = SAMPLES if len(spans) == 1 else GRID
    return tuple(tuple(float(v) for v in np.linspace(lo, hi, count))
                 for lo, hi in spans)


@dataclass(frozen=True)
class Sweep:
    axes: Tuple[str, ...]
    grids: Tuple[Tuple[float, ...], ...]
    values: np.ndarray
    problem: str = ""

    @property
    def ok(self) -> bool:
        return not self.problem

    @property
    def finite(self) -> np.ndarray:
        return self.values[np.isfinite(self.values)]

    @property
    def peak(self) -> float:
        finite = self.finite
        return float(np.max(np.abs(finite))) if finite.size else 0.0


def where(axes: Sequence[str], point: Sequence[float]) -> str:
    return ", ".join("%s = %.6g" % (name, at) for name, at in zip(axes, point))


def sweep(expr: sp.Expr, axes: Sequence[str],
          grids: Tuple[Tuple[float, ...], ...]) -> Sweep:
    compiled = Expression(expr)
    if axes:
        mesh = np.meshgrid(*[np.asarray(g, dtype=float) for g in grids],
                           indexing="ij")
        env = {name: array.ravel() for name, array in zip(axes, mesh)}
        shape = (int(mesh[0].size),)
    else:
        env, shape = {}, (1,)
    missing = [name for name in compiled.inputs if name not in env]
    if missing:
        raise ValueError("the unit reads %s, which the grid does not walk"
                         % ", ".join(missing))
    values = np.array(compiled(env, shape), dtype=float, copy=True)
    bad = np.flatnonzero(~np.isfinite(values))
    problem = ""
    if bad.size:
        point = next(itertools.islice(itertools.product(*grids),
                                      int(bad[0]), None)) if axes else ()
        problem = "not real and finite at %s" % (where(axes, point)
                                                 or "every point")
    return Sweep(tuple(axes), tuple(grids), values, problem)


__all__ = ["GRID", "SAMPLES", "Sweep", "grid_axes", "grids_for", "sweep",
           "where"]
