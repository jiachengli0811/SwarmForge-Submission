from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

from .sweep import where

NONZERO, NONNEGATIVE, POSITIVE = "nonzero", "nonnegative", "positive"

STEPS = 60
SUSPECT = 1e-3
ZERO = 1e-12


@dataclass(frozen=True)
class Hazard:
    expr: sp.Expr
    kind: str
    guard: sp.Basic
    node: sp.Expr


def hazards(expr: sp.Expr) -> List[Hazard]:
    out: List[Hazard] = []

    def add(base: sp.Expr, kind: str, guard, node) -> None:
        if base.free_symbols:
            out.append(Hazard(base, kind, guard, node))

    def walk(node, guard) -> None:
        if isinstance(node, sp.Piecewise):
            earlier = sp.false
            for pair in node.args:
                taken = sp.And(guard, pair.cond, sp.Not(earlier))
                walk(pair.cond, sp.And(guard, sp.Not(earlier)))
                walk(pair.expr, taken)
                earlier = sp.Or(earlier, pair.cond)
            return
        if isinstance(node, sp.ITE):
            condition, then, otherwise = node.args
            walk(condition, guard)
            walk(then, sp.And(guard, condition))
            walk(otherwise, sp.And(guard, sp.Not(condition)))
            return
        if isinstance(node, sp.Pow):
            base, exponent = node.args
            if exponent.is_number:
                value = float(exponent)
                if value < 0:
                    add(base, NONZERO if float(value).is_integer()
                        else POSITIVE, guard, node)
                elif not float(value).is_integer():
                    add(base, NONNEGATIVE, guard, node)
            else:
                add(base, POSITIVE, guard, node)
        elif getattr(node, "func", None) is sp.log:
            add(node.args[0], POSITIVE, guard, node)
        for child in node.args:
            walk(child, guard)

    walk(expr, sp.true)
    return out


def _on_grid(expr, axes: Sequence[str], mesh) -> np.ndarray:
    if expr is sp.true:
        return np.ones(mesh[0].size if mesh else 1, dtype=bool)
    if expr is sp.false:
        return np.zeros(mesh[0].size if mesh else 1, dtype=bool)
    call = sp.lambdify([sp.Symbol(a) for a in axes], expr,
                       modules=[{"Max": np.maximum, "Min": np.minimum},
                                "numpy"])
    with np.errstate(all="ignore"):
        values = call(*[m.ravel() for m in mesh]) if mesh else call()
    size = mesh[0].size if mesh else 1
    return np.broadcast_to(np.asarray(values), (size,)).copy()


def _point(expr, axes: Sequence[str]) -> Callable[..., Optional[float]]:
    call = sp.lambdify([sp.Symbol(a) for a in axes], expr, "math")

    def evaluate(*at: float) -> Optional[float]:
        try:
            value = float(call(*at))
        except (ZeroDivisionError, ValueError, OverflowError, TypeError):
            return None
        return value if math.isfinite(value) else None
    return evaluate


def _golden(f: Callable[[float], Optional[float]], lo: float,
            hi: float) -> Tuple[float, float]:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(STEPS):
        if fc is None or fd is None:
            break
        if abs(fc) < abs(fd):
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = f(d)
    best = min([(abs(v), x) for v, x in ((fc, c), (fd, d)) if v is not None],
               default=(math.inf, 0.5 * (lo + hi)))
    return best


def singular_point(expr: sp.Expr, axes: Sequence[str],
                   grids: Tuple[Tuple[float, ...], ...],
                   guard: Any = None) -> Optional[str]:
    found = [h for h in hazards(expr)
             if h.expr.free_symbols <= {sp.Symbol(a) for a in axes}]
    if guard is not None:
        found = [Hazard(h.expr, h.kind, sp.And(guard, h.guard), h.node)
                 for h in found]
    if not found or not axes:
        return None
    mesh = np.meshgrid(*[np.asarray(g, dtype=float) for g in grids],
                       indexing="ij")
    shape = tuple(len(g) for g in grids)
    for hazard in found:
        values = np.asarray(_on_grid(hazard.expr, axes, mesh),
                            dtype=float).reshape(shape)
        inside = np.asarray(_on_grid(hazard.guard, axes, mesh),
                            dtype=bool).reshape(shape)
        finite = np.isfinite(values) & inside
        evaluate = None
        for axis in range(len(axes)):
            others = [i for i in range(len(axes)) if i != axis]
            for fixed in itertools.product(*(range(shape[i])
                                             for i in others)):
                index: List = [slice(None)] * len(axes)
                for i, k in zip(others, fixed):
                    index[i] = k
                line, ok = values[tuple(index)], finite[tuple(index)]
                grid = grids[axis]

                def at(k: int) -> Tuple[float, ...]:
                    point = [0.0] * len(axes)
                    for i, j in zip(others, fixed):
                        point[i] = grids[i][j]
                    point[axis] = grid[k]
                    return tuple(point)

                for k in range(len(grid)):
                    if not ok[k]:
                        continue
                    v = line[k]
                    if (v == 0.0 and hazard.kind != NONNEGATIVE) or \
                            (v < 0.0 and hazard.kind != NONZERO):
                        return ("%s reaches %s at %s" % (
                            hazard.node, "zero" if v == 0.0 else "a negative",
                            where(axes, at(k))))
                    if k + 1 < len(grid) and ok[k + 1] and \
                            v * line[k + 1] < 0.0:
                        return ("%s crosses zero between %s and %s"
                                % (hazard.node, where(axes, at(k)),
                                   where(axes, at(k + 1))))
                if hazard.kind == NONNEGATIVE:
                    continue
                size = np.abs(np.where(ok, line, np.nan))
                if not np.any(ok):
                    continue
                peak = float(np.nanmax(size))
                for k in range(1, len(grid) - 1):
                    if not (ok[k - 1] and ok[k] and ok[k + 1]):
                        continue
                    if not (size[k] <= size[k - 1] and size[k] <= size[k + 1]
                            and size[k] < SUSPECT * peak):
                        continue
                    if evaluate is None:
                        evaluate = _point(hazard.expr, axes)
                    point = list(at(k))

                    def along(x: float) -> Optional[float]:
                        point[axis] = x
                        return evaluate(*point)

                    least, x = _golden(along, grid[k - 1], grid[k + 1])
                    if least <= ZERO * max(1.0, peak):
                        point[axis] = x
                        return ("%s touches zero near %s"
                                % (hazard.node, where(axes, point)))
    return None


__all__ = ["Hazard", "NONNEGATIVE", "NONZERO", "POSITIVE", "hazards",
           "singular_point"]
