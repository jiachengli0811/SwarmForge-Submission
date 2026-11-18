from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import sympy as sp

from ...solve.laws import Expression
from ...solve.record import INPUTS, _is_size, _restore_clearance
from ..document import Document, digest, namespace, read_site, gate_of
from .base import OperatorContext
from .tree import is_input, literal

DIST, CLEARANCE = sp.Symbol("dist"), sp.Symbol("clearance")

POINTS = {1: 49, 2: 13, 3: 7}

MATCH_RANGE = 1e12


class Calibration:
    def __init__(self, document: Document, context: OperatorContext):
        self.document = document
        self.context = context
        values = dict(namespace(document))
        for name, value in context.config.physics.record_params.items():
            if values.get(name) is None:
                values[name] = float(value)
        self.namespace: Dict[str, Optional[float]] = values
        self._bind = {sp.Symbol(k): v for k, v in values.items()
                      if v is not None}
        self._terms = {str(t.get("name")): t
                       for t in document.get("terms") or ()}
        self._surface: Dict[Tuple[str, str], bool] = {}
        self._cache: Dict[Tuple[str, str, str], Optional[np.ndarray]] = {}

    @classmethod
    def of(cls, document: Document, context: OperatorContext
           ) -> "Calibration":
        key = digest(document)
        found = context.cache.get(key)
        if found is None:
            found = context.cache[key] = cls(document, context)
        return found


    def surface(self, law: str, slot: str) -> bool:
        key = (law, slot)
        if key not in self._surface:
            term = self._terms[law]
            surface = (term.get("measures") or "centre_to_centre") \
                == "centre_to_surface"
            if surface:
                if slot == "gate":
                    expr = gate_of(self.document, law).threshold
                else:
                    from ..document import Site
                    expr = read_site(self.document, Site(law, slot)).expr
                surface = expr is None or not any(
                    _is_size(s.name) for s in expr.free_symbols)
            self._surface[key] = surface
        return self._surface[key]

    def flown(self, law: str, slot: str, expr: Any) -> sp.Expr:
        expr = sp.sympify(expr)
        if self.surface(law, slot):
            expr = expr.subs(DIST, CLEARANCE)
        expr = _restore_clearance(expr)
        return expr.subs(self._bind) if self._bind else expr

    def semantic(self, law: str, slot: str, name: str) -> str:
        if name == "dist" and slot != "weight" and self.surface(law, slot):
            return "clearance"
        return name


    def span(self, law: str, name: str, slot: str = "magnitude",
             extent: bool = False) -> Tuple[float, float]:
        domain = self.context.domains[law]
        key = self.semantic(law, slot, name)
        pair = domain.reach_span(key) if extent else domain.span(key)
        if pair is None:
            pair = domain.reach_span(key) or (0.0, 1.0)
        return float(pair[0]), float(pair[1])

    def reference(self, law: str, name: str, slot: str = "magnitude") -> float:
        lo, hi = self.span(law, name, slot)
        middle = 0.5 * (lo + hi)
        if middle == 0.0:
            middle = hi or 1.0
        return abs(middle)

    def level(self, law: str, name: str, q: float, slot: str = "magnitude",
              extent: bool = False) -> float:
        lo, hi = self.span(law, name, slot, extent)
        return lo + q * (hi - lo)


    def _env(self, law: str, names) -> Tuple[Dict[str, np.ndarray], tuple]:
        domain = self.context.domains[law]
        offset = domain.offset
        axes = sorted(set(names))
        if offset is not None and "clearance" in axes and "dist" in axes:
            axes.remove("clearance")
        count = POINTS.get(len(axes), 5)
        grids = []
        for name in axes:
            pair = domain.span(name) or domain.reach_span(name) or (0.0, 1.0)
            grids.append(np.linspace(pair[0], pair[1], count))
        if axes:
            mesh = np.meshgrid(*grids, indexing="ij")
            env = {n: m.ravel() for n, m in zip(axes, mesh)}
            shape = (int(mesh[0].size),)
        else:
            env, shape = {}, (1,)
        if offset is not None and "dist" in env:
            env["clearance"] = env["dist"] - offset
        return env, shape

    def values(self, law: str, slot: str, expr: Any) -> Optional[np.ndarray]:
        key = (law, slot, sp.srepr(expr))
        if key in self._cache:
            return self._cache[key]
        out = None
        try:
            flown = self.flown(law, slot, expr)
            if not (flown.free_symbols - {sp.Symbol(n) for n in INPUTS}):
                names = [s.name for s in flown.free_symbols if is_input(s)]
                env, shape = self._env(law, names)
                with np.errstate(all="ignore"):
                    raw = np.asarray(Expression(flown)(env, shape),
                                     dtype=float)
                finite = raw[np.isfinite(raw)]
                out = finite if finite.size else None
        except (TypeError, ValueError, ZeroDivisionError, OverflowError,
                AttributeError, KeyError, NameError, SyntaxError):
            out = None
        self._cache[key] = out
        return out

    def size(self, law: str, slot: str, expr: Any) -> Optional[float]:
        values = self.values(law, slot, expr)
        if values is None:
            return None
        size = float(np.mean(np.abs(values)))
        return size if size > 0.0 and np.isfinite(size) else None

    def mean(self, law: str, slot: str, expr: Any) -> Optional[float]:
        values = self.values(law, slot, expr)
        return None if values is None else float(np.mean(values))

    def quantile(self, law: str, slot: str, expr: Any,
                 q: float) -> Optional[float]:
        values = self.values(law, slot, expr)
        return None if values is None else float(np.quantile(values, q))

    def match(self, law: str, slot: str, old: Any, new: Any) -> Any:
        size_old, size_new = (self.size(law, slot, old),
                              self.size(law, slot, new))
        if size_old is None or size_new is None:
            return new
        factor = size_old / size_new
        if not 1.0 / MATCH_RANGE <= factor <= MATCH_RANGE:
            return None
        mean_old, mean_new = (self.mean(law, slot, old),
                              self.mean(law, slot, new))
        if mean_old is not None and mean_new is not None \
                and mean_old * mean_new < 0.0:
            factor = -factor
        factor_literal = literal(factor)
        if factor_literal == 1:
            return new
        return factor_literal * new


__all__ = ["Calibration", "MATCH_RANGE", "POINTS"]
