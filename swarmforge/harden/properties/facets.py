from __future__ import annotations

import base64
import collections
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

from .core import Change, Property
from .response import (AXES, DISTANCES, LINE, PLANE, ZERO, LawAnalysis,
                       axis_grid)

DRIFT_FLOOR = 1e-6


def _close(mask: np.ndarray, kept: Optional[np.ndarray] = None
           ) -> np.ndarray:
    hole = np.zeros(len(mask), dtype=bool)
    hole[1:-1] = ~mask[1:-1] & mask[:-2] & mask[2:]
    if len(mask) > 1:
        hole[0] = not mask[0] and mask[1]
        hole[-1] = not mask[-1] and mask[-2]
    if kept is not None:
        hole &= ~kept
    return mask | hole


def _closed(analysis: LawAnalysis) -> np.ndarray:
    support = analysis.support.copy()
    beside: Dict[Tuple[str, Tuple[int, ...]], List[Any]] = \
        collections.defaultdict(list)
    for s in analysis.switches:
        beside[(s.axis, s.line)].append(s)
    for axis, name in enumerate(analysis.axes):
        grid = analysis.grids[axis]
        tol = ZERO * (analysis.width(axis) or 1.0)
        moved = np.moveaxis(support, axis, -1)
        flat = moved.reshape(-1, moved.shape[-1])
        for row, line in enumerate(np.ndindex(moved.shape[:-1])):
            kept = np.zeros(flat.shape[1], dtype=bool)
            for s in beside.get((name, line), ()):
                for i in (s.index, s.index + 1):
                    kept[i] |= abs(s.at - grid[i]) > tol
            flat[row] = _close(flat[row], kept)
        support = np.moveaxis(flat.reshape(moved.shape), -1, axis)
    return support


class Facet(Property):
    question = ""

    def extract(self, analysis: LawAnalysis) -> Dict[str, Any]:
        raise NotImplementedError

    def drift(self, before: Mapping[str, Any],
              after: Mapping[str, Any]) -> float:
        return 0.0

    def shifted(self, before: Mapping[str, Any],
                after: Mapping[str, Any]) -> bool:
        return before["class"] != after["class"]

    def compare(self, before: Any, after: Any) -> Change:
        shifted = self.shifted(before, after)
        drift = 0.0 if shifted else float(self.drift(before, after))
        changed = shifted or drift > DRIFT_FLOOR
        if shifted:
            detail = "%s -> %s" % (before["class"], after["class"])
        elif changed:
            detail = "same class, drift %.3g" % drift
        else:
            detail = "same"
        return Change(changed, 1.0 if shifted else min(1.0, drift), detail,
                      self.text(before), self.text(after), shifted=shifted,
                      drift=drift, classes=(before["class"], after["class"]))

    def text(self, value: Any) -> str:
        return str(value.get("label") or value["class"])


_PIECEWISE = (sp.Piecewise, sp.Max, sp.Min, sp.Abs, sp.sign, sp.ITE,
              sp.Heaviside)
_SATURATING = (sp.tanh, sp.atan)
_OSCILLATING = (sp.sin, sp.cos)


def _degree(expr: sp.Expr, x: sp.Symbol) -> Optional[int]:
    try:
        return sp.Poly(expr, x).degree()
    except (sp.PolynomialError, sp.GeneratorsNeeded, ValueError):
        return None


def _power(expr: sp.Expr, x: sp.Symbol) -> Optional[float]:
    for kind in (sp.Add, sp.Mul):
        if isinstance(expr, kind):
            dependent = [t for t in expr.args if x in t.free_symbols]
            if len(dependent) != 1:
                return None
            expr = dependent[0]
    if isinstance(expr, sp.Pow) and expr.exp.is_number \
            and _degree(expr.base, x) == 1:
        return float(expr.exp)
    return None


def family(expr: sp.Expr, x: sp.Symbol) -> str:
    if x not in expr.free_symbols:
        return "constant"
    if any(x in node.free_symbols for node in expr.atoms(*_PIECEWISE)):
        return "piecewise"
    degree = _degree(expr, x)
    if degree is not None:
        if degree == 1:
            return ("linear" if sp.Poly(expr, x).eval(0) == 0 else "affine")
        return "polynomial(%d)" % degree
    k = _power(expr, x)
    if k is not None:
        return "power(%g)" % k
    if expr.is_rational_function(x):
        return "rational"
    kinds = set()
    for node in sp.preorder_traversal(expr):
        if x not in getattr(node, "free_symbols", ()):
            continue
        if isinstance(node, sp.exp) or (isinstance(node, sp.Pow)
                                        and x in node.exp.free_symbols):
            argument = node.args[0] if isinstance(node, sp.exp) else node.exp
            try:
                quadratic = sp.Poly(argument, x).degree() == 2
            except (sp.PolynomialError, sp.GeneratorsNeeded, ValueError):
                quadratic = False
            kinds.add("gaussian" if quadratic else "exponential")
        elif isinstance(node, sp.log):
            kinds.add("logarithmic")
        elif isinstance(node, _SATURATING):
            kinds.add("saturating")
        elif isinstance(node, _OSCILLATING):
            kinds.add("oscillatory")
    if len(kinds) == 1:
        return kinds.pop()
    return "composite(%s)" % "+".join(sorted(kinds)) if kinds else "composite"


def _families(families: Optional[Mapping[str, str]]) -> str:
    return ", ".join("%s:%s" % kv for kv in families.items()) or "constant"


class Variables(Facet):
    name = "variables"
    question = "which inputs does the law react to?"
    description = "the inputs the response reads, and how each enters"

    def extract(self, analysis: LawAnalysis) -> Dict[str, Any]:
        r = analysis.response
        term = r.term
        out: Dict[str, Any] = {"class": ", ".join(analysis.axes) or "none",
                               "inputs": list(analysis.axes),
                               "magnitude": None, "threshold": None,
                               "nodes": int(sp.count_ops(term.magnitude))}

        def canonical(expr):
            if r.offset is not None:
                return expr.subs(sp.Symbol("clearance"),
                                 sp.Symbol("dist") - r.offset)
            return expr

        def families(expr) -> Dict[str, str]:
            names = sorted(s.name for s in expr.free_symbols)
            return {n: family(expr, sp.Symbol(n)) for n in names}

        try:
            out["magnitude"] = families(canonical(term.magnitude))
            out["threshold"] = (None if term.gate is None
                                else families(canonical(term.gate.threshold)))
        except (TypeError, ValueError, AttributeError) as error:
            out["error"] = "the form could not be read: %s" % error
        return out

    def text(self, value) -> str:
        parts = [value["class"]]
        if value.get("magnitude") is not None:
            parts.append("magnitude %s" % _families(value["magnitude"]))
        if value.get("threshold") is not None:
            parts.append("threshold %s" % _families(value["threshold"]))
        return "; ".join(parts)


def _compress(signs: Sequence[float]) -> str:
    out = []
    for s in signs:
        mark = "+" if s > 0 else "-"
        if not out or out[-1] != mark:
            out.append(mark)
    return "".join(out)


class Sign(Facet):
    name = "sign"
    question = "which way does the law push?"
    description = "sign sequence of the response along its primary input"

    def extract(self, analysis: LawAnalysis) -> Dict[str, Any]:
        r = analysis.response
        positive, negative = r.senses
        if analysis.peak == 0.0:
            return {"class": "0", "label": "silent", "axis": r.primary,
                    "equilibria": [], "width": 0.0}
        if not analysis.axes:
            value = float(analysis.values.ravel()[0])
            sign = "+" if value > 0 else "-"
            return {"class": sign, "axis": None, "width": 0.0,
                    "label": positive if sign == "+" else negative,
                    "equilibria": []}
        axis = analysis.axes.index(r.primary)
        grid = analysis.grids[axis]
        outward = (1.0 if r.primary in DISTANCES
                   and positive in ("repulsive", "goal-avoiding") else -1.0)
        sequences: collections.Counter = collections.Counter()
        found: Dict[int, List[Tuple[float, Optional[Tuple[float, float]],
                                    bool]]] = collections.defaultdict(list)
        for _line, _where, values, support in analysis.lines(axis):
            idx = np.flatnonzero(support)
            if not idx.size:
                continue
            signs = np.sign(values[idx])
            sequences[_compress(signs)] += 1
            rank = 0
            for k in range(len(idx) - 1):
                if signs[k] * signs[k + 1] > 0:
                    continue
                i, j = idx[k], idx[k + 1]
                below, above = values[i], values[j]
                stable = outward * below > 0.0 > outward * above
                if j == i + 1:
                    at = grid[i] - below * (grid[j] - grid[i]) / (above - below)
                    found[rank].append((float(at), None, stable))
                else:
                    found[rank].append((float(0.5 * (grid[i] + grid[j])),
                                        (float(grid[i]), float(grid[j])),
                                        stable))
                rank += 1
        if len(sequences) == 1:
            cls = next(iter(sequences))
        else:
            cls = "mixed"
        equilibria = []
        for rank in sorted(found):
            places = found[rank]
            ats = [p[0] for p in places]
            stable = sum(1 for p in places if p[2]) * 2 >= len(places)
            band = places[0][1]
            equilibria.append({
                "at": float(np.median(ats)),
                "spread": [float(min(ats)), float(max(ats))],
                "band": None if band is None else list(band),
                "stability": "stable" if stable else "unstable"})
        names = {"+": positive, "-": negative}
        if cls == "mixed":
            label = "%s or %s, varying with %s" % (
                positive, negative, ", ".join(a for a in analysis.axes
                                              if a != r.primary))
        else:
            label = " then ".join(names[c] for c in cls)
            if len(cls) > 1:
                label += " with rising %s" % r.primary
        lo, hi = analysis.spans[axis]
        return {"class": cls, "label": label, "axis": r.primary,
                "equilibria": equilibria, "width": float(hi - lo)}

    def drift(self, before, after) -> float:
        old, new = before["equilibria"], after["equilibria"]
        width = max(before["width"], after["width"]) or 1.0
        if len(old) != len(new):
            return 1.0
        return max([abs(a["at"] - b["at"]) / width
                    for a, b in zip(old, new)] or [0.0])

    def text(self, value) -> str:
        eq = "; ".join("%s at %s = %.4g" % (e["stability"], value["axis"],
                                            e["at"])
                       for e in value["equilibria"])
        return "%s%s" % (value["label"], " (%s)" % eq if eq else "")


OFF, ON, SATURATED = 0, 1, 2
STATES = ("off", "on", "saturated")
JOINS = ("smooth", "kink", "jump")


def states(analysis: LawAnalysis) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        capped = (~np.isnan(analysis.values)
                  & (np.abs(analysis.values) >= analysis.command))
    return np.where(capped, SATURATED,
                    np.where(_closed(analysis), ON, OFF)).astype(np.int8)


def _line(grid: np.ndarray, line: np.ndarray, switches: Sequence[Any]
          ) -> Tuple[str, List[Dict[str, Any]]]:
    at: Dict[int, Any] = {}
    end = None
    for s in switches:
        if s.source == "sensing range":
            end = s
        elif s.index not in at or (JOINS.index(s.kind)
                                   > JOINS.index(at[s.index].kind)):
            at[s.index] = s
    changes = [int(k) for k in np.flatnonzero(line[:-1] != line[1:])]
    owner = {k: at[k] for k in changes if k in at}
    used = set(owner)
    for k in changes:
        j = next((j for j in (k - 1, k + 1)
                  if k not in owner and j in at and j not in used), None)
        if j is not None:
            owner[k] = at[j]
            used.add(j)
    breaks = [(k, owner.get(k)) for k in changes]
    breaks += [(j, s) for j, s in at.items() if j not in used]
    text = STATES[line[0]]
    joins = []
    for k, s in sorted(breaks, key=lambda b: b[0]):
        kind = "smooth" if s is None else s.kind
        joins.append({"kind": kind,
                      "at": (0.5 * (grid[k] + grid[k + 1]) if s is None
                             else s.at)})
        text += " -%s- %s" % (kind, STATES[line[k + 1]])
    if end is not None:
        joins.append({"kind": "jump", "at": end.at})
        text += " -jump- off"
    return text, joins


def _sequences(analysis: LawAnalysis, axis: int,
               state: np.ndarray) -> List[Dict[str, Any]]:
    name = analysis.axes[axis]
    by_line: Dict[Tuple[int, ...], List[Any]] = collections.defaultdict(list)
    for s in analysis.switches:
        if s.axis == name:
            by_line[s.line].append(s)
    moved = np.moveaxis(state, axis, -1)
    found: Dict[str, List[List[Dict[str, Any]]]] = {}
    for fixed in np.ndindex(moved.shape[:-1]):
        line = moved[fixed]
        if not line.any():
            continue
        text, joins = _line(analysis.grids[axis], line, by_line[fixed])
        found.setdefault(text, []).append(joins)
    out = []
    for text, lines in found.items():
        joins = []
        for rank, first in enumerate(lines[0]):
            ats = [joins_[rank]["at"] for joins_ in lines]
            joins.append({"kind": first["kind"],
                          "at": float(np.median(ats)),
                          "spread": [float(min(ats)), float(max(ats))]})
        out.append({"class": text, "lines": len(lines), "joins": joins})
    return out


def _states(value: Mapping[str, Any]) -> set:
    if not value["axes"] or value["class"] == "off":
        return {value["class"]}
    return {word for a in value["axes"].values() for s in a["sequences"]
            for word in s["class"].split() if word in STATES}


class Switching(Facet):
    name = "switching"
    question = "where does the law switch, and to what?"
    description = "pieces off / on / saturated along each input, and joins"

    def extract(self, analysis: LawAnalysis) -> Dict[str, Any]:
        state = states(analysis)
        out: Dict[str, Any] = {"axes": {}}
        if not analysis.axes:
            out["class"] = STATES[int(state.ravel()[0])]
            return out
        for i, axis in enumerate(analysis.axes):
            out["axes"][axis] = {"width": analysis.width(i),
                                 "sequences": _sequences(analysis, i, state)}
        out["class"] = "; ".join(
            "%s:%s" % (axis, " / ".join(s["class"] for s in a["sequences"])
                       or "off")
            for axis, a in out["axes"].items())
        if not state.any():
            out["class"] = "off"
        return out

    def shifted(self, before: Mapping[str, Any],
                after: Mapping[str, Any]) -> bool:
        was, now = before["axes"], after["axes"]
        shared = [axis for axis in was if axis in now]
        if any([s["class"] for s in was[axis]["sequences"]]
               != [s["class"] for s in now[axis]["sequences"]]
               for axis in shared):
            return True
        if any(axis not in theirs and any(s["joins"] for s in a["sequences"])
               for mine, theirs in ((was, now), (now, was))
               for axis, a in mine.items()):
            return True
        return not shared and _states(before) != _states(after)

    def drift(self, before, after) -> float:
        out = 0.0
        for axis, was in before["axes"].items():
            now = {s["class"]: s for s in
                   (after["axes"].get(axis) or {}).get("sequences", ())}
            width = was["width"] or 1.0
            for sequence in was["sequences"]:
                other = now.get(sequence["class"])
                if other is None:
                    continue
                for a, b in zip(sequence["joins"], other["joins"]):
                    out = max(out, abs(a["at"] - b["at"]) / width)
        return out

    def text(self, value) -> str:
        parts: List[str] = []
        for axis, a in value["axes"].items():
            for sequence in a["sequences"]:
                for j in sequence["joins"]:
                    part = "%s at %s = %.4g" % (j["kind"], axis, j["at"])
                    if part not in parts:
                        parts.append(part)
        return "%s (%s)" % (value["class"], "; ".join(parts)) if parts \
            else value["class"]


def cells(grids: Sequence[np.ndarray]) -> np.ndarray:
    out = np.ones(tuple(len(g) for g in grids) or (1,))
    for i, grid in enumerate(grids):
        cell = np.ones(len(grid))
        if len(grid) > 1:
            cell[1:-1] = 0.5 * (grid[2:] - grid[:-2])
            cell[0] = 0.5 * (grid[1] - grid[0])
            cell[-1] = 0.5 * (grid[-1] - grid[-2])
            cell /= (grid[-1] - grid[0]) or 1.0
        shape = [1] * len(grids)
        shape[i] = len(grid)
        out = out * cell.reshape(shape)
    return out


def _pack(values: np.ndarray) -> str:
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()
                            ).decode("ascii")


def _unpack(value: Mapping[str, Any]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(value["values"]), dtype="<f4"
                         ).astype(float).reshape(value["shape"])


def _grids(value: Mapping[str, Any]) -> List[np.ndarray]:
    count = LINE if len(value["axes"]) <= 1 else PLANE
    return [axis_grid(a, float(lo), float(hi), count)
            for a, (lo, hi) in zip(value["axes"], value["spans"])]


def _onto(value: Mapping[str, Any], axes: Sequence[str],
          grids: Sequence[np.ndarray]) -> np.ndarray:
    out = _unpack(value)
    own = list(value["axes"])
    for i, (axis, grid) in enumerate(zip(own, _grids(value))):
        target = grids[axes.index(axis)]
        if len(target) != len(grid) or not np.array_equal(target, grid):
            out = np.apply_along_axis(
                lambda row, grid=grid, target=target: np.interp(target, grid,
                                                                row),
                i, out)
    out = (np.transpose(out, [own.index(a) for a in axes if a in own])
           if own else out.reshape(()))
    return np.broadcast_to(
        out.reshape([len(g) if a in own else 1
                     for a, g in zip(axes, grids)]),
        tuple(len(g) for g in grids) or (1,))


def aligned(before: Mapping[str, Any], after: Mapping[str, Any]
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [a for a in AXES if a in set(before["axes"]) | set(after["axes"])]
    grids = []
    for axis in axes:
        sides = [v for v in (before, after) if axis in v["axes"]]
        side = max(sides, key=lambda v: len(v["axes"]))
        grids.append(_grids(side)[list(side["axes"]).index(axis)])
    return (_onto(before, axes, grids), _onto(after, axes, grids),
            cells(grids))


def changed_share(before: Mapping[str, Any],
                  after: Mapping[str, Any]) -> float:
    o, v, share = aligned(before, after)
    ok = np.isfinite(o) & np.isfinite(v)
    moved = np.sum(share * np.where(ok, np.abs(v - o), 0.0))
    push = np.sum(share * np.where(ok, np.maximum(np.abs(o), np.abs(v)),
                                   0.0))
    return float(moved / push) if push > 0.0 else 0.0


class Magnitude(Facet):
    name = "magnitude"
    question = "how much did the law's push change?"
    description = "the share of the push that changed, over the extent"

    def extract(self, analysis: LawAnalysis) -> Dict[str, Any]:
        return {"axes": list(analysis.axes),
                "spans": [[float(lo), float(hi)] for lo, hi in
                          analysis.spans],
                "shape": list(analysis.shape), "peak": analysis.peak,
                "values": _pack(analysis.values)}

    def compare(self, before: Any, after: Any) -> Change:
        share = changed_share(before, after)
        changed = share > DRIFT_FLOOR
        return Change(changed, min(1.0, share),
                      "%.3g of its push changed" % share if changed
                      else "same",
                      self.text(before), self.text(after), shifted=False,
                      drift=share)

    def text(self, value: Any) -> str:
        return "pushes up to %.4g" % value["peak"]


FACETS: Tuple[Facet, ...] = (Variables(), Sign(), Switching(), Magnitude())
SEVERITY: Tuple[str, ...] = tuple(f.name for f in FACETS)


__all__ = ["FACETS", "Facet", "JOINS", "Magnitude", "OFF", "ON", "SATURATED",
           "SEVERITY", "STATES", "Sign", "Switching", "Variables", "aligned",
           "cells", "changed_share", "family", "states"]
