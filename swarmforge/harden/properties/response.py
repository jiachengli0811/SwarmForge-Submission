from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from functools import cached_property
from typing import (Any, Callable, Dict, Iterator, List, Mapping, Optional,
                    Tuple)

import numpy as np
import sympy as sp
from sympy.core.relational import Relational

from ...solve.config import FamilyConfig
from ...solve.laws import Expression
from ...solve.record import Record, Term
from ..domain import CLEARANCE, DIST, REL_SPEED, SHILL_SPEED, Domain

AXES = (DIST, CLEARANCE, REL_SPEED, SHILL_SPEED)
DISTANCES = (DIST, CLEARANCE)
SPEEDS = (REL_SPEED, SHILL_SPEED)

LINE = 513
PLANE = 97

ZERO = 1e-9
JUMP = 1e-4
KINK = 1e-4
BISECTIONS = 64
ULPS = 4

SENSES: Mapping[str, Tuple[str, str]] = {
    "away_unit": ("repulsive", "attractive"),
    "away_raw": ("repulsive", "attractive"),
    "around": ("repulsive", "attractive"),
    "toward": ("attractive", "repulsive"),
    "friction": ("aligning", "anti-aligning"),
    "shill": ("braking", "anti-braking"),
}
TARGET_SENSES = ("goal-seeking", "goal-avoiding")


def sense_of(term: Term) -> Tuple[str, str]:
    if term.direction == "match":
        return (("aligning", "anti-aligning") if term.signal == "velocity"
                else ("attractive", "repulsive"))
    pair = SENSES[term.direction]
    if term.source == "target" and pair[0] in ("attractive", "repulsive"):
        return TARGET_SENSES if pair[0] == "attractive" else TARGET_SENSES[::-1]
    return pair


def acts_on_speed(term: Term) -> bool:
    return term.direction in ("friction", "shill") or (
        term.direction == "match" and term.signal == "velocity")


@dataclass(frozen=True)
class CommandScale:
    command: float
    kind: str
    per_tick: Mapping[str, float]

    @classmethod
    def of(cls, record: Record, config: FamilyConfig) -> "CommandScale":
        plant, dt = config.physics.plant, config.physics.dt
        top = plant.max_step / dt
        command, kind = None, "plant"
        for bound in record.global_bounds:
            if bound.kind == "renormalize":
                command, kind = float(bound.length), "renormalize"
            elif bound.hi is not None:
                command = (float(bound.hi) if command is None
                           else min(command, float(bound.hi)))
                kind = "clamp"
        if command is None:
            command = (plant.max_step if plant.integration == "displacement"
                       else top)
        closing = 2.0 * plant.max_step
        if plant.integration == "velocity" and plant.max_accel is not None:
            turning = 2.0 * plant.max_accel * dt
        else:
            turning = 2.0 * top
        return cls(command, kind, {DIST: closing, CLEARANCE: closing,
                                   REL_SPEED: turning, SHILL_SPEED: turning})

    def to_dict(self) -> Dict[str, Any]:
        return {"command": self.command, "kind": self.kind,
                "per_tick": dict(self.per_tick)}


def law_key(term: Term, offset: Optional[float] = None) -> str:
    gate = (None if term.gate is None else
            (term.gate.kind, term.gate.feature, sp.srepr(term.gate.threshold)))
    bound = (None if term.bound is None else
             (term.bound.kind, term.bound.hi, term.bound.lo,
              term.bound.length))
    payload = repr((term.name, sp.srepr(term.magnitude), gate,
                    float(term.weight), bound, term.direction, term.signal,
                    term.source, term.aggregator, term.range_param, offset))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def law_text(term: Term) -> str:
    gate = ("" if term.gate is None else " if %s %s %s" % (
        term.gate.feature, "<=" if term.gate.kind == "below" else ">",
        term.gate.threshold))
    if term.direction == "away_raw" or (term.direction == "match"
                                        and term.signal != "velocity"):
        length = "*dist"
    elif term.direction == "match":
        length = "*rel_speed"
    else:
        length = ""
    bound = ("" if term.bound is None else " clamped %s" % (
        term.bound.hi if term.bound.kind == "limit"
        else "to %s" % term.bound.length))
    return "%s: %g*(%s)%s%s%s" % (term.direction, term.weight,
                                  term.magnitude, length, bound, gate)


class LawSubject:
    def __init__(self, term: Term, domain: Optional[Domain],
                 scale: "CommandScale"):
        self.term, self.domain, self.scale = term, domain, scale
        self.name = term.name
        self.key = law_key(term, None if domain is None else domain.offset)

    def text(self) -> str:
        return law_text(self.term)

    def analysis(self) -> "LawAnalysis":
        if self.domain is None:
            raise ValueError("law %r has no domain: it is not a law of the "
                             "original" % self.name)
        return LawAnalysis(LawResponse(self.term, self.domain, self.scale))


def _shape(point: Mapping[str, np.ndarray]) -> Tuple[int, ...]:
    for value in point.values():
        return np.shape(value)
    return (1,)


def _relations(expr: sp.Basic) -> List[sp.Expr]:
    out: List[sp.Expr] = []
    for relation in expr.atoms(Relational):
        out.append(relation.lhs - relation.rhs)
    for node in sp.preorder_traversal(expr):
        if isinstance(node, (sp.Max, sp.Min)):
            for a, b in itertools.combinations(node.args, 2):
                out.append(a - b)
        elif isinstance(node, (sp.Abs, sp.sign)):
            out.append(node.args[0])
    return [g for g in out if g.free_symbols]


class LawResponse:
    def __init__(self, term: Term, domain: Domain, scale: CommandScale):
        self.term, self.domain, self.scale = term, domain, scale
        self.offset = domain.offset
        self.magnitude = Expression(term.magnitude)
        self.threshold = (None if term.gate is None
                          else Expression(term.gate.threshold))
        if term.direction == "away_raw" or (term.direction == "match"
                                            and term.signal != "velocity"):
            self.length: Optional[str] = DIST
        elif term.direction == "match":
            self.length = REL_SPEED
        else:
            self.length = None
        reads = set(self.magnitude.inputs)
        if self.threshold is not None:
            reads |= set(self.threshold.inputs) | {term.gate.feature}
        if self.length:
            reads.add(self.length)
        if self.offset is not None and CLEARANCE in reads:
            reads.discard(CLEARANCE)
            reads.add(DIST)
        self.axes: Tuple[str, ...] = tuple(a for a in AXES if a in reads)
        self.extent: Dict[str, Tuple[float, float]] = {}
        for axis in self.axes:
            span = domain.reach_span(axis)
            if span is None:
                raise ValueError("law %r reads %s, which its domain does not "
                                 "span" % (term.name, axis))
            self.extent[axis] = span
        self.senses = sense_of(term)
        preferred = SPEEDS if acts_on_speed(term) else DISTANCES
        self.primary: Optional[str] = next(
            (a for a in preferred if a in self.axes),
            self.axes[0] if self.axes else None)
        self.distance: Optional[str] = next(
            (a for a in DISTANCES if a in self.axes), None)

    @property
    def key(self) -> str:
        return law_key(self.term, self.offset)

    def text(self) -> str:
        return law_text(self.term)


    def env(self, point: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        env = {k: np.asarray(v, dtype=float) for k, v in point.items()}
        if self.offset is not None and DIST in env:
            env[CLEARANCE] = env[DIST] - self.offset
        return env

    def _scaled(self, env, shape) -> np.ndarray:
        value = self.magnitude(env, shape)
        if self.length is not None:
            value = value * env[self.length]
        return np.asarray(value, dtype=float)

    def _fires(self, env, shape) -> np.ndarray:
        threshold = self.threshold(env, shape)
        feature = env[self.term.gate.feature]
        if self.term.gate.kind == "below":
            return feature <= threshold
        return feature > threshold

    def __call__(self, point: Mapping[str, np.ndarray]) -> np.ndarray:
        env, shape = self.env(point), _shape(point)
        term = self.term
        with np.errstate(all="ignore"):
            m = self._scaled(env, shape)
            bound = term.bound
            if bound is not None:
                if bound.kind == "renormalize":
                    m = np.where(m != 0.0, bound.length * np.sign(m), m)
                else:
                    if bound.hi is not None:
                        m = np.clip(m, -bound.hi, bound.hi)
                    if bound.lo > 0.0:
                        m = np.where(np.abs(m) < bound.lo,
                                     np.where(m != 0.0,
                                              np.sign(m) * bound.lo, np.nan),
                                     m)
            r = term.weight * m
            if self.threshold is not None:
                r = np.where(self._fires(env, shape), r, 0.0)
        return np.broadcast_to(np.asarray(r, dtype=float), shape).copy()

    def clamped(self, point: Mapping[str, np.ndarray]) -> np.ndarray:
        bound = self.term.bound
        if bound is None or bound.kind != "limit" or bound.hi is None:
            return np.zeros(_shape(point), dtype=bool)
        env, shape = self.env(point), _shape(point)
        with np.errstate(all="ignore"):
            return np.abs(self._scaled(env, shape)) > bound.hi

    def switch_functions(self) -> List[Tuple[str, Callable]]:
        out: List[Tuple[str, Callable]] = []
        seen = set()
        sources = [("magnitude", self.term.magnitude)]
        if self.term.gate is not None:
            sources.append(("threshold", self.term.gate.threshold))
        for where, expr in sources:
            for g in _relations(expr):
                key = sp.srepr(g)
                if key in seen:
                    continue
                seen.add(key)
                compiled = Expression(g)

                def evaluate(point, compiled=compiled):
                    return compiled(self.env(point), _shape(point))
                out.append(("%s: %s" % (where, g), evaluate))
        if self.term.gate is not None:
            def gate(point):
                env, shape = self.env(point), _shape(point)
                return env[self.term.gate.feature] - self.threshold(env, shape)
            out.append(("gate", gate))
        bound = self.term.bound
        if bound is not None and bound.kind == "limit" and bound.hi is not None:
            def clamp(point):
                env, shape = self.env(point), _shape(point)
                return np.abs(self._scaled(env, shape)) - bound.hi
            out.append(("bound", clamp))
        elif bound is not None and bound.kind == "renormalize":
            def flip(point):
                env, shape = self.env(point), _shape(point)
                return self._scaled(env, shape)
            out.append(("bound", flip))
        return out


def axis_grid(axis: str, lo: float, hi: float, count: int) -> np.ndarray:
    linear = np.linspace(lo, hi, count // 2 + 1 if axis in DISTANCES
                         else count)
    if axis not in DISTANCES or hi <= lo:
        return linear
    start = lo if lo > 0.0 else hi * 1e-4
    if hi / start < 10.0:
        return np.linspace(lo, hi, count)
    geometric = np.geomspace(start, hi, count // 2)
    return np.unique(np.concatenate([linear, geometric[geometric >= lo]]))


def _beyond(values: np.ndarray, direction: float) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    limit = np.inf if direction > 0 else -np.inf
    for _ in range(ULPS):
        out = np.nextafter(out, limit)
    return out


@dataclass(frozen=True)
class Switch:
    axis: str
    at: float
    kind: str
    size: float
    role: str
    where: Mapping[str, float] = field(default_factory=dict)
    source: str = ""
    index: int = 0
    line: Tuple[int, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"axis": self.axis, "at": self.at, "kind": self.kind,
                "size": self.size, "role": self.role,
                "where": dict(self.where), "source": self.source}


class LawAnalysis:
    def __init__(self, response: LawResponse):
        self.response = response
        self.axes = response.axes
        self.spans = tuple(response.extent[a] for a in self.axes)
        count = LINE if len(self.axes) <= 1 else PLANE
        self.grids: Tuple[np.ndarray, ...] = tuple(
            axis_grid(a, lo, hi, count) for a, (lo, hi) in
            zip(self.axes, self.spans))
        self.shape: Tuple[int, ...] = tuple(len(g) for g in self.grids) or (1,)
        if self.axes:
            mesh = np.meshgrid(*self.grids, indexing="ij")
            self.points = {a: m.ravel() for a, m in zip(self.axes, mesh)}
        else:
            self.points = {}
        values = response(self.points) if self.axes else response({})
        self.values = np.asarray(values, dtype=float).reshape(self.shape)
        finite = self.values[np.isfinite(self.values)]
        self.peak = float(np.max(np.abs(finite))) if finite.size else 0.0
        self.eps = max(ZERO * self.peak, 1e-300)
        self.support = np.isfinite(self.values) & (np.abs(self.values)
                                                   > self.eps)
        self.undefined = int(np.sum(~np.isfinite(self.values)))

    @property
    def command(self) -> float:
        return self.response.scale.command

    def width(self, axis: int) -> float:
        lo, hi = self.spans[axis]
        return hi - lo

    def lines(self, axis: int) -> Iterator[Tuple[Tuple[int, ...],
                                                 Dict[str, float],
                                                 np.ndarray, np.ndarray]]:
        others = [i for i in range(len(self.axes)) if i != axis]
        for fixed in itertools.product(*(range(self.shape[i])
                                         for i in others)):
            index: List[Any] = [slice(None)] * len(self.axes)
            for i, k in zip(others, fixed):
                index[i] = k
            where = {self.axes[i]: float(self.grids[i][k])
                     for i, k in zip(others, fixed)}
            yield (tuple(fixed), where, self.values[tuple(index)],
                   self.support[tuple(index)])

    def _at(self, axis: int, along: np.ndarray,
            lines: np.ndarray) -> Dict[str, np.ndarray]:
        out = {self.axes[axis]: np.asarray(along, dtype=float)}
        others = [i for i in range(len(self.axes)) if i != axis]
        for column, i in enumerate(others):
            out[self.axes[i]] = self.grids[i][lines[:, column]]
        return out

    @cached_property
    def switches(self) -> List[Switch]:
        if not self.axes or self.peak == 0.0:
            return []
        found: List[Switch] = []
        for source, g in self.response.switch_functions():
            try:
                with np.errstate(all="ignore"):
                    values = np.asarray(g(self.points), dtype=float)
                values = np.broadcast_to(values, (int(np.prod(self.shape)),)
                                         ).reshape(self.shape)
            except (ValueError, TypeError, ArithmeticError, KeyError,
                    NameError):
                continue
            for axis in range(len(self.axes)):
                found.extend(self._switches_along(axis, values, source, g))
        found.extend(self._sensing_cutoff())
        return self._dedupe(found)

    def _sensing_cutoff(self) -> List[Switch]:
        r = self.response
        if r.domain.sensing is None or r.distance is None:
            return []
        axis = self.axes.index(r.distance)
        last = self.shape[axis] - 1
        out = []
        for line, where, values, support in self.lines(axis):
            if support[last]:
                out.append(Switch(
                    axis=r.distance, at=float(self.grids[axis][last]),
                    kind="jump", size=float(abs(values[last]) / self.command),
                    role="onset", where=where, source="sensing range",
                    index=last - 1, line=line))
        return out

    def _switches_along(self, axis: int, g_values: np.ndarray, source: str,
                        g: Callable) -> List[Switch]:
        moved = np.moveaxis(g_values, axis, -1)
        side = moved >= 0.0
        flips = (np.isfinite(moved[..., :-1]) & np.isfinite(moved[..., 1:])
                 & (side[..., :-1] != side[..., 1:]))
        where = np.argwhere(flips)
        if not where.size:
            return []
        grid = self.grids[axis]
        lines, index = where[:, :-1], where[:, -1]
        lo, hi = grid[index].astype(float), grid[index + 1].astype(float)
        start = moved[tuple(where.T)] >= 0.0
        for _ in range(BISECTIONS):
            mid = 0.5 * (lo + hi)
            if np.all((mid <= lo) | (mid >= hi)):
                break
            with np.errstate(all="ignore"):
                value = np.asarray(g(self._at(axis, mid, lines)), dtype=float)
            same = ~np.isnan(value) & ((value >= 0.0) == start)
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        width = self.width(axis)
        tick = self.response.scale.per_tick[self.axes[axis]]
        step = max(width * 1e-7, 1e-12)
        pad = ZERO * width
        left = np.minimum(_beyond(lo, -1), lo - pad)
        right = np.maximum(_beyond(hi, +1), hi + pad)
        r = self.response
        r_left = r(self._at(axis, left, lines))
        r_right = r(self._at(axis, right, lines))
        r_left2 = r(self._at(axis, left - step, lines))
        r_right2 = r(self._at(axis, right + step, lines))
        support = np.moveaxis(self.support, axis, -1)
        acts_below = support[tuple(where.T)]
        acts_above = support[tuple(np.column_stack([lines, index + 1]).T)]
        out = []
        peak = self.peak
        for k in range(len(lo)):
            if not all(np.isfinite([r_left[k], r_right[k], r_left2[k],
                                    r_right2[k]])):
                continue
            jump = abs(r_right[k] - r_left[k])
            slope_left = (r_left[k] - r_left2[k]) / step
            slope_right = (r_right2[k] - r_right[k]) / step
            bend = abs(slope_right - slope_left)
            if jump > JUMP * peak:
                kind, size = "jump", jump / self.command
            elif bend * width > KINK * peak:
                kind = "kink"
                size = bend * tick / self.command
            else:
                continue
            line = tuple(int(i) for i in lines[k])
            others = [i for i in range(len(self.axes)) if i != axis]
            out.append(Switch(
                axis=self.axes[axis], at=float(0.5 * (lo[k] + hi[k])),
                kind=kind, size=float(size),
                role=("onset" if bool(acts_below[k]) != bool(acts_above[k])
                      else "interior"),
                where={self.axes[i]: float(self.grids[i][j])
                       for i, j in zip(others, line)},
                source=source, index=int(index[k]), line=line))
        return out

    @staticmethod
    def _dedupe(found: List[Switch]) -> List[Switch]:
        rank = {"jump": 1, "kink": 0}
        kept: Dict[Tuple[str, Tuple[int, ...], int], Switch] = {}
        for switch in found:
            key = (switch.axis, switch.line, switch.index)
            held = kept.get(key)
            if held is None or (rank[switch.kind], switch.size) > (
                    rank[held.kind], held.size):
                kept[key] = switch
        return sorted(kept.values(), key=lambda s: (s.axis, s.line, s.at))

    def smooth(self, axis: int) -> np.ndarray:
        mask = np.moveaxis(self.support.copy(), axis, -1)
        for switch in self.switches:
            if switch.axis != self.axes[axis]:
                continue
            mask[switch.line + (switch.index,)] = False
            mask[switch.line + (switch.index + 1,)] = False
        return np.moveaxis(mask, -1, axis)

    def derivatives(self, axis: int) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray]:
        x = self.grids[axis]
        values = np.moveaxis(self.values, axis, -1)
        mask = np.moveaxis(self.smooth(axis), axis, -1)
        first = np.full(values.shape, np.nan)
        second = np.full(values.shape, np.nan)
        if len(x) >= 3:
            h_left = x[1:-1] - x[:-2]
            h_right = x[2:] - x[1:-1]
            left = (values[..., 1:-1] - values[..., :-2]) / h_left
            right = (values[..., 2:] - values[..., 1:-1]) / h_right
            first[..., 1:-1] = (values[..., 2:] - values[..., :-2]) / (
                h_left + h_right)
            second[..., 1:-1] = 2.0 * (right - left) / (h_left + h_right)
        valid = np.zeros(values.shape, dtype=bool)
        valid[..., 1:-1] = mask[..., :-2] & mask[..., 1:-1] & mask[..., 2:]
        return (np.moveaxis(first, -1, axis), np.moveaxis(second, -1, axis),
                np.moveaxis(valid, -1, axis))


def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    out, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


__all__ = ["AXES", "CommandScale", "DISTANCES", "LawAnalysis", "LawResponse",
           "LawSubject", "SENSES", "SPEEDS", "Switch", "acts_on_speed",
           "axis_grid", "law_key", "law_text", "runs", "sense_of"]
