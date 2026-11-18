from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import sympy as sp
from sympy.printing.numpy import NumPyPrinter

from .config import FamilyConfig, body_frame
from .record import ADD_VELOCITY, INPUTS, Bound, Record, Term


def _max(*args):
    out = np.asarray(args[0], dtype=float)
    for arg in args[1:]:
        arg = np.asarray(arg, dtype=float)
        out = np.where(arg > out, arg, out)
    return out


def _min(*args):
    out = np.asarray(args[0], dtype=float)
    for arg in args[1:]:
        arg = np.asarray(arg, dtype=float)
        out = np.where(arg < out, arg, out)
    return out


class _Printer(NumPyPrinter):
    def _print_Max(self, expr):
        return "Max(%s)" % ", ".join(self._print(a) for a in expr.args)

    def _print_Min(self, expr):
        return "Min(%s)" % ", ".join(self._print(a) for a in expr.args)


_MODULES = [{"Max": _max, "Min": _min}, "numpy"]


class Expression:
    def __init__(self, expr: sp.Expr):
        self.expr = expr
        self.inputs = tuple(name for name in INPUTS
                            if sp.Symbol(name) in expr.free_symbols)
        self.constant = None if self.inputs else float(expr)
        self.fn = (None if self.constant is not None else
                   sp.lambdify([sp.Symbol(n) for n in self.inputs], expr,
                               modules=_MODULES, printer=_Printer))

    def __call__(self, env: Mapping[str, np.ndarray],
                 shape: Tuple[int, ...]) -> np.ndarray:
        if self.constant is not None:
            return np.full(shape, self.constant)
        with np.errstate(all="ignore"):
            out = self.fn(*(env[name] for name in self.inputs))
        return np.broadcast_to(np.asarray(out, dtype=float), shape)


def norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(v * v, axis=-1))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)[..., None]
    with np.errstate(all="ignore"):
        return np.where(n > 0.0, v / n, 0.0)


def apply_bound(v: np.ndarray, bound: Bound) -> np.ndarray:
    n = norm(v)[..., None]
    with np.errstate(all="ignore"):
        if bound.kind == "renormalize":
            return np.where(n > 0.0, v * (bound.length / n), v)
        out = v
        if bound.hi is not None:
            out = np.where(n > bound.hi, v * (bound.hi / n), out)
        if bound.lo > 0.0:
            out = np.where(n < bound.lo,
                           np.where(n > 0.0, v * (bound.lo / n), np.nan), out)
    return out


def square_surface(point: np.ndarray, centre: np.ndarray, half: np.ndarray
                   ) -> np.ndarray:
    offset = point - centre
    half = half[..., None]
    nearest = centre + np.clip(offset, -half, half)
    inside = np.all(np.abs(offset) < half, axis=-1, keepdims=True)
    return np.where(inside, centre, nearest)


def body_surface(point: np.ndarray, centre: np.ndarray, radii: np.ndarray,
                 axes: Optional[np.ndarray] = None) -> np.ndarray:
    local, half, frame = body_frame(point - centre, radii, axes)
    if frame is None:
        return square_surface(point, centre, radii)
    along, across = frame
    clipped = np.clip(local, -half, half)
    nearest = centre + clipped[..., :1] * along + clipped[..., 1:] * across
    inside = np.all(np.abs(local) < half, axis=-1, keepdims=True)
    return np.where(inside, centre, nearest)


@dataclass
class View:
    positions: np.ndarray
    velocities: np.ndarray
    obstacles: np.ndarray
    obstacle_radii: np.ndarray
    targets: np.ndarray
    obstacle_axes: Optional[np.ndarray] = None


@dataclass
class Members:
    positions: np.ndarray
    velocities: np.ndarray
    sizes: np.ndarray

    @property
    def count(self) -> int:
        return int(self.positions.shape[1])


@dataclass
class Sensed:
    members: Members
    diff: np.ndarray
    keep: np.ndarray
    fired: np.ndarray
    env: Dict[str, np.ndarray]
    shill: Optional[np.ndarray] = None


def _concat(a: Members, b: Members) -> Members:
    return Members(np.concatenate([a.positions, b.positions], axis=1),
                   np.concatenate([a.velocities, b.velocities], axis=1),
                   np.concatenate([a.sizes, b.sizes], axis=1))


class _CompiledTerm:
    def __init__(self, term: Term, params: Mapping[str, Optional[float]]):
        self.term = term
        self.magnitude = Expression(term.magnitude)
        self.threshold = (None if term.gate is None
                          else Expression(term.gate.threshold))
        self.inputs = term.reads()
        self.range = self._param(term.range_param, params)
        self.fov = self._param(term.fov_param, params)
        self.cap = self._param(term.cap_param, params)

    @staticmethod
    def _param(name: Optional[str], params) -> Optional[float]:
        if not name:
            return None
        value = params.get(name)
        if value is None:
            raise ValueError("membership parameter %r has no value: name it "
                             "in the config's physics.record_params" % name)
        return float(value)


class Controller:
    def __init__(self, record: Record, config: FamilyConfig):
        self.record = record
        self.config = config
        self.geometry = config.physics.geometry
        self.agent_radius = config.physics.agent_radius
        self.terms = [_CompiledTerm(t, record.params) for t in record.terms]
        self.leader_sees_peers = (config.physics.targets.kind
                                  != "leader_formation")


    def _agents(self, i: int, view: View, role: str) -> Members:
        b, m = view.positions.shape[:2]
        if role == "leader" and not self.leader_sees_peers:
            return Members(np.zeros((b, 0, 2)), np.zeros((b, 0, 2)),
                           np.zeros((b, 0)))
        others = [j for j in range(m) if j != i]
        centres = view.positions[:, others]
        if self.geometry.peers == "square":
            half = np.full(centres.shape[:2], self.geometry.peer_half_size)
            positions = square_surface(view.positions[:, i:i + 1], centres,
                                       half)
        else:
            positions = centres
        sizes = np.full(centres.shape[:2], self.geometry.peer_size)
        return Members(positions, view.velocities[:, others], sizes)

    def _obstacles(self, i: int, view: View) -> Members:
        centres, radii = view.obstacles, view.obstacle_radii
        if self.geometry.obstacles == "square":
            positions = body_surface(view.positions[:, i:i + 1], centres,
                                     radii, view.obstacle_axes)
        else:
            positions = centres
        return Members(positions, np.zeros_like(centres), radii)

    def _members(self, source: str, i: int, view: View, role: str,
                 cache: Dict[str, Members]) -> Members:
        if source in cache:
            return cache[source]
        if source == "agents":
            out = self._agents(i, view, role)
        elif source == "obstacles":
            out = self._obstacles(i, view)
        elif source == "bodies":
            out = _concat(self._members("obstacles", i, view, role, cache),
                          self._members("agents", i, view, role, cache))
        elif source == "target":
            b = view.positions.shape[0]
            out = Members(view.targets[:, i:i + 1], np.zeros((b, 1, 2)),
                          np.zeros((b, 1)))
        else:
            raise ValueError("source %r" % source)
        cache[source] = out
        return out


    def _sense(self, c: _CompiledTerm, i: int, view: View, role: str,
               cache: Dict[str, Members]) -> Optional["Sensed"]:
        term = c.term
        b = view.positions.shape[0]
        if role != "agent" and term.role not in ("", "agent", role):
            return None
        members = self._members(term.source, i, view, role, cache)
        if members.count == 0:
            return None
        ap = view.positions[:, i]
        av = view.velocities[:, i]
        diff = members.positions - ap[:, None, :]
        dist = norm(diff)

        keep = np.ones(dist.shape, dtype=bool)
        if c.range is not None:
            keep &= ~(dist > c.range)
        if c.fov is not None:
            speed = norm(av)[:, None]
            with np.errstate(all="ignore"):
                cosine = np.sum(av[:, None, :] * diff, axis=-1) / (speed * dist)
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            angle = np.where((speed > 0.0) & (dist > 0.0), angle, 0.0)
            keep &= ~(angle > c.fov)
        if c.cap is not None and c.cap > 0:
            order = np.argsort(np.where(keep, dist, np.inf), axis=1,
                               kind="stable")
            rank = np.empty_like(order)
            np.put_along_axis(rank, order,
                              np.arange(dist.shape[1])[None, :].repeat(b, 0),
                              axis=1)
            keep &= rank < int(round(c.cap))
        if term.aggregator == "nearest" and term.direction != "around":
            pick = np.argmin(np.where(keep, dist, np.inf), axis=1)
            chosen = np.zeros_like(keep)
            np.put_along_axis(chosen, pick[:, None], True, axis=1)
            keep = chosen & np.any(keep, axis=1, keepdims=True)

        env: Dict[str, np.ndarray] = {"dist": dist}
        if "clearance" in c.inputs:
            clearance = self.geometry.clearance
            if clearance == "self":
                env["clearance"] = dist - self.agent_radius
            elif clearance == "other":
                env["clearance"] = dist - members.sizes
            else:
                env["clearance"] = dist
        if "rel_speed" in c.inputs or term.direction in ("friction", "match"):
            relative = members.velocities - av[:, None, :]
            env["rel_speed"] = norm(relative)
        shill = None
        if term.direction == "shill":
            shill = (term.signal_speed * unit(-diff)) - av[:, None, :]
            env["shill_speed"] = norm(shill)

        if term.gate is not None:
            threshold = c.threshold(env, dist.shape)
            feature = env[term.gate.feature]
            fires = (feature <= threshold) if term.gate.kind == "below" \
                else (feature > threshold)
            keep_fired = keep & fires
        else:
            keep_fired = keep
        return Sensed(members, diff, keep, keep_fired, env, shill)

    def _term(self, c: _CompiledTerm, i: int, view: View, role: str,
              cache: Dict[str, Members]) -> np.ndarray:
        term = c.term
        sensed = self._sense(c, i, view, role, cache)
        if sensed is None:
            return np.zeros((view.positions.shape[0], 2))
        members, diff, keep, keep_fired, env, shill = (
            sensed.members, sensed.diff, sensed.keep, sensed.fired,
            sensed.env, sensed.shill)
        ap = view.positions[:, i]
        av = view.velocities[:, i]
        direction = term.direction
        if direction == "around":
            bodies = self._members("obstacles", i, view, role, cache).count
            keep_fired = self._around_fired(term, env["dist"], keep, keep_fired,
                                            bodies)

        magnitude = c.magnitude(env, diff.shape[:2])
        if direction == "toward":
            vectors = unit(diff)
        elif direction in ("away_unit", "around"):
            vectors = unit(-diff)
        elif direction == "away_raw":
            vectors = -diff
        elif direction == "match":
            vectors = (members.velocities - av[:, None, :]
                       if term.signal == "velocity" else diff)
        elif direction == "friction":
            vectors = unit(members.velocities - av[:, None, :])
        elif direction in ("slide", "detour"):
            away = unit(-diff)
            along = np.stack([-away[..., 1], away[..., 0]], axis=-1)
            sense = np.sign(np.sum(along * av[:, None, :], axis=-1))
            vectors = along * np.where(sense == 0.0, 1.0, sense)[..., None]
            if direction == "detour":
                aim = unit(view.targets[:, i] - ap)[:, None, :]
                facing = np.clip(np.sum(-away * aim, axis=-1), 0.0, None)
                vectors = vectors * facing[..., None]
        else:
            vectors = unit(shill)
        with np.errstate(all="ignore"):
            contribution = magnitude[..., None] * vectors
        total = np.sum(np.where(keep_fired[..., None], contribution, 0.0),
                       axis=1)
        if term.aggregator == "mean":
            count = np.sum(keep, axis=1)[:, None]
            with np.errstate(all="ignore"):
                total = np.where(count > 0, total / np.maximum(count, 1), 0.0)
        if term.bound is not None:
            total = apply_bound(total, term.bound)
        if direction == "around":
            return term.weight * total + self._around(
                term, diff, sensed.env["dist"], keep, bodies, av)
        return term.weight * total

    @staticmethod
    def _around_fired(term: Term, dist: np.ndarray, keep: np.ndarray,
                      fired: np.ndarray, bodies: int) -> np.ndarray:
        def nearest(mask: np.ndarray, d: np.ndarray) -> np.ndarray:
            pick = np.argmin(np.where(mask, d, np.inf), axis=1)[:, None]
            out = np.zeros_like(mask)
            np.put_along_axis(out, pick, True, axis=1)
            return out & np.any(mask, axis=1, keepdims=True)
        out = np.zeros_like(fired)
        if bodies:
            out[:, :bodies] = (nearest(keep[:, :bodies], dist[:, :bodies])
                               & fired[:, :bodies])
        if dist.shape[1] > bodies:
            peers = keep[:, bodies:] & (dist[:, bodies:] < term.around[1])
            out[:, bodies:] = nearest(peers, dist[:, bodies:])
        return out

    @staticmethod
    def _around(term: Term, diff: np.ndarray, dist: np.ndarray,
                keep: np.ndarray, bodies: int, av: np.ndarray) -> np.ndarray:
        if bodies == 0:
            return np.zeros_like(av)
        size, reach = term.around
        near = np.where(keep[:, :bodies], dist[:, :bodies], np.inf)
        j = np.argmin(near, axis=1)[:, None]
        away = np.take_along_axis(unit(-diff), j[..., None], axis=1)[:, 0]
        along = np.stack([-away[:, 1], away[:, 0]], axis=-1)
        sense = np.sign(np.sum(along * av, axis=-1))
        sense = np.where(sense == 0.0, 1.0, sense)
        within = np.take_along_axis(near, j, axis=1)[:, 0] < reach
        return size * (sense * within)[:, None] * along


    def sense(self, i: int, view: View, role: str = "agent"
              ) -> Dict[str, Optional["Sensed"]]:
        cache: Dict[str, Members] = {}
        return {c.term.name: self._sense(c, i, view, role, cache)
                for c in self.terms}

    def command(self, i: int, view: View, role: str = "agent",
                terms: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        cache: Dict[str, Members] = {}
        force = np.zeros((view.positions.shape[0], 2))
        for c in self.terms:
            part = self._term(c, i, view, role, cache)
            if terms is not None:
                terms[c.term.name] = part
            force = force + part
        command = (view.velocities[:, i] + force
                   if self.record.base == ADD_VELOCITY else force)
        for bound in self.record.global_bounds:
            command = apply_bound(command, bound)
        return command


__all__ = ["Controller", "Expression", "Members", "Sensed", "View",
           "apply_bound", "body_surface", "norm", "square_surface", "unit"]
