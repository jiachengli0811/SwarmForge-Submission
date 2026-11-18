from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

from swarmforge.solve.dynamics import MOVER_MARGIN
from swarmforge.solve.record import ADD_VELOCITY
from swarmforge.solve.state import State
from swarmforge.solve.verdict import Verdict

from . import interval as iv
from .expr import compile_truth, compile_value
from .interval import I, T

UNDECIDED = -1

_ROOT3 = math.sqrt(3.0)
_SLOTS = {2: (-_ROOT3 / 2, 0.5), 3: (-_ROOT3 / 2, -0.5), 4: (-_ROOT3, 0.0),
          5: (-_ROOT3, 1.0), 6: (-_ROOT3, -1.0), 7: (-1.5 * _ROOT3, -0.5),
          8: (-1.5 * _ROOT3, 0.5), 9: (-2 * _ROOT3, 0.0)}
_TABLE = {1: [], 2: [4], 3: [2, 3], 4: [2, 3, 4], 5: [2, 3, 4, 5],
          6: [2, 3, 4, 5, 6], 7: [2, 3, 4, 5, 6, 7],
          8: [2, 3, 4, 5, 6, 7, 8], 9: [2, 3, 4, 5, 6, 7, 8, 9]}


@dataclass
class Boxes:
    positions: I
    velocities: I
    obstacles: np.ndarray
    obstacle_velocities: np.ndarray
    obstacle_radii: np.ndarray
    obstacle_axes: np.ndarray
    dynamic: np.ndarray
    target: np.ndarray
    route: np.ndarray

    @property
    def size(self) -> int:
        return int(self.target.shape[0])

    @property
    def drones(self) -> int:
        return int(self.positions.lo.shape[1])

    @classmethod
    def around(cls, state: State, position_half=0.0, velocity_half=0.0
               ) -> "Boxes":
        p, v = state.positions, state.velocities
        ph = np.broadcast_to(np.asarray(position_half, dtype=float), p.shape)
        vh = np.broadcast_to(np.asarray(velocity_half, dtype=float), v.shape)
        return cls(I(p - ph, p + ph), I(v - vh, v + vh), state.obstacles,
                   state.obstacle_velocities, state.obstacle_radii,
                   state.obstacle_axes, state.dynamic, state.target,
                   state.route)


@dataclass
class World:
    P: List[I]
    V: List[I]
    O: I
    targets: List[I]
    S: Optional[List[Optional[I]]] = None

    def positions(self) -> I:
        return iv.stack(self.P, axis=1)

    def velocities(self) -> I:
        return iv.stack(self.V, axis=1)


class _Term:
    def __init__(self, term, params):
        self.term = term
        self.magnitude = compile_value(term.magnitude)
        self.threshold = (None if term.gate is None
                          else compile_value(term.gate.threshold))
        self.inputs = term.reads()
        self.range = self._param(term.range_param, params)
        self.fov = self._param(term.fov_param, params)
        if term.cap_param and params.get(term.cap_param):
            raise NotImplementedError("a nearest-k cap (%s)" % term.name)
        if term.direction not in ("toward", "away_unit", "away_raw", "match",
                                  "friction", "shill"):
            raise NotImplementedError("direction %r (%s)"
                                      % (term.direction, term.name))
        length = {"toward": "dist", "away_unit": "dist",
                  "friction": "rel_speed", "shill": "shill_speed"
                  }.get(term.direction)
        self.linear: Optional[float] = None
        if (length is not None and not term.magnitude.has(sp.Piecewise)
                and term.magnitude.free_symbols == {sp.Symbol(length)}):
            ratio = sp.cancel(term.magnitude / sp.Symbol(length))
            if ratio.is_Number:
                self.linear = float(ratio)

    @staticmethod
    def _param(name, params) -> Optional[float]:
        if not name:
            return None
        value = params.get(name)
        if value is None:
            raise ValueError("membership parameter %r has no value" % name)
        return float(value)


def _pairs(m: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(m, 1)


def _minimax(d: np.ndarray) -> np.ndarray:
    out = d.copy()
    for k in range(out.shape[1]):
        out = np.minimum(out, np.maximum(out[:, :, k:k + 1], out[:, k:k + 1, :]))
    return out


class Engine:
    def __init__(self, model):
        if model.spec.horizon != 1:
            raise NotImplementedError("only horizon 1 is modelled, and %s/%s "
                                      "has %d" % (model.config.name, model.name,
                                                  model.spec.horizon))
        self.model = model
        self.name = model.name
        self.config = model.config
        self.physics = model.config.physics
        self.record = model.record
        self.reach = model.reach
        self.geometry = self.physics.geometry
        self.terms = [_Term(t, self.record.params) for t in self.record.terms]
        self.leader_formation = self.physics.targets.kind == "leader_formation"
        plant = self.physics.plant
        self.accel = (plant.max_accel if plant.integration == "velocity"
                      else None)
        for bound in self.record.global_bounds:
            self._check_bound(bound)


    @staticmethod
    def _frame(radii: np.ndarray, axes: np.ndarray):
        length = np.sqrt(np.sum(axes * axes, axis=-1))
        wall = length > 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            along = np.where(wall[..., None], axes / length[..., None],
                             np.array([1.0, 0.0]))
        across = np.stack([-along[..., 1], along[..., 0]], axis=-1)
        half = np.stack([np.where(wall, length, radii),
                         np.broadcast_to(radii, length.shape)], axis=-1)
        return along, across, half

    @staticmethod
    def _local(offset: I, along: np.ndarray, across: np.ndarray) -> I:
        return iv.stack([offset[..., 0] * along[..., 0]
                         + offset[..., 1] * along[..., 1],
                         offset[..., 0] * across[..., 0]
                         + offset[..., 1] * across[..., 1]], axis=-1)

    def body_gap(self, points: I, centres: I, radii: np.ndarray,
                 axes: np.ndarray) -> I:
        physics = self.physics
        offset = points - centres
        if physics.square_bodies:
            along, across, half = self._frame(radii, axes)
            local = self._local(offset, along, across)
            outside = iv.maximum(iv.absolute(local) - half, 0.0)
            surface = iv.norm(outside)
        else:
            surface = iv.norm(offset) - radii
        return surface - (physics.agent_radius + physics.obstacle_min_gap)

    @staticmethod
    def _to_surface(offset: I, half) -> I:
        def f(x):
            return np.clip(x, -half, half) - x
        return I(iv._dn(f(offset.hi)), iv._up(f(offset.lo)))

    def _surface_diff(self, offset: I, radii: np.ndarray, axes: np.ndarray) -> I:
        along, across, half = self._frame(radii, axes)
        local = self._local(offset, along, across)
        to = self._to_surface(local, half)
        nearest = iv.stack([to[..., 0] * along[..., 0] + to[..., 1] * across[..., 0],
                            to[..., 0] * along[..., 1] + to[..., 1] * across[..., 1]],
                           axis=-1)
        inside = iv.lt(iv.absolute(local), half).all(axis=-1)
        return iv.select(inside[..., None], -offset, nearest)


    def movers(self, boxes: Boxes) -> I:
        dt = self.physics.dt
        step = dt * boxes.obstacle_velocities
        ranges = self.config.state.dynamic
        standoff = ranges.standoff if ranges is not None else 0.0
        if not standoff or not np.any(boxes.dynamic):
            return I(boxes.obstacles + step)
        agent = self.physics.agent_radius
        keep = np.maximum(standoff, (boxes.obstacle_radii + agent)
                          * (1.0 + MOVER_MARGIN))
        rel = (I(boxes.obstacles[:, :, None, :])
               - iv.expand(boxes.positions, 1))
        now2 = iv.sqr(rel[..., 0]) + iv.sqr(rel[..., 1])
        c = iv.maximum(now2 - (keep * keep)[:, :, None], 0.0)
        a = np.sum(step * step, axis=-1)[:, :, None]
        b = (rel[..., 0] * step[:, :, None, 0]
             + rel[..., 1] * step[:, :, None, 1]) * 2.0
        disc = iv.sqr(b) - c * (4.0 * a)
        safe = np.where(a > 0.0, a, 1.0)
        enter = (-b - iv.sqrt(disc)) / (2.0 * safe)
        closing = T.of(a > 0.0) & iv.lt(b, 0.0) & iv.gt(disc, 0.0)
        limit = iv.select(closing, iv.clip(enter, 0.0, 1.0), 1.0)
        limit = I(limit.lo.min(axis=2), limit.hi.min(axis=2))
        moving = boxes.dynamic[None, :]
        limit = I(np.where(moving, limit.lo, 1.0), np.where(moving, limit.hi, 1.0))
        return I(boxes.obstacles) + iv.expand(limit, -1) * step


    def _agents(self, i: int, w: World, role: str):
        m = len(w.P)
        if role == "leader" and self.leader_formation:
            return None
        others = [j for j in range(m) if j != i]
        if not others:
            return None
        centres = iv.stack([w.P[j] for j in others], axis=1)
        here = iv.expand(w.P[i], 1)
        if self.geometry.peers == "square":
            half = self.geometry.peer_half_size
            radii = np.full(centres.lo.shape[:2], half)
            diff = self._surface_diff(here - centres, radii,
                                      np.zeros(centres.lo.shape))
        else:
            diff = centres - here
        velocities = iv.stack([w.V[j] for j in others], axis=1)
        sizes = np.full(centres.lo.shape[:2], self.geometry.peer_size)
        return diff, velocities, sizes

    def _obstacles(self, i: int, w: World, boxes: Boxes):
        if boxes.obstacles.shape[1] == 0:
            return None
        here = iv.expand(w.P[i], 1)
        if self.geometry.obstacles == "square":
            diff = self._surface_diff(here - w.O, boxes.obstacle_radii,
                                      boxes.obstacle_axes)
        else:
            diff = w.O - here
        return diff, iv.zeros(diff.lo.shape), boxes.obstacle_radii

    def _members(self, source: str, i: int, w: World, role: str,
                 boxes: Boxes, cache: Dict):
        if source in cache:
            return cache[source]
        if source == "agents":
            out = self._agents(i, w, role)
        elif source == "obstacles":
            out = self._obstacles(i, w, boxes)
        elif source == "bodies":
            parts = [p for p in (self._members("obstacles", i, w, role, boxes, cache),
                                 self._members("agents", i, w, role, boxes, cache))
                     if p is not None]
            out = None if not parts else (
                iv.concat([p[0] for p in parts], axis=1),
                iv.concat([p[1] for p in parts], axis=1),
                np.concatenate([np.broadcast_to(p[2], p[0].lo.shape[:2])
                                for p in parts], axis=1))
        elif source == "target":
            diff = iv.expand(w.targets[i] - w.P[i], 1)
            out = (diff, iv.zeros(diff.lo.shape), np.zeros(diff.lo.shape[:2]))
        else:
            raise NotImplementedError("source %r" % source)
        cache[source] = out
        return out

    @staticmethod
    def _nearest(dist: I, keep: T) -> T:
        m = dist.lo.shape[1]
        other = ~np.eye(m, dtype=bool)[None]
        ahead = keep.no[:, None, :] | (dist.hi[:, :, None] < dist.lo[:, None, :])
        yes = keep.yes & np.all(ahead | ~other, axis=2)
        behind = keep.yes[:, None, :] & (dist.hi[:, None, :] < dist.lo[:, :, None])
        no = keep.no | np.any(behind & other, axis=2)
        return T(yes, no)

    def _check_bound(self, bound) -> None:
        if bound.kind != "renormalize" and bound.lo > 0.0:
            raise NotImplementedError("a clamp with a floor")

    def _bound(self, v: I, bound) -> I:
        if bound.kind == "renormalize":
            return iv.unit(v) * bound.length
        self._check_bound(bound)
        if bound.hi is None:
            return v
        over = iv.gt(iv.norm(v), bound.hi)
        return iv.select(over[..., None], iv.unit(v) * bound.hi, v)

    def _term(self, c: _Term, i: int, w: World, role: str, boxes: Boxes,
              cache: Dict) -> Optional[I]:
        term = c.term
        if role != "agent" and term.role not in ("", "agent", role):
            return None
        members = self._members(term.source, i, w, role, boxes, cache)
        if members is None:
            return None
        diff, velocities, sizes = members
        av = iv.expand(w.V[i], 1)
        dist = iv.norm(diff)
        shape = dist.lo.shape

        keep = T.of(np.ones(shape, dtype=bool))
        if c.range is not None:
            keep = keep & ~iv.gt(dist, c.range)
        if c.fov is not None and c.fov < 180.0:
            speed = iv.norm(av)
            cosine = iv.dot(av, diff) / (speed * dist)
            angle = iv.arccos(cosine) * (180.0 / math.pi)
            wide = (iv.gt(angle, c.fov) & iv.gt(speed, 0.0) & iv.gt(dist, 0.0))
            keep = keep & ~wide
        if term.aggregator == "nearest":
            keep = self._nearest(dist, keep)

        env = {"dist": dist}
        if "clearance" in c.inputs:
            kind = self.geometry.clearance
            if kind == "self":
                env["clearance"] = dist - self.physics.agent_radius
            elif kind == "other":
                env["clearance"] = dist - sizes
            else:
                env["clearance"] = dist
        relative = None
        if "rel_speed" in c.inputs or term.direction in ("friction", "match"):
            relative = velocities - av
            env["rel_speed"] = iv.norm(relative)
        shill = None
        if term.direction == "shill":
            shill = iv.unit(-diff) * term.signal_speed - av
            env["shill_speed"] = iv.norm(shill)

        fired = keep
        if term.gate is not None:
            threshold = c.threshold(env)
            feature = env[term.gate.feature]
            fires = (iv.le(feature, threshold) if term.gate.kind == "below"
                     else iv.gt(feature, threshold))
            fired = keep & fires

        direction = term.direction
        raw = {"toward": diff, "away_unit": -diff, "away_raw": -diff,
               "friction": relative, "shill": shill}.get(direction)
        if direction == "match":
            raw = relative if term.signal == "velocity" else diff
        if c.linear is not None:
            contribution = raw * c.linear
        else:
            magnitude = c.magnitude(env).broadcast_to(shape)
            vectors = raw if direction in ("away_raw", "match") else iv.unit(raw)
            contribution = vectors * iv.expand(magnitude, -1)
        total = iv.total(iv.select(fired[..., None], contribution, 0.0), axis=1)
        if term.aggregator == "mean":
            count = keep.count(axis=1)
            some = count.lo > 0
            none = count.hi == 0
            shared = total / I(np.maximum(count.lo, 1.0),
                               np.maximum(count.hi, 1.0))[:, None]
            total = I(np.where(none[:, None], 0.0,
                               np.where(some[:, None], shared.lo,
                                        np.minimum(shared.lo, 0.0))),
                      np.where(none[:, None], 0.0,
                               np.where(some[:, None], shared.hi,
                                        np.maximum(shared.hi, 0.0))))
        if term.bound is not None:
            total = self._bound(total, term.bound)
        return total * term.weight

    def command(self, i: int, w: World, role: str, boxes: Boxes) -> I:
        cache: Dict = {}
        force = iv.zeros((boxes.size, 2))
        for c in self.terms:
            part = self._term(c, i, w, role, boxes, cache)
            if part is not None:
                force = force + part
        command = w.V[i] + force if self.record.base == ADD_VELOCITY else force
        for bound in self.record.global_bounds:
            command = self._bound(command, bound)
        return command


    def _apply(self, w: World, i: int, command: I) -> None:
        plant, dt = self.physics.plant, self.physics.dt
        if plant.integration == "velocity":
            velocity = command
            if plant.max_accel is not None:
                old = w.V[i]
                accel = (command - old) / dt
                over = iv.gt(iv.norm(accel), plant.max_accel)
                limited = old + iv.unit(accel) * (dt * plant.max_accel)
                velocity = iv.select(over[..., None], limited, command)
            w.V[i] = velocity
            step = velocity * dt
        else:
            step = command
            w.V[i] = command / dt
        w.P[i] = w.P[i] + step
        w.S[i] = step

    def _slots(self, w: World, boxes: Boxes) -> None:
        targets = self.physics.targets
        goal = I(boxes.target)
        v = iv.unit(goal - w.P[0])
        u = iv.perp(v)
        spacing = targets.formation_spacing
        names = _TABLE.get(targets.formation_robots, [2, 3, 4])
        for i in range(1, len(w.P)):
            along, across = _SLOTS[names[i - 1]]
            w.targets[i] = goal + v * (along * spacing) + u * (across * spacing)

    def tick(self, boxes: Boxes) -> World:
        m = boxes.drones
        goal = I(boxes.target)
        w = World(P=[boxes.positions[:, i] for i in range(m)],
                  V=[boxes.velocities[:, i] for i in range(m)],
                  O=self.movers(boxes), targets=[goal] * m, S=[None] * m)
        roles = (["leader"] + ["follower"] * (m - 1) if self.leader_formation
                 else ["agent"] * m)
        if self.physics.plant.update_order == "synchronous":
            commands = [self.command(i, w, roles[i], boxes) for i in range(m)]
            for i, command in enumerate(commands):
                self._apply(w, i, command)
        else:
            for i in range(m):
                if self.leader_formation and i == 1:
                    self._slots(w, boxes)
                self._apply(w, i, self.command(i, w, roles[i], boxes))
        return w


    def _discs(self, boxes: Boxes, offsets: bool = False):
        step, dt = self.physics.plant.max_step, self.physics.dt
        out = [(boxes.positions, step, None)]
        if self.accel is not None:
            drift = boxes.velocities * dt
            out.append((boxes.positions + drift, self.accel * dt * dt, drift))
        return out if offsets else [(c, r) for c, r, _ in out]

    @staticmethod
    def _gain(a: I, step: Optional[I]) -> I:
        if step is None:
            return iv.zeros(a.lo.shape[:-1])
        b = a + step
        na, nb = iv.norm(a), iv.norm(b)
        plain = na - nb
        fine = -iv.dot(step, a * 2.0 + step) / (na + nb)
        return I(np.maximum(plain.lo, fine.lo), np.minimum(plain.hi, fine.hi))

    def admissible(self, boxes: Boxes) -> T:
        m = boxes.drones
        ok = T.of(np.ones(boxes.size, dtype=bool))
        if m > 1:
            first, second = _pairs(m)
            d = iv.norm(boxes.positions[:, first] - boxes.positions[:, second])
            ok = ok & iv.ge(d, self.physics.agent_boundary).all(axis=1)
        if boxes.obstacles.shape[1]:
            gap = self.body_gap(iv.expand(boxes.positions, 2),
                                I(boxes.obstacles[:, None]),
                                boxes.obstacle_radii[:, None],
                                boxes.obstacle_axes[:, None])
            ok = ok & iv.ge(gap, 0.0).all(axis=2).all(axis=1)
        return ok


    def _pair_distance(self, positions: I) -> I:
        first, second = _pairs(positions.lo.shape[1])
        return iv.norm(positions[:, first] - positions[:, second])

    def _full_distance(self, positions: I) -> I:
        d = iv.norm(iv.expand(positions, 2) - iv.expand(positions, 1))
        eye = np.eye(positions.lo.shape[1], dtype=bool)[None]
        return I(np.where(eye, 0.0, d.lo), np.where(eye, 0.0, d.hi))


    def _link_margin(self, positions: I) -> I:
        first, second = _pairs(positions.lo.shape[1])
        d = self._full_distance(positions)
        s = self.reach.sense_range
        return I(iv._dn(s - _minimax(d.hi)[:, first, second]),
                 iv._up(s - _minimax(d.lo)[:, first, second]))

    def _heading_margin(self, velocities: I) -> I:
        reach = self.reach
        first, second = _pairs(velocities.lo.shape[1])
        speed = iv.norm(velocities)
        headed = iv.ge(speed, reach.min_speed) & iv.gt(speed, 0.0)
        both = headed[:, first] & headed[:, second]
        a, b = velocities[:, first], velocities[:, second]
        angle = iv.atan2_upper(iv.absolute(iv.cross(a, b)), iv.dot(a, b))
        margin = reach.max_angle_deg - angle * (180.0 / math.pi)
        return iv.select(both, margin, np.inf)

    def _range_margin(self, positions: I, boxes: Boxes) -> I:
        return self.reach.target_range - iv.norm(positions
                                                 - I(boxes.target[:, None]))

    def _path_offset(self, points: I, boxes: Boxes) -> I:
        length = np.sqrt(np.sum(boxes.route * boxes.route, axis=-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            along = boxes.route / length[:, None]
        along = np.where(length[:, None] > 1e-12, along, 0.0)
        rel = points - I(boxes.target[:, None])
        offset = iv.absolute(rel[..., 0] * along[:, None, 1]
                             - rel[..., 1] * along[:, None, 0])
        has = (length > 1e-12)[:, None]
        return I(np.where(has, offset.lo, np.inf), np.where(has, offset.hi, np.inf))

    def _hold_margin(self, positions: I, velocities: I, boxes: Boxes) -> I:
        if self.name == "fragmentation":
            return self._link_margin(positions)
        if self.name == "misalignment":
            return self._heading_margin(velocities)
        if self.name == "loss_target":
            return self._range_margin(positions, boxes)
        if self.name == "off_path":
            return self.reach.path_width - self._path_offset(positions, boxes)
        raise NotImplementedError(self.name)


    def _at_goal(self, boxes: Boxes) -> np.ndarray:
        if not self.leader_formation:
            return np.ones(boxes.size, dtype=bool)
        return np.sqrt(np.sum(boxes.route * boxes.route, axis=-1)) <= 1e-12

    def _timeout_violation(self, boxes: Boxes, w: World) -> T:
        reach = self.reach
        goal = self._at_goal(boxes)
        target = I(boxes.target[:, None])
        end = w.positions()
        d1 = iv.norm(end - target)
        progress = self._gain(boxes.positions - target, iv.stack(w.S, axis=1))
        inside = iv.le(d1, reach.goal_within)
        credited = T.of(goal[:, None]) & inside
        progress = iv.select(credited, iv.maximum(progress, reach.need()), progress)
        k = reach.arrivals(boxes.drones)
        count = inside.count(axis=1)
        arrived = T(count.lo >= k, count.hi < k)
        if self.leader_formation:
            centre = I(iv._dn(end.lo.mean(axis=1), 4), iv._up(end.hi.mean(axis=1), 4))
            advanced = iv.le(iv.norm(centre - I(boxes.target)), reach.arrive_within)
            arrived = iv.select_t(T.of(goal), arrived, advanced)
        m = boxes.drones
        mean = iv.total(progress, axis=1) / float(m)
        return ~arrived & iv.lt(mean - reach.need(), 0.0)

    def _timeout_doomed(self, boxes: Boxes) -> T:
        reach = self.reach
        target = I(boxes.target[:, None])
        d0 = iv.norm(boxes.positions - target)
        shape = d0.lo.shape
        nearest = iv.zeros(shape)
        centre_near = iv.zeros(shape[:1])
        best = d0
        for centres, radius, offset in self._discs(boxes, offsets=True):
            distance = iv.norm(centres - target)
            nearest = iv.maximum(nearest, iv.maximum(distance - radius, 0.0))
            best = iv.minimum(best, self._gain(boxes.positions - target, offset)
                              + radius)
            middle_at = I(iv._dn(centres.lo.mean(axis=1), 4),
                          iv._up(centres.hi.mean(axis=1), 4))
            middle = iv.norm(middle_at - I(boxes.target))
            centre_near = iv.maximum(centre_near,
                                     iv.maximum(middle - radius, 0.0))
        need = reach.need()
        goal = self._at_goal(boxes)
        close = iv.le(nearest, reach.goal_within)
        best = iv.select(T.of(goal[:, None]) & close, iv.maximum(best, need), best)
        k = reach.arrivals(boxes.drones)
        count = close.count(axis=1)
        can_arrive = T(count.lo >= k, count.hi < k)
        if self.leader_formation:
            can_arrive = iv.select_t(T.of(goal), can_arrive,
                                     iv.le(centre_near, reach.arrive_within))
        mean = iv.total(best, axis=1) / float(boxes.drones)
        return iv.lt(mean, need) & ~can_arrive


    def violation(self, boxes: Boxes, w: World) -> T:
        name = self.name
        if name == "obstacle_collision":
            if np.any((boxes.obstacle_radii == 0.0)
                      & ~np.any(boxes.obstacle_axes != 0.0, axis=-1)):
                raise NotImplementedError("a body of radius 0 (judged as a face)")
            gap = self.body_gap(iv.expand(w.positions(), 2), iv.expand(w.O, 1),
                                boxes.obstacle_radii[:, None],
                                boxes.obstacle_axes[:, None])
            return iv.lt(gap, 0.0).any(axis=2).any(axis=1)
        if name == "agent_collision":
            d = self._pair_distance(w.positions())
            return iv.lt(d, self.physics.agent_boundary).any(axis=1)
        if name == "timeout":
            return self._timeout_violation(boxes, w)
        held = iv.ge(self._hold_margin(boxes.positions, boxes.velocities, boxes), 0.0)
        after = self._hold_margin(w.positions(), w.velocities(), boxes)
        return (held & iv.lt(after, 0.0)).any(axis=1)

    def doomed(self, boxes: Boxes) -> T:
        name = self.name
        never = T.of(np.zeros(boxes.size, dtype=bool))
        if name == "obstacle_collision":
            first = self.movers(boxes)
            out = never
            for centres, radius in self._discs(boxes):
                gap = self.body_gap(iv.expand(centres, 2), iv.expand(first, 1),
                                    boxes.obstacle_radii[:, None],
                                    boxes.obstacle_axes[:, None])
                out = out | iv.lt(gap + radius, 0.0).any(axis=2).any(axis=1)
            return out
        if name == "agent_collision":
            together = None
            discs = self._discs(boxes)
            first, second = _pairs(boxes.drones)
            for ci, ri in discs:
                for cj, rj in discs:
                    d = iv.norm(ci[:, first] - cj[:, second]) + (ri + rj)
                    together = d if together is None else iv.minimum(together, d)
            return iv.lt(together, self.physics.agent_boundary).any(axis=1)
        if name == "timeout":
            return self._timeout_doomed(boxes)
        held = iv.ge(self._hold_margin(boxes.positions, boxes.velocities, boxes), 0.0)
        if name == "misalignment":
            return never
        if name == "fragmentation":
            apart = None
            discs = self._discs(boxes)
            for ci, ri in discs:
                for cj, rj in discs:
                    d = iv.norm(iv.expand(ci, 2) - iv.expand(cj, 1)) - (ri + rj)
                    apart = d if apart is None else iv.maximum(apart, d)
            eye = np.eye(boxes.drones, dtype=bool)[None]
            lo = np.where(eye, 0.0, apart.lo)
            hi = np.where(eye, 0.0, apart.hi)
            first, second = _pairs(boxes.drones)
            chain = I(_minimax(lo)[:, first, second], _minimax(hi)[:, first, second])
            return (held & iv.gt(chain, self.reach.sense_range)).any(axis=1)
        if name in ("loss_target", "off_path"):
            limit = (self.reach.target_range if name == "loss_target"
                     else self.reach.path_width)
            nearest = iv.zeros(held.shape)
            for centres, radius in self._discs(boxes):
                if name == "loss_target":
                    away = iv.norm(centres - I(boxes.target[:, None]))
                else:
                    away = self._path_offset(centres, boxes)
                nearest = iv.maximum(nearest, away - radius)
            return (held & iv.gt(nearest, limit)).any(axis=1)
        raise NotImplementedError(name)


    def classify(self, boxes: Boxes) -> np.ndarray:
        admissible = self.admissible(boxes)
        doomed = self.doomed(boxes)
        violation = self.violation(boxes, self.tick(boxes))
        out = np.full(boxes.size, UNDECIDED, dtype=np.int8)
        live = admissible.yes & doomed.no
        out[live & violation.no] = int(Verdict.SAFE)
        out[live & violation.yes] = int(Verdict.VIOLATES)
        out[admissible.yes & doomed.yes] = int(Verdict.DOOMED)
        out[admissible.no] = int(Verdict.INFEASIBLE)
        return out


__all__ = ["Boxes", "Engine", "UNDECIDED", "World"]
