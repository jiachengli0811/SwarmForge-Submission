from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import FamilyConfig, ModelSpec
from .dynamics import formation, mover_steps
from .laws import norm
from .state import Sampler, State, admissible, concat
from .window import Reaches, in_disc, in_ring, in_union, union_area


class Reach:
    def __init__(self, config: FamilyConfig, horizon: int):
        physics = config.physics
        self.config = config
        self.horizon = int(horizon)
        self.dt = physics.dt
        self.step = physics.plant.max_step
        self.accel = (physics.plant.max_accel
                      if physics.plant.integration == "velocity" else None)
        self.reaches = Reaches.of(config, self.horizon)
        self.leader: Optional[int] = (
            0 if physics.targets.kind == "leader_formation" else None)

    def discs(self, state: State, t: int) -> List[Tuple[np.ndarray, float]]:
        out = [(state.positions, t * self.step)]
        if self.accel is not None:
            out.append((state.positions + t * self.dt * state.velocities,
                        self.accel * self.dt * self.dt * t * (t + 1) / 2.0))
        return out

    def split(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def contested(self, state: State) -> np.ndarray:
        unreachable, doomed = self.split(state)
        return admissible(state, self.config) & ~unreachable & ~doomed

    def check_record(self, record_reach: float, label: str) -> None:
        if record_reach > self.step * (1.0 + 1e-9):
            raise ValueError(
                "%s can move %.6g per tick, past the config's "
                "physics.plant.max_step %.6g: states the sampling calls "
                "unreachable could be reachable for it"
                % (label, record_reach, self.step))


    def formation_reach(self) -> float:
        targets = self.config.physics.targets
        slots = formation(targets.formation_robots, np.zeros((1, 2)),
                          np.array([[1.0, 0.0]]), targets.formation_spacing)
        span = max([float(norm(slot)[0]) for slot in slots] or [0.0])
        return span + targets.waypoint_advance_within

    def drone_reach(self, drone: int) -> float:
        if self.leader is not None and drone == self.leader:
            return min(self.reaches.target, self.formation_reach())
        return self.reaches.drones

    def body_reach(self, dynamic) -> np.ndarray:
        r = self.reaches
        return np.where(dynamic, r.dynamic_bodies, r.static_bodies)

    def body_reaches(self, state: State) -> np.ndarray:
        reach = np.broadcast_to(self.body_reach(state.dynamic),
                                state.obstacle_radii.shape)
        if self.reaches.walls is None:
            return reach
        return np.where(state.walls, self.reaches.walls, reach)

    def room_share(self, state: State,
                   apart: Optional[float] = None) -> np.ndarray:
        if self.reaches.walls is None or not state.obstacles.shape[1]:
            return np.ones(state.size)
        r = self.reaches
        top = np.where(state.dynamic, r.dynamic_bodies,
                       max(r.static_bodies, r.walls))[None, :]
        reach = self.body_reaches(state)
        if apart is None:
            share = (reach / top) ** 2
        else:
            share = union_area(reach, apart) / union_area(top, apart)
        return np.prod(share, axis=1)

    def require(self, spec: ModelSpec) -> None:
        pass

    def describe(self, spec: ModelSpec) -> Dict[str, Any]:
        if self.leader is None:
            return {}
        return {"leader_reach": self.drone_reach(self.leader)}

    def place(self, batch: State, rng: np.random.Generator,
              spec: ModelSpec) -> np.ndarray:
        raise NotImplementedError

    def sample(self, sampler: Sampler, n: int, seed: int,
               contested: bool = True, regime: float = 0.0
               ) -> Tuple[State, Dict[str, Any]]:
        config, spec = sampler.config, sampler.spec
        sampling = config.sampling
        self.require(spec)
        rng = np.random.default_rng(seed)
        kept = []
        have = candidates = placed = admitted = decided = redrawn = 0
        while have < n:
            if candidates >= sampling.max_draws:
                raise ValueError(
                    "only %d of %d window states after %d candidates; raise "
                    "sampling.max_draws or loosen the window"
                    % (have, n, candidates))
            chunk = sampling.chunk
            batch = sampler.draw(rng, chunk)
            thinned = self.place(batch, rng, spec)
            if regime > 0.0:
                from .regime import redraw
                chosen = redraw(batch, regime, self, config, rng)
            ok = thinned & admissible(batch, config)
            hit = ok & self.contested(batch)
            index = np.flatnonzero(hit if contested else ok)
            needed = n - have
            used = chunk
            if index.size >= needed:
                index = index[:needed]
                used = int(index[-1]) + 1
            candidates += used
            placed += int(np.sum(thinned[:used]))
            admitted += int(np.sum(ok[:used]))
            decided += int(np.sum(hit[:used]))
            if index.size:
                kept.append(batch.take(index))
                have += index.size
                if regime > 0.0:
                    redrawn += int(np.sum(chosen[index]))

        share = decided / admitted if admitted else float("nan")
        info = {"focus": "contested" if contested else "uniform",
                "candidates": candidates, "kept": have,
                "thinned_share": placed / candidates,
                "admissible_share": admitted / placed if placed else 0.0,
                "contested_share": share,
                "contested_share_se": float(np.sqrt(
                    share * (1.0 - share) / max(admitted, 1)))}
        if regime > 0.0:
            info.update(regime=regime, regime_kept=redrawn / max(have, 1))
        return concat(kept), info


def _square_distance(points: np.ndarray, r: np.ndarray) -> np.ndarray:
    outside = np.maximum(np.abs(points) - r[..., None], 0.0)
    return np.sqrt(np.sum(outside * outside, axis=-1))


def _swept_square_distance(rel: np.ndarray, d: np.ndarray,
                           r: np.ndarray) -> np.ndarray:
    a = np.broadcast_to(rel, np.broadcast_shapes(rel.shape, d.shape))
    seg = np.broadcast_to(-d, a.shape)
    best = np.minimum(_square_distance(a, r), _square_distance(a + seg, r))
    length = np.sum(seg * seg, axis=-1)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            corner = np.stack([sx * r, sy * r], axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(length > 0.0,
                             np.sum((corner - a) * seg, axis=-1) / length, 0.0)
            t = np.clip(t, 0.0, 1.0)
            best = np.minimum(best, norm(a + t[..., None] * seg - corner))
    lo = np.zeros(a.shape[:-1])
    hi = np.ones(a.shape[:-1])
    for axis in (0, 1):
        p, v = a[..., axis], seg[..., axis]
        within = np.abs(p) <= r
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-r - p) / v, (r - p) / v
        moving = v != 0.0
        lo = np.maximum(lo, np.where(moving, np.minimum(t1, t2),
                                     np.where(within, -np.inf, np.inf)))
        hi = np.minimum(hi, np.where(moving, np.maximum(t1, t2),
                                     np.where(within, np.inf, -np.inf)))
    return np.where(lo <= hi, 0.0, best)


class ObstacleReach(Reach):
    def boundary(self, radii: np.ndarray) -> np.ndarray:
        return self.config.physics.obstacle_boundary(radii)

    def ring_width(self, obstacle_velocities: np.ndarray) -> np.ndarray:
        return self.horizon * (self.step + self.dt * norm(obstacle_velocities))

    def ring_area(self, radii: np.ndarray, obstacle_velocities: np.ndarray,
                  axes: Optional[np.ndarray] = None) -> np.ndarray:
        physics = self.config.physics
        w = self.ring_width(obstacle_velocities)
        if physics.square_bodies:
            a = physics.agent_radius + physics.obstacle_min_gap
            along = radii if axes is None else np.where(
                np.any(axes != 0.0, axis=-1), norm(axes), radii)
            return (4.0 * (along + radii) * w
                    + np.pi * ((a + w) ** 2 - a ** 2))
        b = self.boundary(radii)
        return np.pi * ((b + w) ** 2 - b ** 2)

    def _gaps(self, points: np.ndarray, centres: np.ndarray,
              radii: np.ndarray, axes: Optional[np.ndarray] = None
              ) -> np.ndarray:
        return self.config.physics.body_gap(
            points[:, :, None, :], centres[:, None, :, :], radii[:, None, :],
            None if axes is None else axes[:, None, :, :])

    def in_ring(self, state: State) -> np.ndarray:
        w = self.ring_width(state.obstacle_velocities)[:, None, :]
        gap = self._gaps(state.positions, state.obstacles,
                         state.obstacle_radii, state.obstacle_axes)
        return (gap >= 0.0) & (gap < w)

    def _segment_gaps(self, points: np.ndarray, starts: np.ndarray,
                      ends: np.ndarray, radii: np.ndarray) -> np.ndarray:
        physics = self.config.physics
        d = (ends - starts)[:, None, :, :]
        rel = points[:, :, None, :] - starts[:, None, :, :]
        if physics.square_bodies:
            distance = _swept_square_distance(
                rel, d, np.broadcast_to(radii[:, None, :], rel.shape[:-1]))
            return distance - physics.agent_radius - physics.obstacle_min_gap
        length = np.sum(d * d, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(length > 0.0, np.sum(rel * d, axis=-1) / length, 0.0)
        u = np.clip(u, 0.0, 1.0)
        distance = norm(rel - u[..., None] * d)
        return (distance - radii[:, None, :] - physics.agent_radius
                - physics.obstacle_min_gap)

    def split(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        shape = state.positions.shape[:2] + state.obstacles.shape[1:2]
        safe = np.ones(shape, dtype=bool)
        doomed = np.zeros(state.size, dtype=bool)
        radii = state.obstacle_radii
        ranges = self.config.state.dynamic
        held = (np.asarray(state.dynamic, dtype=bool)
                & bool(ranges is not None and ranges.standoff))
        first = mover_steps(self.config, state.obstacles,
                            state.obstacle_velocities, radii, state.dynamic,
                            state.positions)
        for t in range(1, self.horizon + 1):
            far = first + (t - 1) * self.dt * state.obstacle_velocities
            near = np.where(held[None, :, None], first, far)
            missed = np.zeros(shape, dtype=bool)
            inside = np.zeros(shape, dtype=bool)
            for centres, radius in self.discs(state, t):
                gap = self._gaps(centres, far, radii, state.obstacle_axes)
                if t > 1 and held.any():
                    least = np.where(held[None, None, :],
                                     self._segment_gaps(centres, near, far,
                                                        radii), gap)
                    start = self._gaps(centres, near, radii,
                                       state.obstacle_axes)
                    missed |= least - radius >= 0.0
                    inside |= (gap + radius < 0.0) & (start + radius < 0.0)
                else:
                    missed |= gap - radius >= 0.0
                    inside |= gap + radius < 0.0
            safe &= missed
            doomed |= np.any(inside, axis=(1, 2))
        return np.all(safe, axis=(1, 2)), doomed

    def _ring_offsets(self, rng: np.random.Generator, radii: np.ndarray,
                      widths: np.ndarray, axes: Optional[np.ndarray] = None
                      ) -> np.ndarray:
        physics = self.config.physics
        if not physics.square_bodies:
            b = self.boundary(radii)
            return in_ring(rng, np.zeros((radii.size, 2)), b, b + widths)
        a = physics.agent_radius + physics.obstacle_min_gap
        corner = np.sqrt(2.0) * radii
        if axes is not None:
            wall = np.any(axes != 0.0, axis=-1)
            corner = np.where(wall, np.hypot(norm(axes), radii), corner)
        outer = corner + a + widths
        out = np.zeros((radii.size, 2))
        todo = np.arange(radii.size)
        while todo.size:
            u = in_ring(rng, np.zeros((todo.size, 2)), 0.0, outer[todo])
            gap = physics.body_gap(np.zeros((todo.size, 2)), u, radii[todo],
                                   None if axes is None else axes[todo])
            keep = (gap >= 0.0) & (gap < widths[todo])
            out[todo[keep]] = u[keep]
            todo = todo[~keep]
        return out

    def require(self, spec: ModelSpec) -> None:
        if spec.static_obstacles + spec.dynamic_obstacles == 0:
            raise ValueError("an obstacle contact needs a body to anchor its "
                             "window, and this model draws none")


    @property
    def balanced(self) -> bool:
        return self.leader is not None

    def describe(self, spec: ModelSpec) -> Dict[str, Any]:
        out = super().describe(spec)
        if self.balanced:
            out["anchors"] = "each drone alike"
        if self.horizon > 1 and self.config.physics.square_bodies:
            out["swept_square"] = "exact"
        return out

    def _drone_log_weights(self, m: int) -> np.ndarray:
        return np.array([sum(2.0 * np.log(self.drone_reach(e))
                             for e in range(m) if e != i) for i in range(m)])

    def _body_weights(self, state: State) -> np.ndarray:
        room = np.pi * self.body_reaches(state) ** 2
        return self.ring_area(state.obstacle_radii, state.obstacle_velocities,
                              state.obstacle_axes) / room

    def _ceiling(self, spec: ModelSpec) -> float:
        st = self.config.state
        total = 0.0
        for count, ranges, dynamic in ((spec.static_obstacles, st.static,
                                        False),
                                       (spec.dynamic_obstacles, st.dynamic,
                                        True)):
            if not count:
                continue
            speed = ranges.speed.hi if ranges.speed is not None else 0.0
            ring = float(self.ring_area(np.array(ranges.radius.hi),
                                        np.array([speed, 0.0])))
            best = ring / (np.pi * float(self.body_reach(dynamic)) ** 2)
            if ranges.wall_share and self.reaches.walls is not None:
                hi = ranges.wall_extent.hi
                wall = float(self.ring_area(
                    np.array(hi * (np.sqrt(2.0) - 1.0)),
                    np.array([0.0, 0.0]), np.array([hi, 0.0])))
                best = max(best, wall / (np.pi * self.reaches.walls ** 2))
            total += count * best
        return total

    def anchors(self, state: State) -> np.ndarray:
        p = state.positions
        m, k = state.drones, state.obstacles.shape[1]
        apart = norm(p[:, :, None] - p[:, None])
        reach = np.array([[np.inf if e == i else self.drone_reach(e)
                           for e in range(m)] for i in range(m)])
        drones = np.all(apart <= reach[None], axis=2)
        within = (norm(p[:, :, None] - state.obstacles[:, None])
                  <= self.body_reaches(state)[:, None, :])
        outside = np.sum(~within, axis=2)
        bodies = (outside[:, :, None] - (~within)) == 0
        ok = self.in_ring(state) & drones[:, :, None] & bodies
        if self.leader is None:
            goal = norm(state.target[:, None] - p) <= self.reaches.target
            ok &= goal[:, :, None]
        return ok

    def place(self, batch: State, rng: np.random.Generator,
              spec: ModelSpec) -> np.ndarray:
        size = batch.size
        rows = np.arange(size)
        m, k = spec.drones, spec.static_obstacles + spec.dynamic_obstacles
        if self.balanced:
            drone = rng.integers(m, size=size)
        else:
            log_w = self._drone_log_weights(m)
            weights = np.exp(log_w - log_w.max())
            drone = rng.choice(m, size=size, p=weights / weights.sum())
        g = self._body_weights(batch)
        z = g.sum(axis=1)
        cumulative = np.cumsum(g, axis=1) / z[:, None]
        body = np.minimum(np.sum(rng.random(size)[:, None] > cumulative,
                                 axis=1), k - 1)

        offset = batch.target - batch.positions[:, 0]
        centre = batch.positions[rows, drone].copy()
        w = self.ring_width(batch.obstacle_velocities[rows, body])
        batch.obstacles[rows, body] = centre + self._ring_offsets(
            rng, batch.obstacle_radii[rows, body], w,
            batch.obstacle_axes[rows, body])
        reach = self.body_reaches(batch)
        for e in range(k):
            other = np.flatnonzero(body != e)
            batch.obstacles[other, e] = in_disc(rng, centre[other],
                                                reach[other, e])
        for e in range(m):
            other = np.flatnonzero(drone != e)
            batch.positions[other, e] = in_disc(rng, centre[other],
                                                self.drone_reach(e))
        if self.leader is None:
            batch.target = in_disc(rng, centre, self.reaches.target)
        else:
            batch.target = batch.positions[:, self.leader] + offset

        anchors = self.anchors(batch)
        count = (anchors[rows, drone].sum(axis=1) if self.balanced
                 else anchors.sum(axis=(1, 2)))
        with np.errstate(divide="ignore", invalid="ignore"):
            keep = np.where(count > 0, z * self.room_share(batch)
                            / (count * self._ceiling(spec)), 0.0)
        return rng.random(size) < keep


class AgentReach(Reach):
    @property
    def boundary(self) -> float:
        return self.config.physics.agent_boundary

    @property
    def ring_width(self) -> float:
        return 2.0 * self.horizon * self.step

    @property
    def ring(self) -> Tuple[float, float]:
        return self.boundary, self.boundary + self.ring_width

    @staticmethod
    def pairs(drones: int) -> Tuple[np.ndarray, np.ndarray]:
        return np.triu_indices(drones, 1)

    def in_ring(self, state: State) -> np.ndarray:
        first, second = self.pairs(state.drones)
        d = norm(state.positions[:, first] - state.positions[:, second])
        return (d >= self.boundary) & (d < self.boundary + self.ring_width)

    def pair_bounds(self, state: State, t: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
        first, second = self.pairs(state.drones)
        apart = np.full((state.size, first.size), -np.inf)
        together = np.full((state.size, first.size), np.inf)
        discs = self.discs(state, t)
        for centres_i, radius_i in discs:
            for centres_j, radius_j in discs:
                distance = norm(centres_i[:, first] - centres_j[:, second])
                apart = np.maximum(apart, distance - radius_i - radius_j)
                together = np.minimum(together,
                                      distance + radius_i + radius_j)
        return apart, together

    def split(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        first, _ = self.pairs(state.drones)
        b = self.boundary
        safe = np.ones((state.size, first.size), dtype=bool)
        doomed = np.zeros(state.size, dtype=bool)
        for t in range(1, self.horizon + 1):
            apart, together = self.pair_bounds(state, t)
            safe &= apart >= b
            doomed |= np.any(together < b, axis=1)
        return np.all(safe, axis=1), doomed

    def require(self, spec: ModelSpec) -> None:
        if spec.drones < 2:
            raise ValueError("an agent contact needs two drones, and this "
                             "model draws %d" % spec.drones)


    def _around(self, m: int, dynamic: np.ndarray
                ) -> List[Tuple[str, int, float]]:
        out = [("drone", e, self.drone_reach(e)) for e in range(m)]
        out += [("body", e, float(self.body_reach(moving)))
                for e, moving in enumerate(dynamic)]
        if self.leader is None:
            out.append(("target", 0, self.reaches.target))
        return out

    @staticmethod
    def _log_room(around, first, second, distance) -> np.ndarray:
        total = 0.0
        for kind, e, radius in around:
            member = ((first == e) | (second == e)) if kind == "drone" \
                else np.zeros(first.size, dtype=bool)
            total = total + np.where(member, 0.0,
                                     np.log(union_area(radius, distance)))
        return total

    def anchor_weights(self, state: State) -> np.ndarray:
        m = state.drones
        first, second = self.pairs(m)
        p = state.positions
        pi, pj = p[:, first], p[:, second]
        distance = norm(pi - pj)
        ok = self.in_ring(state)
        bodies = self.body_reaches(state)
        around = [(kind, e, bodies[:, e:e + 1] if kind == "body" else radius)
                  for kind, e, radius in self._around(m, state.dynamic)]
        for kind, e, radius in around:
            if kind == "drone":
                where = p[:, e]
            elif kind == "body":
                where = state.obstacles[:, e]
            else:
                where = state.target
            near = np.minimum(norm(where[:, None] - pi),
                              norm(where[:, None] - pj))
            if kind == "drone":
                member = (first == e) | (second == e)
                ok &= member[None] | (near <= radius)
            else:
                ok &= near <= radius
        widest = self._log_room(around, first, second, self.ring[1])
        now = self._log_room(around, first, second, distance)
        return np.where(ok, np.exp(widest - now), 0.0)

    def place(self, batch: State, rng: np.random.Generator,
              spec: ModelSpec) -> np.ndarray:
        size = batch.size
        rows = np.arange(size)
        m = spec.drones
        first, second = self.pairs(m)
        around = self._around(m, batch.dynamic)
        inner, outer = self.ring
        log_w = self._log_room(around, first, second, outer)
        weights = np.exp(log_w - log_w.max())
        pair = rng.choice(first.size, size=size, p=weights / weights.sum())
        flip = rng.random(size) < 0.5
        anchor = np.where(flip, second[pair], first[pair])
        mover = np.where(flip, first[pair], second[pair])

        offset = batch.target - batch.positions[:, 0]
        pa = batch.positions[rows, anchor].copy()
        pm = in_ring(rng, pa, inner, outer)
        batch.positions[rows, mover] = pm
        for e in range(m):
            other = np.flatnonzero((anchor != e) & (mover != e))
            batch.positions[other, e] = in_union(
                rng, pa[other], pm[other], self.drone_reach(e))
        reach = self.body_reaches(batch)
        for e in range(batch.obstacles.shape[1]):
            batch.obstacles[:, e] = in_union(rng, pa, pm, reach[:, e])
        if self.leader is None:
            batch.target = in_union(rng, pa, pm, self.reaches.target)
        else:
            batch.target = batch.positions[:, self.leader] + offset

        total = self.anchor_weights(batch).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            keep = np.where(total > 0, self.room_share(batch, outer) / total,
                            0.0)
        return rng.random(size) < keep


class SwarmReach(Reach):
    def __init__(self, config: FamilyConfig, horizon: int):
        super().__init__(config, horizon)
        self.goal_reach = self.reaches.target

    def drone_reach(self, drone: int) -> float:
        if self.leader is None:
            return super().drone_reach(drone)
        targets = self.config.physics.targets
        slots = formation(targets.formation_robots, np.zeros((1, 2)),
                          np.array([[1.0, 0.0]]), targets.formation_spacing)
        span = max([float(norm(slot)[0]) for slot in slots] or [0.0])
        if drone == self.leader:
            return span + targets.waypoint_advance_within
        return max(self.reaches.drones, span)

    def require(self, spec: ModelSpec) -> None:
        if spec.drones < 1:
            raise ValueError("a swarm needs a drone")

    def describe(self, spec: ModelSpec) -> Dict[str, Any]:
        return {"drone_reaches": [self.drone_reach(e)
                                  for e in range(spec.drones)],
                "goal_reach": self.goal_reach}

    def _drone_log_weights(self, m: int) -> np.ndarray:
        return np.array([sum(2.0 * np.log(self.drone_reach(e))
                             for e in range(m) if e != i) for i in range(m)])

    def anchors(self, state: State) -> np.ndarray:
        p = state.positions
        m = state.drones
        apart = norm(p[:, :, None] - p[:, None])
        reach = np.array([[np.inf if e == i else self.drone_reach(e)
                           for e in range(m)] for i in range(m)])
        ok = np.all(apart <= reach[None], axis=2)
        if state.obstacles.shape[1]:
            ok &= np.all(norm(p[:, :, None] - state.obstacles[:, None])
                         <= self.body_reaches(state)[:, None, :], axis=2)
        if self.leader is None:
            ok &= norm(state.target[:, None] - p) <= self.goal_reach
        return ok

    def place(self, batch: State, rng: np.random.Generator,
              spec: ModelSpec) -> np.ndarray:
        size = batch.size
        rows = np.arange(size)
        m = spec.drones
        log_w = self._drone_log_weights(m)
        weights = np.exp(log_w - log_w.max())
        drone = rng.choice(m, size=size, p=weights / weights.sum())
        offset = batch.target - batch.positions[:, 0]
        centre = batch.positions[rows, drone].copy()
        reach = self.body_reaches(batch)
        for e in range(batch.obstacles.shape[1]):
            batch.obstacles[:, e] = in_disc(rng, centre, reach[:, e])
        for e in range(m):
            other = np.flatnonzero(drone != e)
            batch.positions[other, e] = in_disc(rng, centre[other],
                                                self.drone_reach(e))
        if self.leader is None:
            batch.target = in_disc(rng, centre, self.goal_reach)
        else:
            batch.target = batch.positions[:, self.leader] + offset
        count = self.anchors(batch).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            keep = np.where(count > 0, self.room_share(batch) / count, 0.0)
        return rng.random(size) < keep


class TimeoutReach(SwarmReach):
    def __init__(self, config: FamilyConfig, horizon: int,
                 pace: Optional[float] = None):
        super().__init__(config, horizon)
        if pace is None or pace <= 0.0:
            raise ValueError("the timeout question needs models.timeout.pace, "
                             "the progress per tick a mission needs")
        if config.mission is None:
            raise ValueError("the timeout question reads the mission's goal "
                             "disc and arrival share; %s declares no mission "
                             "block" % config.path)
        if pace >= self.step:
            raise ValueError("models.timeout.pace %.6g is no less than "
                             "max_step %.6g: no drone could keep it"
                             % (pace, self.step))
        self.pace = float(pace)
        mission = config.mission
        self.goal_within = float(mission.goal_radius)
        if self.leader is None:
            self.arrive_within = self.goal_within
            self.goal_reach = self.reaches.target + self.arrive_within
        else:
            self.arrive_within = float(
                config.physics.targets.waypoint_advance_within)
            self.goal_reach = self.reaches.target
        self.arrive_fraction = float(mission.arrive_fraction)

    def need(self) -> float:
        return self.pace * self.horizon

    def arrivals(self, drones: int) -> int:
        return max(1, int(np.ceil(self.arrive_fraction * drones - 1e-9)))


    def at_goal(self, route: Optional[np.ndarray], size: int) -> np.ndarray:
        if self.leader is None or route is None:
            return np.full(size, self.leader is None)
        return norm(route) <= 1e-12

    def progress(self, start: np.ndarray, end: np.ndarray,
                 target: np.ndarray, route: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        d0 = norm(start - target[:, None])
        d1 = norm(end - target[:, None])
        progress = d0 - d1
        goal = self.at_goal(route, start.shape[0])
        inside = d1 <= self.goal_within
        progress = np.where(goal[:, None] & inside,
                            np.maximum(progress, self.need()), progress)
        arrived = inside.sum(axis=1) >= self.arrivals(start.shape[1])
        if self.leader is not None:
            advanced = (norm(end.mean(axis=1) - target)
                        <= self.arrive_within)
            arrived = np.where(goal, arrived, advanced)
        return progress.mean(axis=1) - self.need(), arrived


    def split(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        target = state.target[:, None]
        d0 = norm(state.positions - target)
        nearest = np.zeros(d0.shape)
        farthest = np.full(d0.shape, np.inf)
        centre_near = np.zeros(state.size)
        centre_far = np.full(state.size, np.inf)
        for centres, radius in self.discs(state, self.horizon):
            distance = norm(centres - target)
            nearest = np.maximum(nearest, np.maximum(distance - radius, 0.0))
            farthest = np.minimum(farthest, distance + radius)
            middle = norm(centres.mean(axis=1) - state.target)
            centre_near = np.maximum(centre_near,
                                     np.maximum(middle - radius, 0.0))
            centre_far = np.minimum(centre_far, middle + radius)
        need = self.need()
        best, worst = d0 - nearest, d0 - farthest
        goal = self.at_goal(state.route, state.size)
        within = self.goal_within
        best = np.where(goal[:, None] & (nearest <= within),
                        np.maximum(best, need), best)
        worst = np.where(goal[:, None] & (farthest <= within),
                         np.maximum(worst, need), worst)
        k = self.arrivals(state.drones)
        can_arrive = np.sum(nearest <= within, axis=1) >= k
        must_arrive = np.sum(farthest <= within, axis=1) >= k
        if self.leader is not None:
            can_arrive = np.where(goal, can_arrive,
                                  centre_near <= self.arrive_within)
            must_arrive = np.where(goal, must_arrive,
                                   centre_far <= self.arrive_within)
        doomed = (best.mean(axis=1) < need) & ~can_arrive
        unreachable = (worst.mean(axis=1) >= need) | must_arrive
        return unreachable, doomed


    def describe(self, spec: ModelSpec) -> Dict[str, Any]:
        out = super().describe(spec)
        out["arrive_within"] = self.arrive_within
        if self.leader is not None:
            out["goal_within"] = self.goal_within
        return out


__all__ = ["AgentReach", "ObstacleReach", "Reach", "SwarmReach",
           "TimeoutReach"]
