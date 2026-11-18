from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from ..solve.axes import targets_of
from ..solve.config import body_frame
from ..solve.dynamics import formation, mover_steps
from ..solve.state import State

FEATURES_VERSION = 2

SPACES: Dict[str, Tuple[str, ...]] = {
    "obstacle_collision": (
        "m_goal_away1", "m_away_body", "peer_dist", "coast_gap", "goal_dist",
        "flock_bearing", "body_push_size", "goal_bearing", "flock_now_size",
        "peer_bearing_now", "m_goal_away1_flock_near", "m_goal_flock_near",
        "flock_near_size", "brake_excess", "stop_speed"),
    "agent_collision": (
        "probe_d_ws0.3_wb3", "d_coast_gap", "gap1", "probe_d_ws0.3_wb0.3",
        "goal_dist", "probe_d_ws3_wb0.3", "peer_bearing_now", "d_gap2",
        "peer_dist", "hz_goal_at_me", "hz_body_gap", "probe_b_ws0.03_wb0.3",
        "probe_d_ws3_wb3", "flock_size", "m_body_push"),
    "timeout": (
        "s_centroid_dist", "s_v_goal", "t_body_push_along", "peer_dist_now",
        "coast_gap1", "goal_dist", "t_free_best_over_need", "goal_bearing",
        "t_flock_along", "t_coast_over_need", "order", "t_mates_v_along",
        "t_v_cross", "brake_excess", "centroid_dist"),
}

DIRECTIONS = 180
_ANGLES = np.arange(DIRECTIONS) * 2.0 * np.pi / DIRECTIONS
_U = np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1)
_SPAN = np.abs(np.arange(DIRECTIONS)[:, None] - np.arange(DIRECTIONS)[None])
_APART = np.minimum(_SPAN, DIRECTIONS - _SPAN) * (360.0 / DIRECTIONS)

CHUNK = 3000


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1), 1e-12)[..., None]


def _angle(v: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def margin(mask: np.ndarray, bearing: np.ndarray) -> np.ndarray:
    j = np.round(np.mod(bearing, 360.0) / (360.0 / DIRECTIONS)).astype(int) \
        % DIRECTIONS
    apart = _APART[j]
    inside = np.take_along_axis(mask, j[..., None], axis=2)[..., 0]
    to_in = np.where(mask, apart, np.inf).min(axis=2)
    to_out = np.where(~mask, apart, np.inf).min(axis=2)
    out = np.where(inside, -to_out, to_in)
    return np.where(np.isfinite(out), out, 180.0)


def nearest_surface(physics, points: np.ndarray, centres: np.ndarray,
                    radii: np.ndarray, axes: np.ndarray) -> np.ndarray:
    out = []
    for k in range(centres.shape[1]):
        offset = points - centres[:, None, k]
        if physics.square_bodies:
            local, half, frame = body_frame(offset, radii[:, None, k],
                                            axes[:, None, k])
            clipped = np.clip(local, -half, half)
            point = clipped if frame is None else (
                clipped[..., :1] * frame[0] + clipped[..., 1:] * frame[1])
            out.append(centres[:, None, k] + point)
        else:
            out.append(centres[:, None, k]
                       + _unit(offset) * radii[:, None, k, None])
    return np.stack(out, axis=2) if out else np.zeros(points.shape[:2]
                                                       + (0, 2))


class _Window:
    def __init__(self, state: State, model):
        config = model.config
        physics = config.physics
        plant = physics.plant
        self.model = model
        self.state, self.config, self.physics = state, config, physics
        self.dt, self.step, self.horizon = physics.dt, plant.max_step, \
            model.spec.horizon
        self.vmax = self.step / self.dt
        self.velocity = plant.integration == "velocity"
        self.limited = self.velocity and plant.max_accel is not None
        self.amax = plant.max_accel
        h = self.horizon
        self.rho = (self.amax * self.dt ** 2 * h * (h + 1) / 2.0
                    if self.limited else h * self.step)
        self.window = config.state.window
        reaches = model.reach.reaches
        self.goal_cap = float(getattr(model.reach, "goal_reach",
                                      reaches.target))
        self.drone_cap = float(reaches.drones)
        self.sequential = plant.update_order != "synchronous"
        self.formation = physics.targets.kind == "leader_formation"
        b, m = state.size, state.drones
        self.b, self.m = b, m
        self.rows = np.arange(b)[:, None]
        self.p = state.positions
        self.v_state = state.velocities
        self.v = state.velocities if self.velocity else np.zeros_like(self.p)
        self._cache: Dict[str, np.ndarray] = {}

    def cached(self, key: str, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]


    @property
    def bodies(self) -> np.ndarray:
        def make():
            s = self.state
            obstacles = s.obstacles
            for _ in range(self.horizon):
                obstacles = mover_steps(self.config, obstacles,
                                        s.obstacle_velocities,
                                        s.obstacle_radii, s.dynamic, self.p)
            return obstacles
        return self.cached("bodies", make)

    def body_gaps(self, points: np.ndarray) -> np.ndarray:
        s = self.state
        lead = (self.b,) + (1,) * (points.ndim - 2)
        return self.physics.body_gap(
            points[..., None, :],
            self.bodies.reshape(lead + self.bodies.shape[1:]),
            s.obstacle_radii.reshape(lead + s.obstacle_radii.shape[1:]),
            s.obstacle_axes.reshape(lead + s.obstacle_axes.shape[1:]))

    @property
    def body_set(self) -> np.ndarray:
        def make():
            k = self.state.obstacles.shape[1]
            if k == 0:
                return np.zeros((self.b, self.m, DIRECTIONS), dtype=bool)
            if self.limited:
                centre = self.p + self.horizon * self.dt * self.v
                ends = centre[:, :, None, :] + self.rho * _U[None, None]
                return (self.body_gaps(ends) < 0).any(axis=3)
            mask = np.zeros((self.b, self.m, DIRECTIONS), dtype=bool)
            for t in range(1, self.horizon + 1):
                ends = self.p[:, :, None, :] + t * self.step * _U[None, None]
                mask |= (self.body_gaps(ends) < 0).any(axis=3)
            return mask
        return self.cached("body_set", make)

    @property
    def gaps_now(self) -> np.ndarray:
        return self.cached("gaps_now", lambda: self.body_gaps(self.p))

    @property
    def nearest_body(self) -> np.ndarray:
        return self.cached("nearest_body",
                           lambda: np.argmin(self.gaps_now, axis=2))

    @property
    def gap1(self) -> np.ndarray:
        return np.take_along_axis(self.gaps_now, self.nearest_body[..., None],
                                  2)[..., 0]

    @property
    def surface(self) -> np.ndarray:
        s = self.state
        return self.cached("surface", lambda: nearest_surface(
            self.physics, self.p, self.bodies, s.obstacle_radii,
            s.obstacle_axes))

    @property
    def toward_body(self) -> np.ndarray:
        def make():
            point = self.surface[self.rows, np.arange(self.m)[None],
                                 self.nearest_body]
            return _unit(point - self.p)
        return self.cached("toward_body", make)

    @property
    def body_push(self) -> np.ndarray:
        return self.cached("body_push", lambda: self._push(self.surface))

    @property
    def body_push_now(self) -> np.ndarray:
        def make():
            s = self.state
            return self._push(nearest_surface(
                self.physics, self.p, s.obstacles, s.obstacle_radii,
                s.obstacle_axes))
        return self.cached("body_push_now", make)

    def _push(self, surface: np.ndarray) -> np.ndarray:
        s, w = self.state, self.window
        away = self.p[:, :, None] - surface
        distance = np.linalg.norm(away, axis=-1)
        reach = np.where(s.dynamic, w.dynamic_bodies or 0.0,
                         w.static_bodies or 0.0)[None, None]
        if w.walls is not None:
            reach = np.where(s.walls[:, None], w.walls, reach)
        weight = np.clip(1.0 - distance / np.maximum(reach, 1e-12), 0.0,
                         None)
        return (_unit(away) * weight[..., None]).sum(axis=2)

    @property
    def paths(self) -> np.ndarray:
        def make():
            p = self.p
            ticks = range(1, self.horizon + 1)
            if self.velocity:
                return np.stack([p + t * self.dt * self.v_state for t in ticks])
            to = targets_of(self.state, self.config) - p
            far = np.linalg.norm(to, axis=-1)
            return np.stack([p + np.minimum(t * self.step, far)[..., None]
                             * _unit(to) for t in ticks])
        return self.cached("paths", make)

    @property
    def drone_set(self) -> np.ndarray:
        def make():
            boundary = self.physics.agent_boundary
            other = ~np.eye(self.m, dtype=bool)[None, :, None, :]
            q = self.paths
            if self.limited:
                centre = self.p + self.horizon * self.dt * self.v
                ends = centre[:, :, None, :] + self.rho * _U[None, None]
                d = np.linalg.norm(ends[:, :, :, None, :]
                                   - q[-1][:, None, None, :, :], axis=-1)
                return ((d < boundary) & other).any(axis=3)
            mask = np.zeros((self.b, self.m, DIRECTIONS), dtype=bool)
            for t in range(1, self.horizon + 1):
                ends = self.p[:, :, None, :] + t * self.step * _U[None, None]
                d = np.linalg.norm(ends[:, :, :, None, :]
                                   - q[t - 1][:, None, None, :, :], axis=-1)
                mask |= ((d < boundary) & other).any(axis=3)
            return mask
        return self.cached("drone_set", make)


    def frame_of(self, mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        cos = (mask * _U[None, None, :, 0]).sum(axis=2)
        sin = (mask * _U[None, None, :, 1]).sum(axis=2)
        return np.where(mask.any(axis=2), np.degrees(np.arctan2(sin, cos)),
                        _angle(fallback))

    def margin_of(self, mask: np.ndarray, command: np.ndarray) -> np.ndarray:
        move = command - self.v if self.limited else command
        moving = np.linalg.norm(move, axis=-1) > 1e-9
        return np.where(moving, margin(mask, _angle(move)), 180.0)


    @property
    def nominal(self) -> np.ndarray:
        def make():
            if self.velocity:
                return self.p + self.dt * self.v
            to = targets_of(self.state, self.config) - self.p
            return self.p + self.step * _unit(to)
        return self.cached("nominal", make)

    @property
    def targets(self) -> np.ndarray:
        def make():
            targets = targets_of(self.state, self.config)
            if not self.formation or self.m < 2:
                return targets
            t = self.physics.targets
            slots = formation(t.formation_robots, self.state.target,
                              self.state.target - self.nominal[:, 0],
                              t.formation_spacing)
            out = targets.copy()
            for i in range(1, self.m):
                out[:, i] = slots[i - 1]
            return out
        return self.cached("targets", make)

    @property
    def seen(self) -> np.ndarray:
        def make():
            p = self.p
            if self.sequential:
                earlier = np.arange(self.m)[None, :] < np.arange(self.m)[:, None]
                return np.where(earlier[None, :, :, None],
                                self.nominal[:, None], p[:, None])
            return np.broadcast_to(p[:, None], (self.b, self.m, self.m, 2))
        return self.cached("seen", make)

    def flock_push(self, seen: np.ndarray, reach: float) -> np.ndarray:
        rel = seen - self.p[:, :, None]
        distance = np.linalg.norm(rel, axis=-1) + np.eye(self.m)[None] * 1e12
        weight = np.clip(1.0 - distance / reach, 0.0, None)
        return -(_unit(rel) * weight[..., None]).sum(axis=2)


def _obstacle(w: _Window, names: Sequence[str]) -> Dict[str, np.ndarray]:
    mask = w.body_set
    toward = w.toward_body
    alpha = w.frame_of(mask, toward)
    goal = w.targets - w.p
    side = np.where(_wrap(_angle(goal) - alpha) < 0, -1.0, 1.0)
    vmax, rho = w.vmax, w.rho
    ug = _unit(goal)
    reach_d = w.window.drones

    def bearing(vec):
        return side * _wrap(_angle(vec) - alpha)

    def near_push():
        return w.flock_push(w.seen, reach_d / 4.0)

    def approach():
        return np.sum(w.v_state * toward, axis=-1) / vmax

    def stop_speed():
        surface = np.maximum(w.gap1 + w.physics.agent_radius
                             + w.physics.obstacle_min_gap, 0.0)
        a = w.amax if w.limited else vmax / w.dt
        return np.minimum(np.sqrt(2.0 * a * surface) / vmax, 10.0)

    def nearest_now():
        rel = w.p[:, None, :, :] - w.p[:, :, None, :]
        distance = np.linalg.norm(rel, axis=-1) + np.eye(w.m)[None] * 1e12
        j = distance.argmin(axis=2)
        return np.take_along_axis(
            rel, j[..., None, None].repeat(2, -1), 2)[:, :, 0]

    def peer_dist():
        rel = w.seen - w.p[:, :, None]
        distance = np.linalg.norm(rel, axis=-1) + np.eye(w.m)[None] * 1e12
        return np.minimum(np.log1p(distance.min(axis=2) / rho),
                          np.log1p(w.drone_cap / rho))

    def coast_gap():
        least = np.min(np.stack([
            w.body_gaps(w.p + t * w.dt * w.v).min(axis=2)
            for t in range(1, w.horizon + 1)]), axis=0)
        return np.clip(least / rho, -3.0, 3.0)

    def flock_bearing():
        push = w.flock_push(w.seen, reach_d)
        return np.where(np.linalg.norm(push, axis=-1) > 0, bearing(push), 0.0)

    def flock_now_size():
        rel = w.p[:, None, :, :] - w.p[:, :, None, :]
        distance = np.linalg.norm(rel, axis=-1) + np.eye(w.m)[None] * 1e12
        weight = np.clip(1.0 - distance / reach_d, 0.0, None)
        return np.linalg.norm((_unit(rel) * weight[..., None]).sum(axis=2),
                              axis=-1)

    make = {
        "share": lambda: mask.mean(axis=2),
        "gap1": lambda: w.gap1 / rho,
        "m_goal_away1": lambda: w.margin_of(mask, vmax * ug - vmax * toward),
        "m_away_body": lambda: w.margin_of(mask, -vmax * toward),
        "peer_dist": peer_dist,
        "coast_gap": coast_gap,
        "goal_dist": lambda: np.minimum(
            np.log1p(np.linalg.norm(goal, axis=-1) / rho),
            np.log1p(w.goal_cap / rho)),
        "flock_bearing": flock_bearing,
        "flock_size": lambda: np.linalg.norm(w.flock_push(w.seen, reach_d),
                                             axis=-1),
        "m_body_push": lambda: w.margin_of(mask, vmax * w.body_push),
        "body_push_size": lambda: np.linalg.norm(w.body_push, axis=-1),
        "goal_bearing": lambda: bearing(goal),
        "flock_now_size": flock_now_size,
        "peer_bearing_now": lambda: bearing(nearest_now()),
        "peer_dist_now": lambda: np.minimum(
            np.log1p(np.linalg.norm(nearest_now(), axis=-1) / rho),
            np.log1p(w.drone_cap / rho)),
        "coast_gap1": lambda: np.clip(
            w.body_gaps(w.p + w.dt * w.v).min(axis=2) / w.step, -3.0, 30.0),
        "centroid_dist": lambda: np.minimum(np.log1p(np.linalg.norm(
            (w.seen * (~np.eye(w.m, dtype=bool))[None, ..., None]).sum(axis=2)
            / (w.m - 1) - w.p, axis=-1) / rho), np.log1p(w.drone_cap / rho)),
        "order": lambda: np.broadcast_to(
            np.arange(w.m)[None] / max(w.m - 1, 1) if w.sequential
            else np.zeros((1, w.m)), (w.b, w.m)),
        "m_goal_away1_flock_near": lambda: w.margin_of(
            mask, vmax * ug - vmax * toward + vmax * near_push()),
        "m_goal_flock_near": lambda: w.margin_of(
            mask, vmax * ug + vmax * near_push()),
        "flock_near_size": lambda: np.linalg.norm(near_push(), axis=-1),
        "brake_excess": lambda: approach() - stop_speed(),
        "stop_speed": stop_speed,
    }
    return {name: make[name]() for name in names}


def probe_gaps(w: _Window, separation: float, push: float
               ) -> Tuple[np.ndarray, np.ndarray]:
    s, physics, window = w.state, w.physics, w.window
    boundary = physics.agent_boundary
    targets = targets_of(s, w.config)
    reach = np.where(s.dynamic, window.dynamic_bodies or 0.0,
                     window.static_bodies or 0.0)[None, :]
    if window.walls is not None:
        reach = np.where(s.walls, window.walls, reach)
    reach = np.maximum(reach, 1e-9)
    pos = s.positions.copy()
    vel = s.velocities.copy() if w.velocity else np.zeros_like(pos)
    bodies = s.obstacles.copy()
    k = bodies.shape[1]
    least_d = np.full((w.b, w.m), np.inf)
    least_b = np.full((w.b, w.m), np.inf)
    for _ in range(w.horizon):
        bodies = mover_steps(w.config, bodies, s.obstacle_velocities,
                             s.obstacle_radii, s.dynamic, pos)
        snap = pos.copy()
        for i in range(w.m):
            ref = pos if w.sequential else snap
            here = ref[:, i]
            rel = here[:, None] - ref
            weight = np.clip(1.0 - np.linalg.norm(rel, axis=-1)
                             / window.drones, 0.0, None)
            weight[:, i] = 0.0
            sep = (_unit(rel) * weight[..., None]).sum(axis=1)
            body = 0.0
            if k:
                near = nearest_surface(physics, here[:, None], bodies,
                                       s.obstacle_radii, s.obstacle_axes)[:, 0]
                away = here[:, None] - near
                bw = np.clip(1.0 - np.linalg.norm(away, axis=-1) / reach, 0.0,
                             None)
                body = (_unit(away) * bw[..., None]).sum(axis=1)
            desired = w.vmax * _unit(_unit(targets[:, i] - here)
                                     + separation * sep + push * body)
            if not w.velocity:
                new_v = vel[:, i]
                new_p = here + w.step * _unit(desired)
            elif w.limited:
                turn = desired - vel[:, i]
                size = np.linalg.norm(turn, axis=-1, keepdims=True)
                new_v = vel[:, i] + np.minimum(size, w.amax * w.dt) * _unit(turn)
                new_p = here + w.dt * new_v
            else:
                new_v = desired
                new_p = here + w.dt * new_v
            pos = pos.copy()
            pos[:, i], vel[:, i] = new_p, new_v
        apart = (np.linalg.norm(pos[:, :, None] - pos[:, None], axis=-1)
                 + np.eye(w.m)[None] * 1e12)
        least_d = np.minimum(least_d, apart.min(axis=2) - boundary)
        if k:
            gaps = physics.body_gap(pos[:, :, None], bodies[:, None],
                                    s.obstacle_radii[:, None],
                                    s.obstacle_axes[:, None])
            least_b = np.minimum(least_b, gaps.min(axis=2))
    return (np.clip(least_d / w.step, -3.0, 10.0),
            np.clip(least_b / w.step, -3.0, 10.0))


def _probe_weights(name: str) -> Tuple[float, float]:
    ws, wb = name[len("probe_d_ws"):].split("_wb")
    return float(ws), float(wb)


def _agent(w: _Window, names: Sequence[str]) -> Dict[str, np.ndarray]:
    boundary = w.physics.agent_boundary
    rho = w.rho

    def coast_gap():
        least = np.full((w.b, w.m), np.inf)
        for t in range(1, w.horizon + 1):
            mine = w.p + t * w.dt * w.v
            apart = (np.linalg.norm(mine[:, :, None] - w.paths[t - 1][:, None],
                                    axis=-1) + np.eye(w.m)[None] * 1e12)
            least = np.minimum(least, apart.min(axis=2) - boundary)
        return np.clip(least / rho, -3.0, 3.0)

    def gaps():
        apart = (np.linalg.norm(w.p[:, :, None] - w.p[:, None], axis=-1)
                 + np.eye(w.m)[None] * 1e12)
        return np.sort(apart, axis=2) - boundary

    def nearest():
        apart = (np.linalg.norm(w.p[:, :, None] - w.p[:, None], axis=-1)
                 + np.eye(w.m)[None] * 1e12)
        return apart.argmin(axis=2)

    def goal_at_me():
        j = nearest()
        there = w.p[w.rows, j]
        target = targets_of(w.state, w.config)[w.rows, j]
        return np.sum(_unit(target - there) * _unit(w.p - there), axis=-1)

    def body_gap_of_nearest():
        s = w.state
        gaps_now = w.physics.body_gap(w.p[:, :, None], s.obstacles[:, None],
                                      s.obstacle_radii[:, None],
                                      s.obstacle_axes[:, None]).min(axis=2)
        return np.take_along_axis(gaps_now, nearest(), 1) / rho

    make = {"d_share": lambda: w.drone_set.mean(axis=2),
            "d_coast_gap": coast_gap,
            "d_gap1": lambda: gaps()[..., 0] / rho,
            "d_gap2": lambda: np.minimum(gaps()[..., min(1, w.m - 2)],
                                         w.drone_cap) / rho,
            "hz_goal_at_me": goal_at_me,
            "hz_body_gap": body_gap_of_nearest}
    out = {}
    probes: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
    for n in names:
        if n.startswith(("probe_d_", "probe_b_")):
            weights = _probe_weights(n)
            if weights not in probes:
                probes[weights] = probe_gaps(w, *weights)
            out[n] = probes[weights][0 if n.startswith("probe_d_") else 1]
        elif n in make:
            out[n] = make[n]()
    out.update(_obstacle(w, [n for n in names if n not in out]))
    return out


def _timeout(w: _Window, names: Sequence[str]) -> Dict[str, np.ndarray]:
    target = w.state.target
    goal = target[:, None] - w.p
    d0 = np.linalg.norm(goal, axis=-1)
    ug = _unit(goal)
    need = w.model.reach.need()
    span = w.horizon * w.step
    vmax = w.vmax

    def free_best():
        if w.limited:
            centre = w.p + w.horizon * w.dt * w.v
            ends = centre[:, :, None, :] + w.rho * _U[None, None]
        else:
            ends = w.p[:, :, None, :] + span * _U[None, None]
        gain = d0[..., None] - np.linalg.norm(ends - target[:, None, None],
                                              axis=-1)
        blocked = w.body_set | w.drone_set
        best = np.where(blocked, -np.inf, gain).max(axis=2)
        return np.clip(np.where(np.isfinite(best), best, -span) / need,
                       -20.0, 20.0)

    def flock_along():
        rel = w.p[:, None] - w.p[:, :, None]
        distance = np.linalg.norm(rel, axis=-1) + np.eye(w.m)[None] * 1e12
        weight = np.clip(1.0 - distance / w.window.drones, 0.0, None)
        push = -(_unit(rel) * weight[..., None]).sum(axis=2)
        return np.sum(push * ug, axis=-1)

    def swarm(values):
        return np.broadcast_to(values[:, None], (w.b, w.m))

    make = {
        "s_centroid_dist": lambda: swarm(np.log1p(np.linalg.norm(
            w.p.mean(axis=1) - target, axis=-1) / span)),
        "s_v_goal": lambda: swarm((np.sum(w.v_state * ug, axis=-1)
                                   / vmax).mean(axis=1)),
        "t_v_goal": lambda: np.sum(w.v_state * ug, axis=-1) / vmax,
        "t_v_cross": lambda: np.abs(w.v_state[..., 0] * ug[..., 1]
                                    - w.v_state[..., 1] * ug[..., 0]) / vmax,
        "t_mates_v_along": lambda: np.sum(
            (w.v_state[:, None] * (~np.eye(w.m, dtype=bool))[None, ..., None])
            .sum(axis=2) / (w.m - 1) * ug, axis=-1) / vmax,
        "t_body_push_along": lambda: np.sum(w.body_push_now * ug, axis=-1),
        "t_free_best_over_need": free_best,
        "t_flock_along": flock_along,
        "t_coast_over_need": lambda: np.clip((d0 - np.linalg.norm(
            w.p + w.horizon * w.dt * w.v - target[:, None], axis=-1)) / need,
            -20.0, 20.0),
    }
    out = _obstacle(w, [n for n in names if n not in make])
    out.update({n: make[n]() for n in names if n in make})
    return out


_BLOCKS = {"obstacle_collision": (_obstacle, ("share", "m_goal_away1", "gap1")),
           "agent_collision": (_agent, ("d_share", "probe_d_ws0.3_wb3",
                                        "d_coast_gap")),
           "timeout": (_timeout, None)}


def drone_features(model, state: State, names: Sequence[str] = ()
                   ) -> np.ndarray:
    names = tuple(names) or SPACES[model.name]
    block = _BLOCKS[model.name][0]
    out = np.zeros((state.size, state.drones, len(names)), dtype=np.float32)
    for start in range(0, state.size, CHUNK):
        part = state.take(np.arange(start, min(state.size, start + CHUNK)))
        values = block(_Window(part, model), names)
        out[start:start + part.size] = np.stack(
            [np.asarray(values[n], dtype=np.float32) for n in names], axis=-1)
    return out


def window_drone(values: np.ndarray) -> np.ndarray:
    risk, choice, tie = values[..., 0], values[..., 1], values[..., 2]
    at_risk = risk > 0
    at_risk |= ~at_risk.any(axis=1, keepdims=True)
    return np.lexsort((tie, np.where(at_risk, choice, np.inf)), axis=1)[:, 0]


def window_points(model, state: State) -> np.ndarray:
    names = SPACES[model.name]
    chooser = _BLOCKS[model.name][1]
    if chooser is None:
        return drone_features(model, state, names).mean(axis=1)
    extra = tuple(n for n in chooser if n not in names)
    values = drone_features(model, state, names + extra)
    columns = names + extra
    pick = window_drone(values[..., [columns.index(n) for n in chooser]])
    return values[np.arange(state.size), pick, :len(names)]


__all__ = ["CHUNK", "DIRECTIONS", "FEATURES_VERSION", "SPACES",
           "drone_features", "margin", "nearest_surface", "window_drone",
           "window_points"]
