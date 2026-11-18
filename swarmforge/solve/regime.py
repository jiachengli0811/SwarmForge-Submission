from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import FamilyConfig
from .state import State

MODELS = ("obstacle_collision", "agent_collision", "timeout")


def _log_uniform(rng: np.random.Generator, lo, hi, shape) -> np.ndarray:
    lo = np.minimum(lo, hi)
    return np.exp(rng.uniform(np.log(lo), np.log(hi), shape))


def _turn(v: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def anchors(state: State, reach) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .reach import AgentReach, ObstacleReach

    b, m = state.positions.shape[:2]
    k = state.obstacles.shape[1]
    rows = np.arange(b)
    drones, bodies = np.zeros((b, m), bool), np.zeros((b, k), bool)
    if isinstance(reach, ObstacleReach):
        ring = reach.in_ring(state)
        gap = reach._gaps(state.positions, state.obstacles, state.obstacle_radii, state.obstacle_axes)
        i, j = np.unravel_index(np.where(ring, gap, np.inf).reshape(b, -1).argmin(1), (m, k))
        drones[rows, i] = bodies[rows, j] = True
        centre = state.positions[rows, i].copy()
    elif isinstance(reach, AgentReach):
        first, second = reach.pairs(m)
        p = reach.in_ring(state).argmax(1)
        i, j = first[p], second[p]
        drones[rows, i] = drones[rows, j] = True
        centre = 0.5 * (state.positions[rows, i] + state.positions[rows, j])
    else:
        raise ValueError("the regime layer draws contact windows only, not %s" % type(reach).__name__)
    if reach.leader is not None:
        drones[:, reach.leader] = True
    return drones, bodies, centre, i


def startup_speed(config: FamilyConfig, horizon: int, pace: float) -> float:
    plant, dt = config.physics.plant, config.physics.dt
    if plant.max_accel is None:
        return 0.0
    gain = plant.max_accel * dt
    need = max(0.0, pace / dt - gain * (horizon + 1) / 2.0)
    return min(np.ceil(need / gain - 1e-12) * gain, plant.max_step / dt)


def start(state: State, reach, config: FamilyConfig) -> State:
    from .axes import targets_of

    speed = startup_speed(config, reach.horizon, reach.pace)
    if speed == 0.0:
        state.velocities = np.zeros_like(state.velocities)
        return state
    to = targets_of(state, config) - state.positions
    size = np.linalg.norm(to, axis=-1, keepdims=True)
    state.velocities = speed * np.where(size > 0, to / np.maximum(size, 1e-12), 0.0)
    return state


def transform(state: State, reach, config: FamilyConfig, rng: np.random.Generator) -> State:
    from .axes import targets_of
    from .reach import TimeoutReach

    if isinstance(reach, TimeoutReach):
        return start(state, reach, config)
    physics, ranges = config.physics, config.state
    b, m = state.positions.shape[:2]
    k = state.obstacles.shape[1]
    rows = np.arange(b)
    fixed, fixed_bodies, centre, anchor = anchors(state, reach)
    farthest = max(reach.drone_reach(e) for e in range(m))
    f = _log_uniform(rng, physics.agent_boundary / farthest, 2.0, (b, 1, 1))
    p = state.positions
    state.positions = np.where(fixed[..., None], p, centre[:, None] + f * (p - centre[:, None]))
    if k:
        h = _log_uniform(rng, 1.0, 2.0, (b, 1, 1))
        o = state.obstacles
        state.obstacles = np.where(fixed_bodies[..., None], o, centre[:, None] + h * (o - centre[:, None]))
    pinned = reach.leader is not None
    if not pinned:
        r = reach.reaches.target
        d = np.where(rng.random(b) < 0.5, r * np.sqrt(rng.random(b)),
                     _log_uniform(rng, physics.plant.max_step, r, (b,)))
        u = state.target - centre
        a = np.where(np.linalg.norm(u, axis=1) > 0, np.arctan2(u[:, 1], u[:, 0]), rng.uniform(-np.pi, np.pi, b))
        state.target = centre + d[:, None] * np.stack([np.cos(a), np.sin(a)], -1)
    pa = state.positions[rows, anchor]
    own = targets_of(state, config)[rows, anchor] - pa
    if k:
        rel = state.obstacles - pa[:, None]
        nearest = rel[rows, np.argmin(np.linalg.norm(rel, axis=-1), axis=1)]
        want = np.arctan2(nearest[:, 1], nearest[:, 0]) + rng.uniform(0.0, np.pi, b) * rng.uniform(-1, 1, b)
        alpha = want - np.arctan2(own[:, 1], own[:, 0])
        if pinned:
            state.obstacles = pa[:, None] + _turn(rel, -alpha[:, None])
            state.obstacle_velocities = _turn(state.obstacle_velocities, -alpha[:, None])
            state.obstacle_axes = _turn(state.obstacle_axes, -alpha[:, None])
        else:
            state.target = pa + _turn(state.target - pa, alpha)
            own = state.target - pa
    speeds = ranges.drone_speed
    if speeds is not None:
        common = np.arctan2(own[:, 1], own[:, 0])[:, None] + rng.uniform(0.0, np.pi, (b, 1)) * rng.uniform(-1, 1, (b, 1))
        heading = common + rng.uniform(0.0, np.pi, (b, 1)) * rng.uniform(-1, 1, (b, m))
        top = min(speeds.hi, physics.plant.max_step / physics.dt)
        level = np.sqrt(rng.uniform(speeds.lo ** 2, top ** 2, (b, 1)))
        tolerance = rng.uniform(0.0, 1.0, (b, 1)) * (top - speeds.lo)
        free = np.clip(level + tolerance * rng.uniform(-1, 1, (b, m)), speeds.lo, top)
        speed = np.where(rng.random((b, 1)) < 0.5, top, free)
        state.velocities = speed[..., None] * np.stack([np.cos(heading), np.sin(heading)], -1)
    return state


def redraw(batch: State, share: float, reach, config: FamilyConfig, rng: np.random.Generator) -> np.ndarray:
    chosen = rng.random(batch.size) < share
    rows = np.flatnonzero(chosen)
    if rows.size:
        part = transform(batch.take(rows), reach, config, rng)
        for name in ("positions", "velocities", "obstacles", "obstacle_velocities", "obstacle_axes", "target"):
            getattr(batch, name)[rows] = getattr(part, name)
    return chosen


__all__ = ["MODELS", "anchors", "redraw", "start", "startup_speed", "transform"]
