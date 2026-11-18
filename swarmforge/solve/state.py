from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Box, FamilyConfig, ModelSpec, Span, surface_distance


@dataclass
class State:
    positions: np.ndarray
    velocities: np.ndarray
    obstacles: np.ndarray
    obstacle_velocities: np.ndarray
    obstacle_radii: np.ndarray
    dynamic: np.ndarray
    target: np.ndarray
    route: np.ndarray
    obstacle_axes: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.obstacle_axes is None:
            self.obstacle_axes = np.zeros(np.shape(self.obstacles))

    @property
    def walls(self) -> np.ndarray:
        return np.any(self.obstacle_axes != 0.0, axis=-1)

    @property
    def size(self) -> int:
        return int(self.positions.shape[0])

    @property
    def drones(self) -> int:
        return int(self.positions.shape[1])

    def take(self, index) -> "State":
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = value if f.name == "dynamic" else value[index]
        return State(**out)

    def copy(self) -> "State":
        return State(**{f.name: np.array(getattr(self, f.name), copy=True)
                        for f in fields(self)})

    def to_dicts(self) -> List[Dict]:
        out = []
        for b in range(self.size):
            out.append({
                "positions": self.positions[b].tolist(),
                "velocities": self.velocities[b].tolist(),
                "obstacles": self.obstacles[b].tolist(),
                "obstacle_velocities": self.obstacle_velocities[b].tolist(),
                "obstacle_radii": self.obstacle_radii[b].tolist(),
                "dynamic": self.dynamic.tolist(),
                "target": self.target[b].tolist(),
                "route": self.route[b].tolist(),
                "obstacle_axes": self.obstacle_axes[b].tolist(),
            })
        return out

    @classmethod
    def from_dicts(cls, rows: List[Dict]) -> "State":
        def stack(key, shape_tail):
            return np.array([row[key] for row in rows],
                            dtype=float).reshape((len(rows),) + shape_tail)
        first = rows[0]
        m, k = len(first["positions"]), len(first["obstacles"])
        return cls(positions=stack("positions", (m, 2)),
                   velocities=stack("velocities", (m, 2)),
                   obstacles=stack("obstacles", (k, 2)),
                   obstacle_velocities=stack("obstacle_velocities", (k, 2)),
                   obstacle_radii=stack("obstacle_radii", (k,)),
                   dynamic=np.array(first["dynamic"], dtype=bool),
                   target=stack("target", (2,)),
                   route=stack("route", (2,)),
                   obstacle_axes=(stack("obstacle_axes", (k, 2))
                                  if "obstacle_axes" in first else None))


def _pair_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, :, None, :] - b[:, None, :, :], axis=-1)


def admissible(state: State, config: FamilyConfig) -> np.ndarray:
    physics = config.physics
    ok = np.ones(state.size, dtype=bool)
    m = state.drones
    if m > 1:
        d = _pair_distances(state.positions, state.positions)
        iu = np.triu_indices(m, 1)
        ok &= np.all(d[:, iu[0], iu[1]] >= physics.agent_boundary, axis=1)
    k = state.obstacles.shape[1]
    if k:
        gap = physics.body_gap(state.positions[:, :, None, :],
                               state.obstacles[:, None, :, :],
                               state.obstacle_radii[:, None, :],
                               state.obstacle_axes[:, None, :, :])
        ok &= np.all(gap >= 0.0, axis=(1, 2))
    if k > 1:
        iu = np.triu_indices(k, 1)
        d = _pair_distances(state.obstacles, state.obstacles)[:, iu[0], iu[1]]
        apart, wall = _body_separation(state, iu[0], iu[1], d)
        ok &= np.all(apart >= 0.0, axis=1)
        ranges = config.state.static
        spacing = ranges.min_spacing if ranges else 0.0
        static = ~state.dynamic
        both = static[iu[0]] & static[iu[1]]
        if spacing > 0.0 and np.any(both):
            corridor = max(0.0, spacing - 2.0 * ranges.radius.hi)
            need = np.where(wall, apart - corridor, d - spacing)
            ok &= np.all(need[:, both] >= 0.0, axis=1)
    return ok


def _wall_frame(radii: np.ndarray, axes: np.ndarray):
    length = np.linalg.norm(axes, axis=-1)
    wall = length > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        along = np.where(wall[..., None], axes / length[..., None],
                         np.array([1.0, 0.0]))
    across = np.stack([-along[..., 1], along[..., 0]], axis=-1)
    return along, across, np.where(wall, length, radii), radii


def _rectangle_gap(ci, ri, ai, cj, rj, aj) -> np.ndarray:
    fi, fj = _wall_frame(ri, ai), _wall_frame(rj, aj)
    offset = cj - ci
    separated = np.zeros(offset.shape[:-1], dtype=bool)
    for axis in (fi[0], fi[1], fj[0], fj[1]):
        reach = sum(f[2] * np.abs(np.sum(f[0] * axis, axis=-1))
                    + f[3] * np.abs(np.sum(f[1] * axis, axis=-1))
                    for f in (fi, fj))
        separated |= np.abs(np.sum(offset * axis, axis=-1)) > reach
    signs = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])

    def corners(c, f):
        along, across, ha, hc = f
        return (c[..., None, :]
                + signs[:, :1] * (ha[..., None, None] * along[..., None, :])
                + signs[:, 1:] * (hc[..., None, None] * across[..., None, :]))

    to_j = surface_distance(corners(ci, fi) - cj[..., None, :], rj[..., None],
                            aj[..., None, :]).min(axis=-1)
    to_i = surface_distance(corners(cj, fj) - ci[..., None, :], ri[..., None],
                            ai[..., None, :]).min(axis=-1)
    return np.where(separated, np.minimum(to_i, to_j), -1.0)


def _body_separation(state: State, i: np.ndarray, j: np.ndarray,
                     d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c, r, a = state.obstacles, state.obstacle_radii, state.obstacle_axes
    wall = state.walls | (r <= 0.0)
    wi, wj = wall[:, i], wall[:, j]
    discs = d - r[:, i] - r[:, j]
    if not np.any(wi | wj):
        return discs, wi | wj
    wall_disc = surface_distance(c[:, j] - c[:, i], r[:, i], a[:, i]) - r[:, j]
    disc_wall = surface_distance(c[:, i] - c[:, j], r[:, j], a[:, j]) - r[:, i]
    walls = _rectangle_gap(c[:, i], r[:, i], a[:, i], c[:, j], r[:, j], a[:, j])
    out = np.where(wi & wj, walls,
                   np.where(wi, wall_disc, np.where(wj, disc_wall, discs)))
    return out, wi | wj


def _box(rng: np.random.Generator, box: Box, shape: Tuple[int, ...]
         ) -> np.ndarray:
    x = rng.uniform(box.x.lo, box.x.hi, shape)
    y = rng.uniform(box.y.lo, box.y.hi, shape)
    return np.stack([x, y], axis=-1)


def _by_area(rng: np.random.Generator, span: Span, shape: Tuple[int, ...]
             ) -> np.ndarray:
    u = rng.uniform(0.0, 1.0, shape)
    return np.sqrt(span.lo ** 2 + u * (span.hi ** 2 - span.lo ** 2))


def _heading(rng: np.random.Generator, degrees: Optional[Span],
             shape: Tuple[int, ...]) -> np.ndarray:
    lo, hi = (-180.0, 180.0) if degrees is None else (degrees.lo, degrees.hi)
    angle = np.radians(rng.uniform(lo, hi, shape))
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


class Sampler:
    def __init__(self, config: FamilyConfig, spec: ModelSpec):
        self.config = config
        self.spec = spec

    def draw(self, rng: np.random.Generator, n: int) -> State:
        config, spec, st = self.config, self.spec, self.config.state
        m, ks, kd = spec.drones, spec.static_obstacles, spec.dynamic_obstacles
        positions = _box(rng, st.drone_region, (n, m))
        if st.drone_speed is not None:
            velocities = (_by_area(rng, st.drone_speed, (n, m))[..., None]
                          * _heading(rng, None, (n, m)))
        else:
            velocities = np.zeros((n, m, 2))

        vels, radii, axes = [], [], []
        if ks:
            vels.append(np.zeros((n, ks, 2)))
            drawn = rng.uniform(st.static.radius.lo, st.static.radius.hi,
                                (n, ks))
            wall_axes = np.zeros((n, ks, 2))
            if st.static.wall_share:
                walls = rng.random((n, ks)) < st.static.wall_share
                extent, length, thick = (
                    rng.uniform(span.lo, span.hi, (n, ks))
                    for span in (st.static.wall_extent, st.static.wall_length,
                                 st.static.wall_thickness))
                angle = rng.uniform(0.0, np.pi, (n, ks))
                scale = extent / np.hypot(length, thick)
                drawn = np.where(walls, thick * scale, drawn)
                wall_axes = np.where(
                    walls[..., None], (length * scale)[..., None]
                    * np.stack([np.cos(angle), np.sin(angle)], axis=-1), 0.0)
            radii.append(drawn)
            axes.append(wall_axes)
        if kd:
            vels.append(_by_area(rng, st.dynamic.speed, (n, kd))[..., None]
                        * _heading(rng, None, (n, kd)))
            radii.append(rng.uniform(st.dynamic.radius.lo,
                                     st.dynamic.radius.hi, (n, kd)))
            axes.append(np.zeros((n, kd, 2)))
        obstacles = np.repeat(positions[:, :1], ks + kd, axis=1)
        obstacle_velocities = (np.concatenate(vels, axis=1) if vels
                               else np.zeros((n, 0, 2)))
        obstacle_radii = (np.concatenate(radii, axis=1) if radii
                          else np.zeros((n, 0)))
        obstacle_axes = (np.concatenate(axes, axis=1) if axes
                         else np.zeros((n, 0, 2)))
        dynamic = np.array([False] * ks + [True] * kd, dtype=bool)

        if config.physics.targets.kind == "shared_goal":
            target = positions[:, 0, :].copy()
            route = np.zeros((n, 2))
        else:
            offset = (_by_area(rng, st.waypoint_distance, (n,))[:, None]
                      * _heading(rng, st.waypoint_bearing_deg, (n,)))
            target = positions[:, 0, :] + offset
            route = _heading(rng, st.route_heading_deg, (n,))
            if st.route_final_share:
                final = rng.random(n) < st.route_final_share
                route = np.where(final[:, None], 0.0, route)
        return State(positions, velocities, obstacles, obstacle_velocities,
                     obstacle_radii, dynamic, target, route, obstacle_axes)


def concat(states: List[State]) -> State:
    out = {}
    for f in fields(State):
        if f.name == "dynamic":
            out[f.name] = states[0].dynamic
        else:
            out[f.name] = np.concatenate([getattr(s, f.name) for s in states])
    return State(**out)


__all__ = ["Sampler", "State", "admissible", "concat"]
