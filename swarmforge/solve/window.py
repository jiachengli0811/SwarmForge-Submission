from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import FamilyConfig


@dataclass(frozen=True)
class Reaches:
    drones: float
    static_bodies: float
    dynamic_bodies: float
    target: float
    walls: Optional[float] = None

    @classmethod
    def of(cls, config: FamilyConfig, horizon: int) -> "Reaches":
        physics, st = config.physics, config.state
        window = st.window
        step, dt, h = physics.plant.max_step, physics.dt, int(horizon)
        earlier = 1 if physics.plant.update_order == "sequential" else 0
        own = (h - 1) * step
        peer = (2 * (h - 1) + earlier) * step
        drones = max(window.drones + peer,
                     physics.agent_boundary + 2 * h * step)
        static = dynamic = 0.0
        if st.static is not None:
            static = max(window.static_bodies + own,
                         physics.contact_reach(st.static.radius.hi)
                         + h * step)
        if st.dynamic is not None:
            flight = h * dt * st.dynamic.speed.hi
            dynamic = max(window.dynamic_bodies + own + flight,
                          physics.contact_reach(st.dynamic.radius.hi)
                          + h * step + flight)
        walls = None
        if st.static is not None and st.static.wall_share:
            walls = max(window.walls + own,
                        physics.agent_radius + st.static.wall_extent.hi
                        + physics.obstacle_min_gap + h * step)
        leader = physics.targets.kind == "leader_formation"
        return cls(drones=drones, static_bodies=static,
                   dynamic_bodies=dynamic,
                   target=window.target + (peer if leader else own),
                   walls=walls)


def union_area(radius: float, distance) -> np.ndarray:
    d = np.minimum(np.asarray(distance, dtype=float), 2.0 * radius)
    with np.errstate(invalid="ignore", divide="ignore"):
        lens = (2.0 * radius ** 2 * np.arccos(d / (2.0 * radius))
                - 0.5 * d * np.sqrt(np.maximum(4.0 * radius ** 2 - d * d,
                                               0.0)))
    return 2.0 * np.pi * radius ** 2 - np.where(radius > 0.0, lens, 0.0)


def in_ring(rng: np.random.Generator, centres: np.ndarray,
            inner, outer) -> np.ndarray:
    inner = np.broadcast_to(np.asarray(inner, dtype=float), centres.shape[:1])
    outer = np.broadcast_to(np.asarray(outer, dtype=float), centres.shape[:1])
    radius = np.sqrt(inner ** 2 + rng.random(inner.shape)
                     * (outer ** 2 - inner ** 2))
    angle = rng.uniform(-np.pi, np.pi, inner.shape)
    return centres + radius[:, None] * np.stack([np.cos(angle),
                                                 np.sin(angle)], axis=-1)


def in_disc(rng: np.random.Generator, centres: np.ndarray,
            radius) -> np.ndarray:
    return in_ring(rng, centres, 0.0, radius)


def in_union(rng: np.random.Generator, first: np.ndarray, second: np.ndarray,
             radius) -> np.ndarray:
    out = np.empty_like(first)
    radius = np.broadcast_to(np.asarray(radius, dtype=float), first.shape[:1])
    todo = np.arange(first.shape[0])
    while todo.size:
        pick = rng.random(todo.size) < 0.5
        centre = np.where(pick[:, None], first[todo], second[todo])
        point = in_disc(rng, centre, radius[todo])
        other = np.where(pick[:, None], second[todo], first[todo])
        both = np.linalg.norm(point - other, axis=-1) <= radius[todo]
        keep = ~both | (rng.random(todo.size) < 0.5)
        out[todo[keep]] = point[keep]
        todo = todo[~keep]
    return out


__all__ = ["Reaches", "in_disc", "in_ring", "in_union", "union_area"]
