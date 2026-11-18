from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import FamilyConfig, ModelSpec
from .laws import norm
from .reach import AgentReach, SwarmReach
from .state import State
from .window import in_disc, in_ring


class HoldReach:
    def members(self, drones: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def margin(self, state) -> np.ndarray:
        raise NotImplementedError

    def held(self, state: State) -> np.ndarray:
        return self.margin(state) >= 0.0


def _drones(drones: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.arange(drones), np.full(drones, -1)


def minimax(positions: np.ndarray) -> np.ndarray:
    out = norm(positions[:, :, None] - positions[:, None])
    for k in range(positions.shape[1]):
        out = np.minimum(out, np.maximum(out[:, :, k:k + 1],
                                         out[:, k:k + 1, :]))
    return out


def joined(linked: np.ndarray, drones: int) -> np.ndarray:
    first, second = AgentReach.pairs(drones)
    adjacency = np.zeros((linked.shape[0], drones, drones))
    adjacency[:, first, second] = linked
    adjacency[:, second, first] = linked
    adjacency += np.eye(drones)[None]
    reach = adjacency > 0.0
    span = 1
    while span < drones - 1:
        square = reach.astype(float)
        reach = np.matmul(square, square) > 0.0
        span *= 2
    return reach[:, first, second]


class LinkReach(HoldReach, AgentReach):
    def __init__(self, config: FamilyConfig, horizon: int,
                 sense_range: Optional[float] = None):
        super().__init__(config, horizon)
        if sense_range is None or not sense_range > 0.0:
            raise ValueError("the fragmentation question needs "
                             "models.fragmentation.sense_range, the centre "
                             "distance within which two drones sense each "
                             "other")
        if sense_range <= config.physics.agent_boundary:
            raise ValueError("models.fragmentation.sense_range %.6g is no "
                             "wider than contact (%.6g): no admissible pair "
                             "is linked" % (sense_range,
                                            config.physics.agent_boundary))
        self.sense_range = float(sense_range)

    @property
    def ring(self) -> Tuple[float, float]:
        return (max(self.boundary, self.sense_range - self.ring_width),
                self.sense_range)

    def in_ring(self, state: State) -> np.ndarray:
        first, second = self.pairs(state.drones)
        d = norm(state.positions[:, first] - state.positions[:, second])
        inner, outer = self.ring
        return (d > inner) & (d <= outer)

    def members(self, drones: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.pairs(drones)

    def margin(self, state) -> np.ndarray:
        first, second = self.pairs(state.positions.shape[1])
        return (self.sense_range
                - minimax(state.positions)[:, first, second])

    def split(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        held = self.held(state)
        m = state.drones
        safe = np.ones(state.size, dtype=bool)
        doomed = np.zeros(state.size, dtype=bool)
        for t in range(1, self.horizon + 1):
            apart, together = self.pair_bounds(state, t)
            surely = joined(together <= self.sense_range, m)
            possibly = joined(apart <= self.sense_range, m)
            safe &= np.all(~held | surely, axis=1)
            doomed |= np.any(held & ~possibly, axis=1)
        return safe, doomed

    def describe(self, spec: ModelSpec) -> Dict[str, Any]:
        return {"ring": list(self.ring)}


def turn_reach(speed: np.ndarray, radius: float, least: float) -> np.ndarray:
    speed = np.asarray(speed, dtype=float)
    with np.errstate(all="ignore"):
        tangent = np.arcsin(np.clip(radius / speed, 0.0, 1.0))
        tangent_speed = np.sqrt(np.maximum(speed * speed - radius * radius,
                                           0.0))
        if least > 0.0:
            cut = np.arccos(np.clip(
                (least * least + speed * speed - radius * radius)
                / (2.0 * least * speed), -1.0, 1.0))
        else:
            cut = np.full(speed.shape, np.pi)
    out = np.where((radius < speed) & (tangent_speed >= least), tangent, cut)
    out = np.where(radius >= speed + least, np.pi, out)
    return np.where(np.isfinite(out), out, np.pi)


class HeadingReach(HoldReach, SwarmReach):
    def __init__(self, config: FamilyConfig, horizon: int,
                 max_angle_deg: Optional[float] = None,
                 min_speed: Optional[float] = None):
        super().__init__(config, horizon)
        if config.physics.plant.integration != "velocity":
            raise ValueError("the misalignment question reads the drones' "
                             "velocities, and %s's displacement plant "
                             "carries none" % config.name)
        if max_angle_deg is None or not 0.0 < max_angle_deg < 180.0:
            raise ValueError("the misalignment question needs "
                             "models.misalignment.max_angle_deg, the widest "
                             "two headings may be apart, in (0, 180)")
        self.max_angle_deg = float(max_angle_deg)
        self.max_angle = math.radians(self.max_angle_deg)
        self.min_speed = float(min_speed or 0.0)
        self.turn = None if self.accel is None else self.accel * self.dt

    def members(self, drones: int) -> Tuple[np.ndarray, np.ndarray]:
        return AgentReach.pairs(drones)

    def headed(self, velocities: np.ndarray) -> np.ndarray:
        speed = norm(velocities)
        return (speed >= self.min_speed) & (speed > 0.0)

    @staticmethod
    def angles(velocities: np.ndarray) -> np.ndarray:
        first, second = AgentReach.pairs(velocities.shape[1])
        a, b = velocities[:, first], velocities[:, second]
        cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        return np.arctan2(np.abs(cross), np.sum(a * b, axis=-1))

    def margin(self, state) -> np.ndarray:
        first, second = AgentReach.pairs(state.velocities.shape[1])
        headed = self.headed(state.velocities)
        both = headed[:, first] & headed[:, second]
        return np.where(both, self.max_angle_deg
                        - np.degrees(self.angles(state.velocities)), np.inf)

    def split(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        held = self.held(state)
        if self.turn is None:
            fast = self.step / self.dt >= self.min_speed
            return ~(np.any(held, axis=1) & fast), np.zeros(state.size,
                                                             dtype=bool)
        first, second = self.members(state.drones)
        speed = norm(state.velocities)
        angle = self.angles(state.velocities)
        least = self.min_speed
        safe = np.ones(state.size, dtype=bool)
        doomed = np.zeros(state.size, dtype=bool)
        for t in range(1, self.horizon + 1):
            radius = t * self.turn
            spread = turn_reach(speed, radius, least)
            can = (speed + radius >= least) & (speed + radius > 0.0)
            must = (speed - radius >= least) & (speed - radius > 0.0)
            widest = angle + spread[:, first] + spread[:, second]
            narrowest = angle - spread[:, first] - spread[:, second]
            turns = can[:, first] & can[:, second] & (widest > self.max_angle)
            turned = (must[:, first] & must[:, second]
                      & (narrowest > self.max_angle))
            safe &= np.all(~held | ~turns, axis=1)
            doomed |= np.any(held & turned, axis=1)
        return safe, doomed

    def require(self, spec: ModelSpec) -> None:
        if spec.drones < 2:
            raise ValueError("a heading is held between two drones, and this "
                             "model draws %d" % spec.drones)


class RangeReach(HoldReach, SwarmReach):
    def __init__(self, config: FamilyConfig, horizon: int,
                 target_range: Optional[float] = None):
        super().__init__(config, horizon)
        if self.leader is not None:
            raise ValueError("the loss-of-target question asks of a shared "
                             "goal, and %s flies a leader's formation"
                             % config.name)
        if target_range is None or not target_range > 0.0:
            raise ValueError("the loss-of-target question needs "
                             "models.loss_target.target_range, how near its "
                             "target a drone holds it")
        self.target_range = float(target_range)
        self.ring_width = self.horizon * self.step
        self.goal_reach = self.target_range

    @property
    def ring(self) -> Tuple[float, float]:
        return (max(0.0, self.target_range - self.ring_width),
                self.target_range)

    def members(self, drones: int) -> Tuple[np.ndarray, np.ndarray]:
        return _drones(drones)

    def margin(self, state) -> np.ndarray:
        return self.target_range - norm(state.positions
                                        - state.target[:, None])

    def in_ring(self, state: State) -> np.ndarray:
        d = norm(state.positions - state.target[:, None])
        inner, outer = self.ring
        return (d > inner) & (d <= outer)

    def split(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        held = self.held(state)
        goal = state.target[:, None]
        safe = np.ones(state.size, dtype=bool)
        doomed = np.zeros(state.size, dtype=bool)
        for t in range(1, self.horizon + 1):
            nearest = np.zeros(held.shape)
            farthest = np.full(held.shape, np.inf)
            for centres, radius in self.discs(state, t):
                distance = norm(centres - goal)
                nearest = np.maximum(nearest, distance - radius)
                farthest = np.minimum(farthest, distance + radius)
            safe &= np.all(~held | (farthest <= self.target_range), axis=1)
            doomed |= np.any(held & (nearest > self.target_range), axis=1)
        return safe, doomed

    def describe(self, spec: ModelSpec) -> Dict[str, Any]:
        return {"drone_reaches": [self.drone_reach(e)
                                  for e in range(spec.drones)],
                "ring": list(self.ring)}

    def anchors(self, state: State) -> np.ndarray:
        return super().anchors(state) & self.in_ring(state)

    def place(self, batch: State, rng: np.random.Generator,
              spec: ModelSpec) -> np.ndarray:
        size = batch.size
        rows = np.arange(size)
        m = spec.drones
        log_w = self._drone_log_weights(m)
        weights = np.exp(log_w - log_w.max())
        drone = rng.choice(m, size=size, p=weights / weights.sum())
        centre = batch.positions[rows, drone].copy()
        reach = self.body_reaches(batch)
        for e in range(batch.obstacles.shape[1]):
            batch.obstacles[:, e] = in_disc(rng, centre, reach[:, e])
        for e in range(m):
            other = np.flatnonzero(drone != e)
            batch.positions[other, e] = in_disc(rng, centre[other],
                                                self.drone_reach(e))
        inner, outer = self.ring
        batch.target = in_ring(rng, centre, inner, outer)
        count = self.anchors(batch).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            keep = np.where(count > 0, self.room_share(batch) / count, 0.0)
        return rng.random(size) < keep


class PathReach(HoldReach, SwarmReach):
    def __init__(self, config: FamilyConfig, horizon: int,
                 path_width: Optional[float] = None):
        super().__init__(config, horizon)
        if self.leader is None:
            raise ValueError("the off-path question follows a leader's "
                             "route, and %s flies to a shared goal"
                             % config.name)
        if path_width is None or not path_width > 0.0:
            raise ValueError("the off-path question needs "
                             "models.off_path.path_width, how far from its "
                             "route a drone may stray")
        self.path_width = float(path_width)

    def members(self, drones: int) -> Tuple[np.ndarray, np.ndarray]:
        return _drones(drones)

    @staticmethod
    def offsets(points: np.ndarray, state) -> np.ndarray:
        length = norm(state.route)
        with np.errstate(divide="ignore", invalid="ignore"):
            along = state.route / length[:, None]
        rel = points - state.target[:, None]
        cross = (rel[..., 0] * along[:, None, 1]
                 - rel[..., 1] * along[:, None, 0])
        return np.where(length[:, None] > 1e-12, np.abs(cross), np.inf)

    def margin(self, state) -> np.ndarray:
        return self.path_width - self.offsets(state.positions, state)

    def split(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        held = self.held(state)
        safe = np.ones(state.size, dtype=bool)
        doomed = np.zeros(state.size, dtype=bool)
        for t in range(1, self.horizon + 1):
            nearest = np.zeros(held.shape)
            farthest = np.full(held.shape, np.inf)
            for centres, radius in self.discs(state, t):
                offset = self.offsets(centres, state)
                nearest = np.maximum(nearest, offset - radius)
                farthest = np.minimum(farthest, offset + radius)
            safe &= np.all(~held | (farthest <= self.path_width), axis=1)
            doomed |= np.any(held & (nearest > self.path_width), axis=1)
        return safe, doomed


__all__ = ["HeadingReach", "HoldReach", "LinkReach", "PathReach",
           "RangeReach", "joined", "minimax", "turn_reach"]
