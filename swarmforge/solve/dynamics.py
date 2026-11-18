from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Dict, List, Optional

import numpy as np

from .config import FamilyConfig
from .laws import Controller, View, norm, unit
from .record import Record
from .state import State


@dataclass
class World:
    positions: np.ndarray
    velocities: np.ndarray
    obstacles: np.ndarray
    obstacle_velocities: np.ndarray
    obstacle_radii: np.ndarray
    dynamic: np.ndarray
    target: np.ndarray
    route: np.ndarray
    targets: np.ndarray
    singular: np.ndarray
    obstacle_axes: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.obstacle_axes is None:
            self.obstacle_axes = np.zeros(np.shape(self.obstacles))

    @classmethod
    def of(cls, state: State) -> "World":
        copy = state.copy()
        b, m = copy.positions.shape[:2]
        values = {f.name: getattr(copy, f.name) for f in fields(State)}
        return cls(targets=np.repeat(copy.target[:, None, :], m, axis=1),
                   singular=np.zeros(b, dtype=bool), **values)

    def view(self) -> View:
        return View(self.positions, self.velocities, self.obstacles,
                    self.obstacle_radii, self.targets, self.obstacle_axes)


MOVER_MARGIN = 0.05


def mover_steps(config: FamilyConfig, obstacles: np.ndarray,
                velocities: np.ndarray, radii: np.ndarray,
                dynamic: np.ndarray, drones: np.ndarray) -> np.ndarray:
    dt = config.physics.dt
    step = dt * velocities
    ranges = config.state.dynamic
    standoff = ranges.standoff if ranges is not None else 0.0
    if not standoff or not np.any(dynamic):
        return obstacles + step
    agent = config.physics.agent_radius
    keep = np.maximum(standoff, (radii + agent) * (1.0 + MOVER_MARGIN))
    rel = obstacles[:, :, None, :] - drones[:, None, :, :]
    now = np.sqrt(np.sum(rel * rel, axis=-1))
    bound = np.minimum(keep[:, :, None], now)
    a = np.sum(step * step, axis=-1)[:, :, None]
    b = 2.0 * np.sum(step[:, :, None, :] * rel, axis=-1)
    c = now * now - bound * bound
    disc = b * b - 4.0 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        enter = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
    closing = (a > 0.0) & (b < 0.0) & (disc > 0.0)
    limit = np.where(closing, np.clip(enter, 0.0, 1.0), 1.0).min(axis=2)
    limit = np.where(dynamic[None, :], limit, 1.0)
    return obstacles + limit[..., None] * step


def formation(robots: int, leader: np.ndarray, heading: np.ndarray,
              spacing: float) -> List[np.ndarray]:
    v = unit(heading)
    u = np.stack([-v[:, 1], v[:, 0]], axis=-1)
    root3 = math.sqrt(3.0)
    l = spacing
    d2 = leader - v * l * root3 / 2 + u * l / 2
    d3 = leader - v * l * root3 / 2 - u * l / 2
    d4 = leader - v * l * root3
    d5 = leader - v * l * root3 + u * l
    d6 = leader - v * l * root3 - u * l
    d7 = leader - v * l * root3 * 3 / 2 - u * l / 2
    d8 = leader - v * l * root3 * 3 / 2 + u * l / 2
    d9 = leader - v * l * root3 * 2
    table = {1: [], 2: [d4], 3: [d2, d3], 4: [d2, d3, d4],
             5: [d2, d3, d4, d5], 6: [d2, d3, d4, d5, d6],
             7: [d2, d3, d4, d5, d6, d7], 8: [d2, d3, d4, d5, d6, d7, d8],
             9: [d2, d3, d4, d5, d6, d7, d8, d9]}
    if robots <= 1:
        return []
    return table.get(robots, [d2, d3, d4])


class Simulator:
    def __init__(self, record: Record, config: FamilyConfig):
        self.record = record
        self.config = config
        self.physics = config.physics
        self.controller = Controller(record, config)
        self.leader_formation = (config.physics.targets.kind
                                 == "leader_formation")

    def roles(self, drones: int) -> List[str]:
        if self.leader_formation:
            return ["leader"] + ["follower"] * (drones - 1)
        return ["agent"] * drones


    def _apply(self, world: World, i: int, command: np.ndarray) -> None:
        plant, dt = self.physics.plant, self.physics.dt
        world.singular |= ~np.all(np.isfinite(command), axis=1)
        if plant.integration == "velocity":
            velocity = command
            if plant.max_accel is not None:
                old = world.velocities[:, i]
                accel = (command - old) / dt
                size = norm(accel)[:, None]
                with np.errstate(all="ignore"):
                    limited = old + dt * plant.max_accel * accel / size
                velocity = np.where(size > plant.max_accel, limited, command)
            world.velocities[:, i] = velocity
            world.positions[:, i] = world.positions[:, i] + dt * velocity
        else:
            world.positions[:, i] = world.positions[:, i] + command
            world.velocities[:, i] = command / dt


    def tick(self, world: World,
             terms: Optional[List[Dict[str, np.ndarray]]] = None) -> None:
        physics = self.physics
        world.obstacles = mover_steps(self.config, world.obstacles,
                                      world.obstacle_velocities,
                                      world.obstacle_radii, world.dynamic,
                                      world.positions)
        m = world.positions.shape[1]
        roles = self.roles(m)
        if not self.leader_formation:
            world.targets[:] = world.target[:, None, :]
        else:
            world.targets[:, 0] = world.target

        if physics.plant.update_order == "synchronous":
            view = world.view()
            commands = [self.controller.command(
                i, view, roles[i], None if terms is None else terms[i])
                for i in range(m)]
            for i, command in enumerate(commands):
                self._apply(world, i, command)
        else:
            for i in range(m):
                if self.leader_formation and i == 1:
                    self._slots(world)
                command = self.controller.command(
                    i, world.view(), roles[i],
                    None if terms is None else terms[i])
                self._apply(world, i, command)

        if self.leader_formation:
            self._advance_waypoint(world)

    def _slots(self, world: World) -> None:
        targets = self.physics.targets
        heading = world.target - world.positions[:, 0]
        slots = formation(targets.formation_robots, world.target, heading,
                          targets.formation_spacing)
        m = world.positions.shape[1]
        for i in range(1, m):
            world.targets[:, i] = slots[i - 1]

    def _advance_waypoint(self, world: World) -> None:
        targets = self.physics.targets
        centroid = np.mean(world.positions, axis=1)
        near = norm(world.target - centroid) < targets.waypoint_advance_within
        world.target = np.where(near[:, None],
                                world.target + targets.waypoint_spacing
                                * world.route, world.target)


    def rollout(self, state: State, horizon: int, terms: bool = False
                ) -> Dict[str, object]:
        world = World.of(state)
        out = {"positions": [world.positions.copy()],
               "velocities": [world.velocities.copy()],
               "obstacles": [world.obstacles.copy()],
               "targets": [world.targets.copy()],
               "singular": [world.singular.copy()], "terms": []}
        m = state.drones
        for _ in range(horizon):
            record = [dict() for _ in range(m)] if terms else None
            self.tick(world, record)
            out["positions"].append(world.positions.copy())
            out["velocities"].append(world.velocities.copy())
            out["obstacles"].append(world.obstacles.copy())
            out["targets"].append(world.targets.copy())
            out["singular"].append(world.singular.copy())
            if terms:
                out["terms"].append(record)
        for key in ("positions", "velocities", "obstacles", "targets",
                    "singular"):
            out[key] = np.stack(out[key])
        return out


    def step_reach(self) -> float:
        plant, dt = self.physics.plant, self.physics.dt
        reach = math.inf
        for bound in self.record.global_bounds:
            if bound.kind == "renormalize":
                limit = bound.length
            elif bound.hi is not None:
                limit = bound.hi
            else:
                continue
            reach = min(reach, limit * dt if plant.integration == "velocity"
                        else limit)
        return reach


__all__ = ["MOVER_MARGIN", "Simulator", "World", "formation", "mover_steps"]
