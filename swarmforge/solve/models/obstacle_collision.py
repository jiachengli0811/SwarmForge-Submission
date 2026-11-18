from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..dynamics import World
from ..laws import norm
from ..reach import ObstacleReach
from ..state import State
from .contact import ContactModel, Outcome, Reading

MODEL_NAME = "obstacle_collision"


class ObstacleCollision(ContactModel):
    name = MODEL_NAME
    reach_type = ObstacleReach
    margin_gap = "obstacle_min_gap"
    judge = "face"

    def gaps(self, positions: np.ndarray, obstacles: np.ndarray,
             radii: np.ndarray, axes: Optional[np.ndarray] = None
             ) -> np.ndarray:
        gaps = self.config.physics.body_gap(
            positions[:, :, None, :], obstacles[:, None, :, :],
            radii[:, None, :], None if axes is None else axes[:, None, :, :])
        return gaps.reshape(positions.shape[0], -1)

    def margins(self, world: World, state: State) -> np.ndarray:
        physics = self.config.physics
        gaps = physics.body_gap(world.positions[:, :, None, :],
                                world.obstacles[:, None, :, :],
                                world.obstacle_radii[:, None, :],
                                world.obstacle_axes[:, None, :, :])
        start = state.positions[:, :, None, :] - state.obstacles[:, None, :, :]
        bearing = start / np.maximum(norm(start), 1e-12)[..., None]
        offset = world.positions[:, :, None, :] - world.obstacles[:, None, :, :]
        face = (np.sum(offset * bearing, axis=-1)
                - physics.agent_radius - physics.obstacle_min_gap)
        point = ((world.obstacle_radii == 0.0)
                 & ~np.any(world.obstacle_axes != 0.0, axis=-1))[:, None, :]
        return np.where(point, face, gaps).reshape(world.positions.shape[0],
                                                   -1)

    def pair_members(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        k = state.obstacles.shape[1]
        index = np.arange(state.drones * k)
        return index // k if k else index, index % k if k else index


__all__ = ["MODEL_NAME", "ObstacleCollision", "Outcome", "Reading"]
