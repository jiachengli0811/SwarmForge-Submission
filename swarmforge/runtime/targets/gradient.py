from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional

from ..environment import EnvRuntime, Environment, Frame
from ..scene import Scene
from .boids import cover_discs

_CANDIDATES = ('target/swarmalgorithms-gradient',)

DEFAULT_OBSTACLE_SIZE = 2.0


def find_source(root: Optional[str] = None) -> str:
    root = root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    for c in _CANDIDATES:
        p = os.path.join(root, c)
        if os.path.isdir(os.path.join(p, "gradient_swarm")):
            return p
    raise RuntimeError("swarmalgorithms-gradient source not found under %s "
                       "(looked in %s)" % (root, ", ".join(_CANDIDATES)))


def load_module(source: Optional[str] = None):
    src = find_source(source)
    if src not in sys.path:
        sys.path.insert(0, src)
    import gradient_swarm as gs
    from gradient_swarm import params as params_mod, swarm as swarm_mod
    return gs, swarm_mod, params_mod


class _Attractor:
    __slots__ = ("position",)

    def __init__(self, position):
        self.position = list(position)


class GradientScene(Scene):
    dims = 2

    def __init__(self, world: "World", params: Dict[str, float]):
        self.world = world
        self._params = params

    def resolve(self, source: str, agent) -> List:
        if source == "agents":
            return [d for d in self.world.swarm.drones if d is not agent]
        if source == "obstacles":
            return self.world.swarm.obstacles
        if source == "target":
            return [self.world.attractor]
        if source == "self":
            return [agent]
        raise ValueError(
            "no source %r in this scene (have: agents, obstacles, target, "
            "self). Returning nothing here is how a term the record declares "
            "came to be silently inert." % source)

    def param(self, name: Optional[str]) -> Optional[float]:
        return self._params.get(name) if name else None

    def position(self, obj):
        p = obj.position
        return (float(p[0]), float(p[1]))

    def velocity(self, obj):
        v = getattr(obj, "vel", None)
        return (float(v[0]), float(v[1])) if v is not None else (0.0, 0.0)

    def clearance(self, agent, other, dist: float) -> float:
        return dist - float(getattr(other, "size", 0.0) or 0.0)

    def centroid(self):
        return self.world.swarm.centroid()


class World:
    def __init__(self, env: Environment, frame: Frame,
                 source: Optional[str] = None,
                 radius: float = DEFAULT_OBSTACLE_SIZE):
        gs, swarm_mod, params_mod = load_module(source)
        self.gs, self.swarm_mod, self.params = gs, swarm_mod, params_mod
        self.env, self.env_rt, self.frame = env, None, frame
        self.mover_parts = []
        self.tick = 0
        self.radius = float(radius)
        self.swarm = gs.Swarm(self.frame.point(env.goal))
        self.dt = float(env.dt)
        self._lay_out(env, int(env.agents), int(env.seed))

        goal = self.swarm.end_loc
        self.attractor = _Attractor((float(goal[0]), float(goal[1])))


    def _lay_out(self, env: Environment, n_agents: int, seed: int) -> None:
        frame = self.frame
        for body in env.static:
            for c, r in cover_discs(body, DEFAULT_OBSTACLE_SIZE):
                self.swarm.add_obstacle(frame.point(c), r,
                                        reach=DEFAULT_OBSTACLE_SIZE)

        self.env_rt = EnvRuntime(env, frame, DEFAULT_OBSTACLE_SIZE,
                                 agent_radius=self.radius)
        for spec, mover in zip(env.dynamic, self.env_rt.movers):
            cx, cy = spec.body.centre()
            for c, r in cover_discs(spec.body, DEFAULT_OBSTACLE_SIZE):
                offset = (c[0] - cx, c[1] - cy)
                obstacle = self.swarm.add_obstacle(
                    frame.point(c), r, reach=DEFAULT_OBSTACLE_SIZE)
                self.mover_parts.append((mover, offset, obstacle))

        for pos in self._spawn(env, n_agents, seed):
            self.swarm.add_drone(self.radius, pos, 1)

    def _spawn(self, env: Environment, n_agents: int, seed: int):
        box = env.start_region
        x0, y0 = self.frame.point((box[0], box[1]))
        x1, y1 = self.frame.point((box[2], box[3]))
        rng = random.Random(int(seed))
        return [(rng.uniform(min(x0, x1), max(x0, x1)),
                 rng.uniform(min(y0, y1), max(y0, y1)))
                for _ in range(n_agents)]


    def step(self, dt: Optional[float] = None) -> None:
        dt = self.dt if dt is None else float(dt)
        self.dt = dt
        if self.env_rt is not None:
            drones = self.swarm.drones
            self.env_rt.step(self.tick,
                             [d.pos for d in drones],
                             [d.vel for d in drones], dt)
            for mover, offset, obstacle in self.mover_parts:
                obstacle.pos = (mover.position[0] + offset[0],
                                mover.position[1] + offset[1])
        self.swarm.update(dt)
        self.tick += 1


def patch(world: World, controller):
    mod = world.swarm_mod
    original = mod.Swarm.update_command
    scene = GradientScene(world, controller.params())

    def update_command(self, drone, index):
        dx, dy = controller.command(drone, scene)
        return (dx, dy)

    mod.Swarm.update_command = update_command
    return lambda: setattr(mod.Swarm, "update_command", original)
