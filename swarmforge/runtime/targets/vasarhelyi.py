from __future__ import annotations

import math
import os
import random
import sys
from typing import Dict, List, Optional

from ..environment import EnvRuntime, Environment, Frame
from ..scene import Scene
from .boids import cover_discs

_CANDIDATES = ('target/swarmlab-vasarhelyi',)

DEFAULT_ALTITUDE = -50.0

DEFAULT_OBSTACLE_SIZE = 18.75


def find_source(root: Optional[str] = None) -> str:
    root = root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    for c in _CANDIDATES:
        p = os.path.join(root, c)
        if os.path.isdir(os.path.join(p, "vasarhelyi")):
            return p
    raise RuntimeError("swarmlab-vasarhelyi source not found under %s "
                       "(looked in %s)" % (root, ", ".join(_CANDIDATES)))


def load_module(source: Optional[str] = None):
    src = find_source(source)
    if src not in sys.path:
        sys.path.insert(0, src)
    import vasarhelyi as vh
    from vasarhelyi import control as control_mod, swarm as swarm_mod
    return vh, swarm_mod, control_mod


class _Cylinder:
    __slots__ = ("cylinders", "column")

    def __init__(self, cylinders, column: int):
        self.cylinders, self.column = cylinders, column

    @property
    def position(self):
        return self.cylinders[0:2, self.column]

    @position.setter
    def position(self, value):
        self.cylinders[0, self.column] = float(value[0])
        self.cylinders[1, self.column] = float(value[1])

    @property
    def size(self) -> float:
        return float(self.cylinders[2, self.column])


class _Attractor:
    __slots__ = ("position",)

    def __init__(self, position):
        self.position = list(position)


class VasarhelyiScene(Scene):
    dims = 2

    def __init__(self, world: "World", params: Dict[str, float]):
        self.world = world
        self._params = params

    def resolve(self, source: str, agent) -> List:
        if source == "agents":
            return [d for d in self.world.swarm.drones if d is not agent]
        if source == "obstacles":
            return self.world.obstacles
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
        p = getattr(obj, "pos_ned", None)
        return (float(p[0]), float(p[1])) if p is not None else \
            (float(obj.position[0]), float(obj.position[1]))

    def velocity(self, obj):
        v = getattr(obj, "vel_xyz", None)
        if v is not None:
            return (float(v[0]), float(v[1]))
        v = getattr(obj, "velocity", None)
        return (float(v[0]), float(v[1])) if v is not None else (0.0, 0.0)

    def clearance(self, agent, other, dist: float) -> float:
        return dist - float(getattr(other, "size", 0.0) or 0.0)

    def centroid(self):
        drones = self.world.swarm.drones
        n = len(drones) or 1
        return (sum(float(d.pos_ned[0]) for d in drones) / n,
                sum(float(d.pos_ned[1]) for d in drones) / n)


class World:
    def __init__(self, env: Environment, frame: Frame,
                 source: Optional[str] = None,
                 altitude: float = DEFAULT_ALTITUDE, **kw):
        import numpy as np

        vh, swarm_mod, control_mod = load_module(source)
        self.np, self.vh, self.swarm_mod = np, vh, swarm_mod
        self.altitude = float(altitude)
        self.env, self.env_rt, self.frame = env, None, frame
        self.mover_parts = []
        self.tick = 0

        p_sim = vh.SimParams()
        p_sim.dt = float(env.dt)
        n_agents = int(env.agents)
        seed = int(env.seed)
        self.p_sim = p_sim
        self.dt = p_sim.dt

        p_swarm = vh.param_swarm(None, True, nb_agents=n_agents, **kw)
        self.map_params = None
        self.p_swarm = p_swarm
        self._lay_out(env)

        self.swarm = vh.Swarm()
        self.swarm.algorithm = "vasarhelyi"
        for _ in range(p_swarm.nb_agents):
            self.swarm.add_drone("point_mass", p_sim)
        self.swarm.set_pos(self._spawn(seed))

        self.obstacles = [_Cylinder(p_swarm.cylinders, j)
                          for j in range(p_swarm.n_cyl)]
        goal = p_swarm.x_goal if p_swarm.is_active_goal else \
            p_swarm.Pos0.mean(axis=1) + 1e3 * p_swarm.u_ref
        self.attractor = _Attractor((float(goal[0]), float(goal[1])))


    def _lay_out(self, env: Environment) -> None:
        np, frame = self.np, self.frame
        discs = [(frame.point(c), r)
                 for b in env.static
                 for c, r in cover_discs(b, DEFAULT_OBSTACLE_SIZE)]

        self.env_rt = EnvRuntime(env, frame, DEFAULT_OBSTACLE_SIZE,
                                 agent_radius=self.p_swarm.r_coll)
        for spec, mover in zip(env.dynamic, self.env_rt.movers):
            cx, cy = spec.body.centre()
            for c, r in cover_discs(spec.body, DEFAULT_OBSTACLE_SIZE):
                offset = (c[0] - cx, c[1] - cy)
                self.mover_parts.append((mover, offset, len(discs)))
                discs.append((frame.point(c), r))

        p = self.p_swarm
        p.cylinders = np.array([[c[0] for c, _ in discs],
                                [c[1] for c, _ in discs],
                                [r for _, r in discs]], dtype=float) \
            if discs else np.zeros((3, 0))
        p.n_cyl = len(discs)
        p.is_active_cyl = True
        p.is_active_arena = False
        p.is_active_spheres = False

        goal = frame.point(env.goal)
        p.is_active_migration = False
        p.is_active_goal = True
        p.x_goal = np.array([goal[0], goal[1], self.altitude])

    def _spawn(self, seed: int):
        np, p = self.np, self.p_swarm
        box = self.env.start_region
        x0, y0 = self.frame.point((box[0], box[1]))
        x1, y1 = self.frame.point((box[2], box[3]))
        rng = random.Random(int(seed))
        pos = np.zeros((3, p.nb_agents))
        for i in range(p.nb_agents):
            pos[0, i] = rng.uniform(min(x0, x1), max(x0, x1))
            pos[1, i] = rng.uniform(min(y0, y1), max(y0, y1))
            pos[2, i] = self.altitude
        return pos


    def step(self, dt: Optional[float] = None) -> None:
        dt = self.dt if dt is None else float(dt)
        self.p_sim.dt = self.dt = dt
        if self.env_rt is not None:
            drones = self.swarm.drones
            self.env_rt.step(self.tick,
                             [(d.pos_ned[0], d.pos_ned[1]) for d in drones],
                             [(d.vel_xyz[0], d.vel_xyz[1]) for d in drones], dt)
            for mover, offset, column in self.mover_parts:
                self.p_swarm.cylinders[0, column] = mover.position[0] + offset[0]
                self.p_swarm.cylinders[1, column] = mover.position[1] + offset[1]
        self.swarm.update_command(self.p_swarm, self.p_swarm.r_coll, dt)
        self.swarm.update_state(None, self.tick * dt)
        self.tick += 1


def patch(world: World, controller):
    np = world.np
    mod = world.swarm_mod
    original = mod.Swarm.update_command
    scene = VasarhelyiScene(world, controller.params())

    def update_command(self, p_swarm, r_coll, dt, rng=None):
        commands = np.zeros((3, self.nb_agents))
        max_a = getattr(p_swarm, "max_a", None)
        for i, drone in enumerate(self.drones):
            vx, vy = controller.command(drone, scene)
            if max_a is not None:
                vel = drone.vel_xyz
                ax, ay = (vx - float(vel[0])) / dt, (vy - float(vel[1])) / dt
                norm = math.sqrt(ax * ax + ay * ay)
                if norm > max_a:
                    vx = float(vel[0]) + dt * max_a * ax / norm
                    vy = float(vel[1]) + dt * max_a * ay / norm
            commands[0, i], commands[1, i] = vx, vy
        self.set_vel_commands(commands)
        return commands, None

    mod.Swarm.update_command = update_command
    return lambda: setattr(mod.Swarm, "update_command", original)
