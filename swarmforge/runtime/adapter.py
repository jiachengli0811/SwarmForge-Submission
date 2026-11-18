from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..solve.config import FamilyConfig
from ..solve.record import Record
from . import environment
from .control import SceneLaw
from .targets import adaptive as _adaptive
from .targets import boids as _boids
from .targets import gradient as _gradient
from .targets import vasarhelyi as _vasarhelyi

Vec = Tuple[float, float]


@dataclass(frozen=True)
class TargetProfile:
    key: str
    agent_size: float
    spawn_min_sep: float
    agent_kind: str = "agent"


def profile_for(config: FamilyConfig) -> TargetProfile:
    if config.mission is None:
        raise ValueError("%s declares no mission block" % config.path)
    return TargetProfile(key=config.name,
                         agent_size=config.physics.agent_radius,
                         spawn_min_sep=config.mission.spawn_separation)


def frame_for(config: FamilyConfig) -> environment.Frame:
    return environment.Frame(origin=(config.world.x.lo, config.world.y.lo))


@dataclass
class ObjectState:
    oid: int
    kind: str
    position: Vec
    velocity: Vec = (0.0, 0.0)
    meta: dict = field(default_factory=dict)


class _Body:
    __slots__ = ("id", "position", "velocity", "size", "kind")

    def __init__(self, oid, position, velocity, size, kind):
        self.id = oid
        self.position = [float(position[0]), float(position[1])]
        self.velocity = [float(velocity[0]), float(velocity[1])]
        self.size = float(size)
        self.kind = kind


class _World:
    def __init__(self, goal: Vec, goal_id: int = 0):
        self.agents: List[_Body] = []
        self.obstacles: List[_Body] = []
        self.target = _Body(goal_id, goal, (0.0, 0.0), 0.0, "attractor")
        self.tick = 0


class RealEnvAdapter:
    name = "real"

    def __init__(self, config: FamilyConfig, record: Record,
                 environment_doc: Dict):
        self.config = config
        self.profile = profile_for(config)
        self.record = record
        self.environment_doc = dict(environment_doc)
        self.env = environment.from_dict(self.environment_doc)
        self.world_frame = frame_for(config)
        self.bounds = tuple(float(v) for v in self.env.bounds)
        self.goal = tuple(float(v) for v in self.env.goal)
        self.start_region = tuple(float(v) for v in self.env.start_region)
        self.goal_radius = float(self.env.goal_radius)
        self.spawn_min_sep = float(self.profile.spawn_min_sep)
        self.dt = float(self.env.dt)
        self.controller = SceneLaw(record, config)
        self.world: Optional[_World] = None
        self.world_real = None
        self._agent_of: Dict[int, object] = {}
        self._object_of: Dict[int, object] = {}


    def native_frame(self) -> environment.Frame:
        return self.world_frame


    def _build_world(self, seed: int):
        raise NotImplementedError

    def _release_world(self) -> None:
        raise NotImplementedError

    def _advance(self) -> None:
        raise NotImplementedError

    def _real_agents(self) -> List:
        raise NotImplementedError

    def _real_obstacles(self) -> List:
        raise NotImplementedError

    def _state_of(self, body) -> Tuple[Vec, Vec, float]:
        raise NotImplementedError

    def _obstacle_state(self, body) -> Tuple[Vec, float]:
        raise NotImplementedError

    def _goal_point(self) -> Vec:
        return self.goal

    def _aim(self) -> Tuple[Vec, Vec]:
        return self._goal_point(), (0.0, 0.0)

    def _to_mirror(self, point) -> Vec:
        origin = self.world_frame.origin
        return (float(point[0]) - origin[0], float(point[1]) - origin[1])

    def _from_mirror(self, point) -> Vec:
        origin = self.world_frame.origin
        return (float(point[0]) + origin[0], float(point[1]) + origin[1])

    def _seat_swarm(self, seed: int) -> None:
        pass

    def _explicit_placements(self) -> Optional[List[Dict]]:
        seats = self.environment_doc.get("placements")
        return list(seats) if seats else None


    def reset(self, seed: int = 0) -> None:
        self._release()
        doc = dict(self.environment_doc, seed=int(seed))
        self.env = environment.from_dict(doc)
        self.world_real = self._build_world(int(seed))
        self._seat_swarm(int(seed))

        mirror = _World(self.goal)
        self._agent_of, self._object_of = {}, {}
        oid = 0
        for body in self._real_agents():
            position, velocity, size = self._state_of(body)
            mirror.agents.append(
                _Body(oid, position, velocity, size, self.profile.agent_kind))
            self._agent_of[oid] = body
            self._object_of[oid] = body
            oid += 1
        for obstacle in self._real_obstacles():
            position, size = self._obstacle_state(obstacle)
            mirror.obstacles.append(
                _Body(oid, position, (0.0, 0.0), size, "obstacle"))
            self._object_of[oid] = obstacle
            oid += 1
        mirror.target = _Body(oid, self._goal_point(), (0.0, 0.0), 0.0,
                              "attractor")
        self.world = mirror

    def _release(self) -> None:
        if self.world_real is not None:
            self._release_world()
            self.world_real = None

    def close(self) -> None:
        self._release()

    def __del__(self):
        try:
            self._release()
        except Exception:
            pass


    def _mirror(self) -> None:
        for body in self.world.agents:
            position, velocity, size = self._state_of(self._agent_of[body.id])
            body.position[0], body.position[1] = position
            body.velocity[0], body.velocity[1] = velocity
            body.size = float(size)
        for mirrored, real in zip(self.world.obstacles, self._real_obstacles()):
            position, size = self._obstacle_state(real)
            dt = self.dt or 1.0
            mirrored.velocity[0] = (float(position[0]) - mirrored.position[0]) / dt
            mirrored.velocity[1] = (float(position[1]) - mirrored.position[1]) / dt
            mirrored.position[0], mirrored.position[1] = position
            mirrored.size = float(size)

    def step(self) -> None:
        self._advance()
        self._mirror()
        self.world.tick += 1


    def tick(self) -> int:
        return self.world.tick if self.world else 0

    def agents(self) -> List[ObjectState]:
        return [ObjectState(b.id, self.profile.agent_kind,
                            (b.position[0], b.position[1]),
                            (b.velocity[0], b.velocity[1]))
                for b in self.world.agents]

    def aim(self) -> Tuple[Vec, Vec]:
        return self._aim()

    def movers(self) -> Tuple[Tuple[Vec, float, Vec], ...]:
        runtime = getattr(self.world_real, "env_rt", None)
        if runtime is None:
            return ()
        return tuple((self._to_mirror(m.position), float(m.radius),
                      (float(m.velocity[0]), float(m.velocity[1])))
                     for m in runtime.movers)

    def objects(self) -> List[ObjectState]:
        out: List[ObjectState] = list(self.agents())
        for o in self.world.obstacles:
            out.append(ObjectState(o.id, "obstacle",
                                   (o.position[0], o.position[1]),
                                   (o.velocity[0], o.velocity[1]),
                                   {"size": o.size}))
        t = self.world.target
        out.append(ObjectState(t.id, "attractor",
                               (t.position[0], t.position[1]), (0.0, 0.0)))
        return out


class BoidsEnvAdapter(RealEnvAdapter):
    name = "real:boids"

    def _build_world(self, seed: int):
        world = _boids.World(env=self.env, frame=self.native_frame())
        self._restore = _boids.patch(world, self.controller)
        return world

    def _release_world(self) -> None:
        restore, self._restore = getattr(self, "_restore", None), None
        if restore is not None:
            restore()

    def _advance(self) -> None:
        self.world_real.step(self.dt)

    def _real_agents(self) -> List:
        return list(self.world_real.boids)

    def _real_obstacles(self) -> List:
        return list(self.world_real.obstacles)

    def _state_of(self, body) -> Tuple[Vec, Vec, float]:
        return ((float(body.position[0]), float(body.position[1])),
                (float(body.velocity[0]), float(body.velocity[1])),
                float(body.size))

    def _obstacle_state(self, body) -> Tuple[Vec, float]:
        return ((float(body.position[0]), float(body.position[1])),
                float(body.size))

    def _goal_point(self) -> Vec:
        attractors = self.world_real.attractors
        return (tuple(float(v) for v in attractors[0].position)
                if attractors else self.goal)

    def _seat_swarm(self, seed: int) -> None:
        import random

        world = self.world_real
        seats = self._explicit_placements()
        if seats is not None:
            for body, seat in zip(world.boids, seats):
                px, py = (float(v) for v in seat["position"])
                body.position[0], body.position[1] = px, py
                velocity = seat.get("velocity")
                if velocity is not None:
                    body.velocity[0] = float(velocity[0])
                    body.velocity[1] = float(velocity[1])
            return
        rng = random.Random((int(seed) * 6364136223846793005 + 1) & 0x7FFFFFFF)
        x0, y0, x1, y1 = self.start_region
        gap = float(self.spawn_min_sep)
        obstacles = [(float(o.position[0]), float(o.position[1]), float(o.size))
                     for o in world.obstacles]
        placed: List[Vec] = []
        for body in world.boids:
            size = float(body.size)
            for _try in range(500):
                px, py = rng.uniform(x0, x1), rng.uniform(y0, y1)
                if any(math.dist((px, py), q) < gap for q in placed):
                    continue
                if any(math.dist((px, py), (ox, oy)) < osize + size
                       for ox, oy, osize in obstacles):
                    continue
                break
            else:
                raise ValueError(
                    "could not place %d agents at %.3g separation in the "
                    "%.3gx%.3g start region of %r -- the swarm does not fit "
                    "the box (reduce agents, widen start_region, or lower "
                    "spawn_min_sep)"
                    % (len(world.boids), gap, x1 - x0, y1 - y0,
                       self.environment_doc.get("name", "environment")))
            placed.append((px, py))
            body.position[0], body.position[1] = px, py


class AdaptiveEnvAdapter(RealEnvAdapter):
    name = "real:adaptive"

    def _build_world(self, seed: int):
        return _adaptive.World(controller=self.controller, env=self.env,
                               frame=self.native_frame())

    def _release_world(self) -> None:
        self.world_real.close()

    def _advance(self) -> None:
        self.world_real.step()

    def _real_agents(self) -> List:
        return list(self.world_real.robots)

    def _real_obstacles(self) -> List:
        return list(range(len(self.world_real.map_obstacles)))

    def _state_of(self, body) -> Tuple[Vec, Vec, float]:
        native = (float(body.sp[0]), float(body.sp[1]))
        route = getattr(body, "route", None)
        if route is not None and len(route) > 1:
            velocity = ((native[0] - float(route[-2][0])) / self.dt,
                        (native[1] - float(route[-2][1])) / self.dt)
        else:
            velocity = (0.0, 0.0)
        return (self._to_mirror(native),
                (float(velocity[0]), float(velocity[1])),
                float(self.profile.agent_size))

    def _obstacle_state(self, index) -> Tuple[Vec, float]:
        polygon = self.world_real.map_obstacles[index]
        best, best_gap = None, float("inf")
        for robot in self.world_real.robots:
            point = (float(robot.sp[0]), float(robot.sp[1]))
            surface = _adaptive.surface_of(polygon, point)
            gap = 0.0 if surface.inside else math.dist(surface.position, point)
            if gap < best_gap:
                best, best_gap = surface, gap
        if best is None:
            best = _adaptive.surface_of(polygon, (0.0, 0.0))
        return (self._to_mirror(best.position), 0.0)

    def _goal_point(self) -> Vec:
        return self._to_mirror(self.world_real.xy_goal)

    def _aim(self) -> Tuple[Vec, Vec]:
        world = self.world_real
        trajectory = world.traj_global
        index = min(int(world.sp_ind), len(trajectory) - 1)
        here = (float(trajectory[index][0]), float(trajectory[index][1]))
        ahead = trajectory[min(index + 1, len(trajectory) - 1)]
        dx, dy = float(ahead[0]) - here[0], float(ahead[1]) - here[1]
        span = math.hypot(dx, dy)
        route = (dx / span, dy / span) if span > 0.0 else (0.0, 0.0)
        return self._to_mirror(here), route

    def _seat_swarm(self, seed: int) -> None:
        world = self.world_real
        np, tools = world.np, world.tools
        robots = world.robots
        if len(robots) < 2:
            return

        seats = self._explicit_placements()
        if seats is not None:
            for robot, seat in zip(robots, seats):
                robot.sp = np.array([float(v) for v in seat["position"]],
                                    dtype=float)
                robot.route = np.array([robot.sp])
        else:
            leader = robots[0]
            heading = self._route_heading(leader.sp)
            if heading is None:
                return
            slots = tools.formation(world.params.num_robots, leader.sp,
                                    v=heading,
                                    l=world.params.interrobots_dist)
            for robot, slot in zip(robots[1:], slots):
                robot.sp = np.array(slot, dtype=float)
                robot.route = np.array([robot.sp])

        world.followers_sp = [r.sp for r in robots[1:]]
        world.centroid = np.mean([r.sp for r in robots], axis=0)

    def _route_heading(self, origin):
        world = self.world_real
        np, tools = world.np, world.tools
        span = float(world.params.interrobots_dist)
        trajectory = getattr(world, "traj_global", None)
        if trajectory is not None:
            for point in trajectory:
                delta = np.asarray(point, dtype=float) - np.asarray(origin,
                                                                    dtype=float)
                if float(np.linalg.norm(delta)) >= span:
                    return tools.normalize(delta)
        delta = np.asarray(world.xy_goal, dtype=float) - np.asarray(origin,
                                                                    dtype=float)
        if float(np.linalg.norm(delta)) < 1e-9:
            return None
        return tools.normalize(delta)


class VasarhelyiEnvAdapter(RealEnvAdapter):
    name = "real:vasarhelyi"

    def _build_world(self, seed: int):
        world = _vasarhelyi.World(env=self.env, frame=self.native_frame())
        self._restore = _vasarhelyi.patch(world, self.controller)
        return world

    def _release_world(self) -> None:
        restore, self._restore = getattr(self, "_restore", None), None
        if restore is not None:
            restore()

    def _advance(self) -> None:
        self.world_real.step(self.dt)

    def _real_agents(self) -> List:
        return list(self.world_real.swarm.drones)

    def _real_obstacles(self) -> List:
        return list(self.world_real.obstacles)

    def _state_of(self, body) -> Tuple[Vec, Vec, float]:
        return (self._to_mirror((float(body.pos_ned[0]),
                                 float(body.pos_ned[1]))),
                (float(body.vel_xyz[0]), float(body.vel_xyz[1])),
                float(self.profile.agent_size))

    def _obstacle_state(self, body) -> Tuple[Vec, float]:
        return (self._to_mirror((float(body.position[0]),
                                 float(body.position[1]))),
                float(body.size))

    def _goal_point(self) -> Vec:
        goal = self.world_real.p_swarm.x_goal
        return self._to_mirror((float(goal[0]), float(goal[1])))

    def _seat_swarm(self, seed: int) -> None:
        import random

        world = self.world_real
        drones = world.swarm.drones
        altitude = float(world.altitude)

        seats = self._explicit_placements()
        if seats is not None:
            for drone, seat in zip(drones, seats):
                px, py = self._from_mirror([float(v)
                                            for v in seat["position"]])
                drone.pos_ned[0], drone.pos_ned[1] = px, py
                drone.pos_ned[2] = altitude
                velocity = seat.get("velocity")
                if velocity is not None:
                    drone.vel_xyz[0] = float(velocity[0])
                    drone.vel_xyz[1] = float(velocity[1])
                    drone.command[1] = float(velocity[0])
                    drone.command[2] = float(velocity[1])
            return

        rng = random.Random((int(seed) * 6364136223846793005 + 1) & 0x7FFFFFFF)
        x0, y0 = self._from_mirror(self.start_region[:2])
        x1, y1 = self._from_mirror(self.start_region[2:])
        gap = float(self.spawn_min_sep)
        obstacles = [(float(c[0]), float(c[1]), float(c[2]))
                     for c in world.p_swarm.cylinders.T]
        size = float(self.profile.agent_size)
        placed: List[Vec] = []
        for drone in drones:
            for _try in range(500):
                px, py = rng.uniform(x0, x1), rng.uniform(y0, y1)
                if any(math.dist((px, py), q) < gap for q in placed):
                    continue
                if any(math.dist((px, py), (ox, oy)) < osize + size
                       for ox, oy, osize in obstacles):
                    continue
                break
            else:
                raise ValueError(
                    "could not place %d agents at %.3g separation in the "
                    "%.3gx%.3g start region of %r -- the swarm does not fit "
                    "the box (reduce agents, widen start_region, or lower "
                    "spawn_min_sep)"
                    % (len(drones), gap, x1 - x0, y1 - y0,
                       self.environment_doc.get("name", "environment")))
            placed.append((px, py))
            drone.pos_ned[0], drone.pos_ned[1] = px, py
            drone.pos_ned[2] = altitude


class GradientEnvAdapter(RealEnvAdapter):
    name = "real:gradient"

    _motion: Optional[Tuple[Vec, Vec, float]] = None

    def _build_world(self, seed: int):
        world = _gradient.World(env=self.env, frame=self.native_frame())
        self._restore = _gradient.patch(world, self.controller)
        self._motion = self._moving_goal()
        if self._motion is not None:
            self._place_goal(world, 0)
        return world

    def _release_world(self) -> None:
        restore, self._restore = getattr(self, "_restore", None), None
        if restore is not None:
            restore()

    def _moving_goal(self) -> Optional[Tuple[Vec, Vec, float]]:
        block = self.environment_doc.get("moving_goal")
        if not block:
            return None
        start = (float(block["start"][0]), float(block["start"][1]))
        return start, self.goal, float(block["speed"])

    def _goal_at(self, tick: int) -> Vec:
        start, end, speed = self._motion
        span = math.dist(start, end)
        travelled = speed * self.dt * tick
        if travelled >= span:
            return end
        share = travelled / span
        return (start[0] + share * (end[0] - start[0]),
                start[1] + share * (end[1] - start[1]))

    def _place_goal(self, world, tick: int) -> None:
        x, y = self._from_mirror(self._goal_at(tick))
        world.swarm.end_loc = (x, y)
        world.attractor.position[0], world.attractor.position[1] = x, y

    def _advance(self) -> None:
        if self._motion is not None:
            self._place_goal(self.world_real, self.world_real.tick + 1)
        self.world_real.step(self.dt)

    def _mirror(self) -> None:
        super()._mirror()
        if self._motion is not None:
            target = self.world.target
            target.position[0], target.position[1] = self._goal_point()

    def _aim(self) -> Tuple[Vec, Vec]:
        if self._motion is None:
            return self._goal_point(), (0.0, 0.0)
        return self._goal_at(self.world_real.tick + 1), (0.0, 0.0)

    def _real_agents(self) -> List:
        return list(self.world_real.swarm.drones)

    def _real_obstacles(self) -> List:
        return list(self.world_real.swarm.obstacles)

    def _state_of(self, body) -> Tuple[Vec, Vec, float]:
        return (self._to_mirror((float(body.pos[0]), float(body.pos[1]))),
                (float(body.vel[0]), float(body.vel[1])),
                float(self.profile.agent_size))

    def _obstacle_state(self, body) -> Tuple[Vec, float]:
        return (self._to_mirror((float(body.pos[0]), float(body.pos[1]))),
                float(body.radius))

    def _goal_point(self) -> Vec:
        goal = self.world_real.attractor.position
        return self._to_mirror((float(goal[0]), float(goal[1])))

    def _seat_swarm(self, seed: int) -> None:
        import random

        world = self.world_real
        drones = world.swarm.drones

        seats = self._explicit_placements()
        if seats is not None:
            for drone, seat in zip(drones, seats):
                drone.pos = self._from_mirror([float(v)
                                               for v in seat["position"]])
                velocity = seat.get("velocity")
                if velocity is not None:
                    drone.vel = (float(velocity[0]), float(velocity[1]))
            return

        rng = random.Random((int(seed) * 6364136223846793005 + 1) & 0x7FFFFFFF)
        x0, y0 = self._from_mirror(self.start_region[:2])
        x1, y1 = self._from_mirror(self.start_region[2:])
        gap = float(self.spawn_min_sep)
        obstacles = [(float(o.pos[0]), float(o.pos[1]), float(o.radius))
                     for o in world.swarm.obstacles]
        size = float(self.profile.agent_size)
        placed: List[Vec] = []
        for drone in drones:
            for _try in range(500):
                px, py = rng.uniform(x0, x1), rng.uniform(y0, y1)
                if any(math.dist((px, py), q) < gap for q in placed):
                    continue
                if any(math.dist((px, py), (ox, oy)) < osize + size
                       for ox, oy, osize in obstacles):
                    continue
                break
            else:
                raise ValueError(
                    "could not place %d agents at %.3g separation in the "
                    "%.3gx%.3g start region of %r -- the swarm does not fit "
                    "the box (reduce agents, widen start_region, or lower "
                    "spawn_min_sep)"
                    % (len(drones), gap, x1 - x0, y1 - y0,
                       self.environment_doc.get("name", "environment")))
            placed.append((px, py))
            drone.pos = (px, py)


ADAPTERS = {"boids": BoidsEnvAdapter, "adaptive-swarm": AdaptiveEnvAdapter,
            "vasarhelyi": VasarhelyiEnvAdapter, "gradient": GradientEnvAdapter}


def real_env_adapter(config: FamilyConfig, record: Record,
                     environment_doc: Dict) -> RealEnvAdapter:
    if config.name not in ADAPTERS:
        raise ValueError(
            "no real-code adapter for %r (have: %s)"
            % (config.name, ", ".join(sorted(ADAPTERS))))
    return ADAPTERS[config.name](config, record, environment_doc)


__all__ = ["ADAPTERS", "AdaptiveEnvAdapter", "BoidsEnvAdapter",
           "GradientEnvAdapter", "ObjectState", "RealEnvAdapter",
           "TargetProfile", "VasarhelyiEnvAdapter", "WorldFrame", "frame_for",
           "profile_for", "real_env_adapter"]
