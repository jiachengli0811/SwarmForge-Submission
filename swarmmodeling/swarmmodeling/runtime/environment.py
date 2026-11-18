from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Vec = Tuple[float, float]
Region = Tuple[float, float, float, float]


@dataclass
class Body:
    position: Vec = (0.0, 0.0)
    radius: Optional[float] = None
    polygon: Optional[List[Vec]] = None

    def as_polygon(self, default_radius: float) -> List[Vec]:
        if self.polygon:
            return [tuple(p) for p in self.polygon]
        r = self.radius if self.radius is not None else default_radius
        x, y = self.position
        return [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]

    def size(self, default_radius: float) -> float:
        if self.radius is not None:
            return self.radius
        if self.polygon:
            x, y = self.centre()
            return max(math.hypot(p[0] - x, p[1] - y) for p in self.polygon)
        return default_radius

    def centre(self) -> Vec:
        if self.polygon:
            n = len(self.polygon)
            return (sum(p[0] for p in self.polygon) / n,
                    sum(p[1] for p in self.polygon) / n)
        return tuple(self.position)


@dataclass
class Motion:
    kind: str = "static"
    speed: float = 0.0
    strategy: str = "S2"
    params: Dict = field(default_factory=dict)
    stages: List[Dict] = field(default_factory=list)


@dataclass
class DynamicObstacle:
    id: str = "obstacle"
    body: Body = field(default_factory=Body)
    motion: Motion = field(default_factory=Motion)


@dataclass
class Environment:
    name: str = "environment"
    description: str = ""
    bounds: Vec = (1000.0, 1000.0)
    start_region: Region = (80.0, 80.0, 220.0, 220.0)
    goal: Vec = (850.0, 850.0)
    goal_radius: float = 50.0
    static: List[Body] = field(default_factory=list)
    dynamic: List[DynamicObstacle] = field(default_factory=list)
    agents: int = 20
    ticks: int = 600
    dt: float = 0.05
    seed: int = 0

    def start_centre(self) -> Vec:
        x0, y0, x1, y1 = self.start_region
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _body(d) -> Body:
    if d is None:
        return Body()
    if isinstance(d, (list, tuple)):
        return Body(position=(d[0], d[1]),
                    radius=d[2] if len(d) > 2 else None)
    b = Body()
    for k, v in d.items():
        if k in ("position", "radius", "polygon"):
            setattr(b, k, tuple(v) if k == "position" else v)
    return b


def from_dict(d: Dict) -> Environment:
    env = Environment()
    for k, v in d.items():
        if k == "static":
            env.static = [_body(x) for x in v]
        elif k == "dynamic":
            env.dynamic = [
                DynamicObstacle(id=x.get("id", "obstacle_%d" % i),
                                body=_body(x.get("body", x)),
                                motion=Motion(**x.get("motion", {})))
                for i, x in enumerate(v)]
        elif k in ("bounds", "goal", "start_region"):
            setattr(env, k, tuple(v))
        elif hasattr(env, k):
            setattr(env, k, v)
    return env


def load(path: str) -> Environment:
    with open(path) as f:
        return from_dict(json.load(f))


def save(env: Environment, path: str) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(asdict(env), indent=2))


@dataclass
class Frame:
    origin: Vec = (0.0, 0.0)
    scale: float = 1.0

    @classmethod
    def from_world(cls, map_per_native: float, origin: Vec = (0.0, 0.0)
                   ) -> "Frame":
        if not map_per_native:
            raise ValueError("world scale must be non-zero")
        return cls(origin=(float(origin[0]), float(origin[1])),
                   scale=1.0 / float(map_per_native))

    @classmethod
    def to_box(cls, bounds: Vec, lo: Vec, hi: Vec) -> "Frame":
        s = min((hi[0] - lo[0]) / bounds[0], (hi[1] - lo[1]) / bounds[1])
        return cls(origin=(lo[0], lo[1]), scale=s)

    def point(self, p: Sequence[float]) -> Vec:
        return (self.origin[0] + self.scale * p[0],
                self.origin[1] + self.scale * p[1])

    def length(self, v: float) -> float:
        return self.scale * v

    def vector(self, v: Sequence[float]) -> Vec:
        return (self.scale * v[0], self.scale * v[1])

    def polygon(self, poly) -> List[Vec]:
        return [self.point(p) for p in poly]


def _unit(v: Vec) -> Vec:
    m = math.hypot(v[0], v[1])
    return (v[0] / m, v[1] / m) if m else (0.0, 0.0)


SUBSTEP_TRAVEL = 1.0

MAX_SUBSTEPS = 16


def _scaled_params(params: Dict, frame: Frame) -> Dict:
    lengths = getattr(_load_attack_module(), "LENGTH_PARAMS", frozenset())
    return {k: (frame.length(float(v)) if k in lengths else v)
            for k, v in (params or {}).items()}


class _Mover:
    def __init__(self, spec: DynamicObstacle, frame: Frame, default_radius: float):
        self.id = spec.id
        self.motion = spec.motion
        self.body = spec.body
        self.position = frame.point(spec.body.centre())
        self.radius = frame.length(spec.body.size(default_radius / frame.scale
                                                  if frame.scale else default_radius))
        if spec.body.radius is not None:
            self.radius = frame.length(spec.body.radius)
        self.velocity = (0.0, 0.0)
        self.speed = frame.length(spec.motion.speed)
        self.params: Dict = {}
        self.others: Tuple[Tuple[float, float, float], ...] = ()
        self.settled = 0
        self._attack = None
        if spec.motion.kind == "strategy":
            self.params = _scaled_params(spec.motion.params, frame)
            self._attack = _build_attack(spec.motion, self.position,
                                         self.speed, frame)

    def step(self, tick: int, dt: float, swarm: "SwarmState",
             others: Sequence[Tuple[float, float, float]] = ()) -> None:
        if self.motion.kind == "static":
            return
        self.others = tuple(others)
        n = self._substeps(dt)
        h = dt / n
        for _ in range(n):
            self.velocity = self._velocity(tick, swarm)
            self.position = (self.position[0] + self.velocity[0] * h,
                             self.position[1] + self.velocity[1] * h)
            self.position, moved = self._settle(swarm)
            self.settled += 1 if moved else 0

    def _substeps(self, dt: float) -> int:
        if self.speed <= 0:
            return 1
        reach = self.speed * dt
        room = self.radius * SUBSTEP_TRAVEL
        if room <= 0 or reach <= room:
            return 1
        return min(MAX_SUBSTEPS, int(math.ceil(reach / room)))

    def _velocity(self, tick: int, swarm: "SwarmState") -> Vec:
        k = self.motion.kind
        if k == "strategy":
            return self._strategy_velocity(tick, swarm)
        raise ValueError("unknown motion kind %r (have: static, strategy). "
                         "The fixed-path kinds were removed: a mover on a path "
                         "chosen before the swarm existed tests where the path "
                         "went, not what the controller did" % k)

    def _blockers(self, swarm: "SwarmState"
                  ) -> Tuple[Tuple[float, float, float], ...]:
        return tuple(swarm.static_discs) + tuple(self.others)

    def _settle(self, swarm: "SwarmState") -> Tuple[Vec, bool]:
        attack = _load_attack_module()
        limits = attack.keepouts(
            self.position, swarm_positions=(), min_distance=0.0,
            obstacles=self._blockers(swarm), self_radius=self.radius,
            agent_radius=swarm.agent_radius)
        return attack.settle(self.position, limits=limits)

    def _strategy_velocity(self, tick: int, swarm: "SwarmState") -> Vec:
        return self._attack.step_velocity(
            self.position, swarm.centroid, swarm.heading, tick,
            goal=swarm.goal, goal_radius=swarm.goal_radius,
            swarm_positions=swarm.positions,
            min_distance=float(self.params.get("min_distance", 0.0)),
            self_radius=self.radius, agent_radius=swarm.agent_radius,
            obstacles=self._blockers(swarm))


def _build_attack(motion: Motion, position: Vec, speed: float, frame: Frame):
    attack = _load_attack_module()
    stages = [attack.AttackStage(
                  strategy=attack.Strategy(s.get("strategy", "S2")),
                  speed=(frame.length(float(s["speed"])) if s.get("speed")
                         else speed),
                  params=_scaled_params(s.get("params", {}), frame),
                  duration=int(s.get("duration", 0)))
              for s in motion.stages]
    return attack.AttackConfig(position=position,
                               strategy=attack.Strategy(motion.strategy),
                               speed=speed,
                               params=_scaled_params(motion.params, frame),
                               stages=stages)


_ATTACK_MODULE = None


def set_attack_module(module) -> None:
    global _ATTACK_MODULE
    _ATTACK_MODULE = module


def _load_attack_module():
    if _ATTACK_MODULE is not None:
        return _ATTACK_MODULE
    try:
        from swarmforge.testing import attack
        return attack
    except ImportError:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "..", "..", ".."))
        for p in (root, os.path.dirname(root)):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from swarmforge.testing import attack
            return attack
        except ImportError:
            raise ImportError(
                "motion kind 'strategy' needs an attack module: call "
                "environment.set_attack_module(...) (swarmforge.runtime does this "
                "for you), or put swarmforge.testing.attack on the path. It is "
                "the only kind that moves, so a case with any mover needs one")


@dataclass
class SwarmState:
    centroid: Vec = (0.0, 0.0)
    heading: Vec = (1.0, 0.0)
    positions: List[Vec] = field(default_factory=list)
    goal: Optional[Vec] = None
    goal_radius: float = 0.0
    agent_radius: float = 0.0
    static_discs: List[Tuple[float, float, float]] = field(default_factory=list)

    @classmethod
    def of(cls, positions, velocities, **kw) -> "SwarmState":
        n = len(positions) or 1
        cx = sum(p[0] for p in positions) / n
        cy = sum(p[1] for p in positions) / n
        hx = sum(v[0] for v in velocities) / n if velocities else 1.0
        hy = sum(v[1] for v in velocities) / n if velocities else 0.0
        h = _unit((hx, hy)) or (1.0, 0.0)
        return cls(centroid=(cx, cy), heading=h if any(h) else (1.0, 0.0),
                   positions=[tuple(p) for p in positions], **kw)


class EnvRuntime:
    def __init__(self, env: Environment, frame: Frame, default_radius: float,
                 agent_radius: float = 0.0):
        self.env, self.frame = env, frame
        self.default_radius = default_radius
        self.agent_radius = agent_radius
        self.movers = [_Mover(d, frame, default_radius) for d in env.dynamic]
        self.goal = frame.point(env.goal)
        self.goal_radius = frame.length(env.goal_radius)
        self.static_discs = [
            (frame.point(b.centre())[0], frame.point(b.centre())[1],
             frame.length(b.size(default_radius / frame.scale if frame.scale
                                 else default_radius)))
            for b in env.static]

    def step(self, tick: int, positions, velocities, dt: float) -> None:
        if not self.movers:
            return
        swarm = SwarmState.of(positions, velocities, goal=self.goal,
                              goal_radius=self.goal_radius,
                              agent_radius=self.agent_radius,
                              static_discs=self.static_discs)
        discs = [(m.position[0], m.position[1], m.radius) for m in self.movers]
        for i, m in enumerate(self.movers):
            m.step(tick, dt, swarm,
                   others=[d for j, d in enumerate(discs) if j != i])
