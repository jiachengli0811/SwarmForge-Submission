from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

Vec = Tuple[float, float]

BRAKE_RADII = 4.0

BRAKE_FRACTION = 0.15

SLIDE_BAND = 2.5

STALL_SPEED = 0.5

BODY_MARGIN_RATIO = 0.05


def body_margin(*radii: float) -> float:
    return BODY_MARGIN_RATIO * sum(float(r) for r in radii)

DEFAULT_MIN_DISTANCE = 0.0


class Strategy(Enum):
    SPLIT = "S1"
    COMPRESS = "S2"
    HERD = "S3"
    CORRIDOR = "S4"
    SHEAR = "S5"
    BLOCK = "S6"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            key = value.strip().upper()
            for member in cls:
                if member.name == key or member.value == key:
                    return member
        return None


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1])


def _mul(a: Vec, k: float) -> Vec:
    return (a[0] * k, a[1] * k)


def _norm(a: Vec) -> float:
    return math.hypot(a[0], a[1])


def _unit(a: Vec, fallback: Vec = (1.0, 0.0)) -> Vec:
    n = _norm(a)
    return (a[0] / n, a[1] / n) if n > 1e-12 else fallback


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _perp(a: Vec) -> Vec:
    return (-a[1], a[0])


def _rotate(a: Vec, degrees: float) -> Vec:
    r = math.radians(degrees)
    c, s = math.cos(r), math.sin(r)
    return (a[0] * c - a[1] * s, a[0] * s + a[1] * c)


def _centroid(points: Sequence[Vec]) -> Optional[Vec]:
    if not points:
        return None
    n = float(len(points))
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def principal_axis(points: Sequence[Vec], centre: Vec) -> Tuple[Vec, Vec]:
    sxx = syy = sxy = 0.0
    for p in points:
        dx, dy = p[0] - centre[0], p[1] - centre[1]
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    if abs(sxy) < 1e-12 and abs(sxx - syy) < 1e-12:
        return ((1.0, 0.0), (0.0, 1.0))
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    major = (math.cos(theta), math.sin(theta))
    return (major, _perp(major))


@dataclass
class Keepout:
    centre: Vec
    radius: float


def keepouts(position: Vec, *, swarm_positions: Sequence[Vec],
             min_distance: float, obstacles: Sequence[Sequence[float]],
             self_radius: float, agent_radius: float = 0.0) -> List[Keepout]:
    touch = (self_radius + agent_radius
             + body_margin(self_radius, agent_radius))
    gap = max(float(min_distance), touch)
    out = [Keepout(tuple(p), gap) for p in swarm_positions]
    for entry in obstacles or ():
        if len(entry) < 3:
            continue
        out.append(Keepout((float(entry[0]), float(entry[1])),
                           float(entry[2]) + self_radius
                           + body_margin(float(entry[2]), self_radius)))
    return out


def _clip(velocity: Vec, position: Vec, *, limits: Sequence[Keepout],
          speed: float, self_radius: float) -> Vec:
    for limit in limits:
        away = _sub(position, limit.centre)
        distance = _norm(away)
        if distance < limit.radius:
            continue
        normal = _unit(away, (1.0, 0.0)) if distance > 1e-9 else (1.0, 0.0)
        slack = distance - limit.radius
        band = max(SLIDE_BAND * self_radius, BRAKE_FRACTION * limit.radius, 1e-9)
        allowed = speed * min(1.0, slack / band)
        inward = -_dot(velocity, normal)
        if inward > allowed:
            velocity = _add(velocity, _mul(normal, inward - allowed))
    return velocity


def guard(position: Vec, desired: Vec, *, limits: Sequence[Keepout],
          speed: float, self_radius: float) -> Tuple[Vec, bool]:
    velocity = desired
    blocked = False
    escape = (0.0, 0.0)
    worst = 0.0
    stalled_on: Optional[Vec] = None
    stalled_by = 0.0

    for limit in limits:
        away = _sub(position, limit.centre)
        distance = _norm(away)
        normal = _unit(away, (1.0, 0.0)) if distance > 1e-9 else (1.0, 0.0)

        if distance < limit.radius:
            intrusion = limit.radius - distance
            if intrusion > worst:
                worst, escape = intrusion, normal
            blocked = True
            continue

        slack = distance - limit.radius
        band = max(BRAKE_RADII * self_radius,
                   BRAKE_FRACTION * limit.radius, 1e-9)
        if slack < band:
            allowed = speed * (slack / band)
            inward = -_dot(velocity, normal)
            if inward > allowed:
                if inward - allowed > stalled_by:
                    stalled_by, stalled_on = inward - allowed, normal
                velocity = _add(velocity, _mul(normal, inward - allowed))
                blocked = True

        retreat = max(BRAKE_FRACTION * limit.radius, 1e-9)
        if slack < retreat:
            velocity = _add(velocity, _mul(
                normal, 0.5 * speed * (retreat - slack) / retreat))

    if worst > 0.0:
        return (_mul(escape, speed), True)

    magnitude = _norm(velocity)
    if stalled_on is not None and magnitude < STALL_SPEED * speed:
        tangent = _perp(stalled_on)
        if _dot(tangent, desired) < 0.0:
            tangent = _mul(tangent, -1.0)
        velocity = _clip(_add(velocity, _mul(tangent, speed - magnitude)),
                         position, limits=limits, speed=speed,
                         self_radius=self_radius)
        magnitude = _norm(velocity)
    if magnitude > speed:
        velocity = _mul(velocity, speed / magnitude)
    return (velocity, blocked)


SETTLE_PASSES = 6


def settle(position: Vec, *, limits: Sequence[Keepout],
           passes: int = SETTLE_PASSES) -> Tuple[Vec, bool]:
    moved = False
    for _ in range(max(1, int(passes))):
        worst, escape, depth = None, (1.0, 0.0), 0.0
        for limit in limits:
            away = _sub(position, limit.centre)
            distance = _norm(away)
            intrusion = limit.radius - distance
            if intrusion <= 1e-9:
                continue
            if intrusion > depth:
                worst = limit
                depth = intrusion
                escape = (_unit(away, (1.0, 0.0)) if distance > 1e-9
                          else (1.0, 0.0))
        if worst is None:
            return (position, moved)
        position = _add(position, _mul(escape, depth))
        moved = True
    return (position, moved)


@dataclass
class Sighting:
    position: Vec
    centroid: Vec
    heading: Vec
    tick: int
    swarm: Tuple[Vec, ...] = ()
    obstacles: Tuple[Tuple[float, float, float], ...] = ()
    goal: Optional[Vec] = None
    goal_radius: float = 0.0
    drift: Vec = (0.0, 0.0)
    gap: float = 0.0

    @property
    def radius(self) -> float:
        if not self.swarm:
            return 0.0
        return max(math.dist(p, self.centroid) for p in self.swarm)

    def standoff(self, extra: float = 0.0) -> float:
        return self.radius + self.gap + extra

    def nearest_body(self, to: Optional[Vec] = None
                     ) -> Optional[Tuple[float, float, float]]:
        point = to or self.centroid
        best, best_d = None, float("inf")
        for body in self.obstacles:
            d = math.dist(point, (body[0], body[1]))
            if d < best_d:
                best, best_d = body, d
        return best

    def flank(self, side: float) -> Tuple[Vec, ...]:
        across = _perp(self.heading)
        return tuple(p for p in self.swarm
                     if math.copysign(1.0, _dot(_sub(p, self.centroid), across))
                     == math.copysign(1.0, side))


def _split(view: Sighting, params: Dict) -> Vec:
    _major, minor = principal_axis(view.swarm or (view.centroid,), view.centroid)
    approach = _sub(view.position, view.centroid)
    side = math.copysign(1.0, _dot(approach, minor) or 1.0)
    stand = view.gap * float(params.get("approach", 1.0))
    return _add(view.centroid, _mul(minor, -side * stand))


def _compress(view: Sighting, params: Dict) -> Vec:
    bearing = float(params.get("bearing", 180.0))
    offset = _rotate(view.heading, bearing)
    return _add(view.centroid, _mul(offset, view.standoff()))


def _herd(view: Sighting, params: Dict) -> Vec:
    body = view.nearest_body()
    if body is None:
        return _compress(view, params)
    centre = (body[0], body[1])
    if "body_distance" in params:
        target_distance = float(params["body_distance"])
    else:
        target_distance = float(params.get("gap", 0.0)) + body[2]
    closest = min((math.dist(p, centre) for p in view.swarm),
                  default=float("inf"))
    away = _unit(_sub(view.centroid, centre))
    if closest <= target_distance:
        return _add(view.centroid, _mul(away, view.standoff(view.gap)))
    return _add(view.centroid, _mul(away, view.standoff()))


def _corridor(view: Sighting, params: Dict) -> Vec:
    victim = min(view.swarm, key=lambda p: min(
        (math.dist(p, (b[0], b[1])) for b in view.obstacles),
        default=float("inf"))) if view.swarm else view.centroid
    body = view.nearest_body(victim)
    if body is None:
        return _block(view, params)
    centre = (body[0], body[1])
    rival = min((math.dist(centre, (b[0], b[1])) for b in view.obstacles
                 if (b[0], b[1]) != centre), default=float("inf"))
    width = float(params.get("spacing", 0.0)) or (4.0 * max(body[2], 1.0))
    width = min(width, 0.9 * rival) if rival < float("inf") else width
    across = _perp(_unit(_sub(victim, centre), view.heading))
    side = math.copysign(1.0, _dot(_sub(view.centroid, centre), across) or 1.0)
    return _add(centre, _mul(across, side * width))


def _shear(view: Sighting, params: Dict) -> Vec:
    side = float(params.get("side", 1.0))
    flank = view.flank(side)
    target = _centroid(flank) or view.centroid
    across = _perp(view.heading)
    return _add(target, _mul(across, side * (view.gap + 0.5 * view.radius)))


def _block(view: Sighting, params: Dict) -> Vec:
    if view.goal is None:
        return _add(view.centroid, _mul(view.heading, view.gap))
    toward = _unit(_sub(view.goal, view.centroid))
    point = _add(view.centroid, _mul(toward, view.standoff()))
    reach = math.dist(point, view.goal)
    keep = view.goal_radius * float(params.get("goal_clearance", 1.15))
    if reach < keep:
        point = _add(view.goal, _mul(_unit(_sub(point, view.goal), toward), keep))
    return point


DESIRED = {
    Strategy.SPLIT: _split,
    Strategy.COMPRESS: _compress,
    Strategy.HERD: _herd,
    Strategy.CORRIDOR: _corridor,
    Strategy.SHEAR: _shear,
    Strategy.BLOCK: _block,
}


@dataclass
class AttackStage:
    strategy: Strategy = Strategy.COMPRESS
    speed: float = 0.0
    params: Dict = _field(default_factory=dict)
    duration: int = 0


@dataclass
class AttackConfig:
    position: Vec = (0.0, 0.0)
    strategy: Strategy = Strategy.COMPRESS
    speed: float = 0.0
    params: Dict = _field(default_factory=dict)
    stages: List[AttackStage] = _field(default_factory=list)

    blocked: int = 0
    ticks: int = 0

    _last: Optional[Tuple[int, Vec]] = None
    _drift: Vec = (0.0, 0.0)


    def leg(self, tick: int) -> Tuple[Strategy, float, Dict]:
        if not self.stages:
            return (self.strategy, self.speed, self.params)
        elapsed = 0
        for stage in self.stages:
            span = max(0, int(stage.duration))
            if span <= 0 or tick < elapsed + span:
                return (stage.strategy, stage.speed or self.speed,
                        stage.params or self.params)
            elapsed += span
        last = self.stages[-1]
        return (last.strategy, last.speed or self.speed,
                last.params or self.params)


    def step_velocity(self, position: Vec, centroid: Vec, heading: Vec,
                      tick: int, *, goal: Optional[Vec] = None,
                      goal_radius: float = 0.0,
                      swarm_positions: Sequence[Vec] = (),
                      min_distance: float = DEFAULT_MIN_DISTANCE,
                      self_radius: float = 0.0, agent_radius: float = 0.0,
                      obstacles: Sequence[Sequence[float]] = (),
                      **_ignored) -> Vec:
        strategy, speed, params = self.leg(int(tick))
        self_radius = float(self_radius)
        agent_radius = float(agent_radius)
        gap = max(float(min_distance),
                  self_radius + agent_radius
                  + body_margin(self_radius, agent_radius))

        view = Sighting(
            position=tuple(position), centroid=tuple(centroid),
            heading=_unit(tuple(heading)), tick=int(tick),
            swarm=tuple(tuple(p) for p in swarm_positions),
            obstacles=tuple(tuple(float(v) for v in o) for o in obstacles or ()
                            if len(o) >= 3),
            goal=tuple(goal) if goal is not None else None,
            goal_radius=float(goal_radius),
            drift=self._track(int(tick), tuple(centroid)),
            gap=gap)

        want = DESIRED.get(strategy, _compress)(view, params)
        toward = _sub(want, view.position)
        desired = ((0.0, 0.0) if _norm(toward) < 1e-6
                   else _mul(_unit(toward), speed))

        limits = keepouts(view.position, swarm_positions=view.swarm,
                          min_distance=gap, obstacles=view.obstacles,
                          self_radius=self_radius, agent_radius=agent_radius)
        velocity, blocked = guard(view.position, desired, limits=limits,
                                  speed=speed, self_radius=self_radius)
        self.ticks += 1
        self.blocked += 1 if blocked else 0
        return velocity

    def _track(self, tick: int, centroid: Vec) -> Vec:
        if self._last is not None:
            last_tick, last_centroid = self._last
            span = tick - last_tick
            if span > 0:
                step = _sub(centroid, last_centroid)
                self._drift = _mul(step, 1.0 / span)
        self._last = (tick, centroid)
        return self._drift


__all__ = ["Strategy", "AttackStage", "AttackConfig", "Sighting", "Keepout",
           "BODY_MARGIN_RATIO", "SETTLE_PASSES",
           "body_margin", "guard", "keepouts", "principal_axis", "settle"]
