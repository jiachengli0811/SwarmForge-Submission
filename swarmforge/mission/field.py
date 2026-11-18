from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..solve.config import FamilyConfig

Vec = Tuple[float, float]
Region = Tuple[float, float, float, float]
Field = Dict[str, Any]
Seat = Tuple[Vec, float, Dict]

START_FRACTION: Region = (0.08, 0.08, 0.22, 0.22)
GOAL_FRACTION: Vec = (0.85, 0.85)

GOAL_CLEARANCE = 0.09
START_CLEARANCE = 0.04

MOTION_KINDS = ("static", "strategy")
STRATEGIES = ("S1", "S2", "S3", "S4", "S5", "S6")

RADIUS_RATIO = (0.7, 1.0)
STATIC_COUNT_RATIO = (0.2, 1.0)
MOVER_RADIUS_RATIO = (0.8, 1.0)
WALL_THICKNESS_RATIO = (0.92, 1.85)
WALL_LENGTH_RATIO = (3.5, 12.3)
MOVER_SPEED_BAND = (0.4, 1.0)
STANDOFF_BAND = (0.02, 0.06)
_MISS_BUDGET = 12
SPAWN_MARGIN = 1.25


def start_region_for(bounds: Vec, *, agents: int = 0,
                     agent_boundary: float = 0.0) -> Region:
    box = (START_FRACTION[0] * bounds[0], START_FRACTION[1] * bounds[1],
           START_FRACTION[2] * bounds[0], START_FRACTION[3] * bounds[1])
    if agents < 2 or agent_boundary <= 0.0:
        return box
    side = SPAWN_MARGIN * agent_boundary * agents
    if side <= box[2] - box[0]:
        return box
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    half = 0.5 * min(side, 0.6 * min(bounds))
    cx = min(max(cx, half), bounds[0] - half)
    cy = min(max(cy, half), bounds[1] - half)
    return (cx - half, cy - half, cx + half, cy + half)


def goal_for(bounds: Vec) -> Vec:
    return (GOAL_FRACTION[0] * bounds[0], GOAL_FRACTION[1] * bounds[1])


def body_centre(body: Dict) -> Vec:
    poly = body.get("polygon")
    if poly:
        n = len(poly)
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)
    x, y = body.get("position", (0.0, 0.0))
    return (float(x), float(y))


def body_extent(body: Dict, radius: float = 0.0) -> float:
    poly = body.get("polygon")
    if poly:
        cx, cy = body_centre(body)
        return max(math.hypot(p[0] - cx, p[1] - cy) for p in poly)
    value = body.get("radius")
    return float(radius if value is None else value)


def move_body(body: Dict, to: Vec) -> Dict:
    cx, cy = body_centre(body)
    dx, dy = to[0] - cx, to[1] - cy
    if body.get("polygon"):
        return dict(body, polygon=[[p[0] + dx, p[1] + dy]
                                   for p in body["polygon"]])
    return dict(body, position=[to[0], to[1]])


def rectangle(centre: Vec, length: float, thickness: float,
              angle: float) -> List[List[float]]:
    ca, sa = math.cos(angle), math.sin(angle)
    hl, ht = length / 2.0, thickness / 2.0
    return [[centre[0] + ca * sx * hl - sa * sy * ht,
             centre[1] + sa * sx * hl + ca * sy * ht]
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def keepout(centre: Vec, extent: float, *, start_region: Region, goal: Vec,
            bounds: Vec, goal_radius: float = 0.0) -> bool:
    side = min(bounds)
    start_clearance = START_CLEARANCE * side
    x0, y0, x1, y1 = start_region
    if (x0 - start_clearance - extent <= centre[0]
            <= x1 + start_clearance + extent
            and y0 - start_clearance - extent <= centre[1]
            <= y1 + start_clearance + extent):
        return True
    return (math.dist(centre, goal)
            < max(GOAL_CLEARANCE * side, goal_radius) + extent)


def inside_bounds(centre: Vec, extent: float, bounds: Vec) -> bool:
    return (extent <= centre[0] <= bounds[0] - extent
            and extent <= centre[1] <= bounds[1] - extent)


def _edges(poly: Sequence[Sequence[float]]) -> List[Tuple[Vec, Vec]]:
    return [((poly[i][0], poly[i][1]),
             (poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][1]))
            for i in range(len(poly))]


def _point_segment(p: Vec, a: Vec, b: Vec) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    span = dx * dx + dy * dy
    t = (0.0 if span == 0.0 else
         max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / span)))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _turn(o: Vec, a: Vec, b: Vec) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segment_segment(a1: Vec, a2: Vec, b1: Vec, b2: Vec) -> float:
    if ((_turn(b1, b2, a1) > 0.0) != (_turn(b1, b2, a2) > 0.0)
            and (_turn(a1, a2, b1) > 0.0) != (_turn(a1, a2, b2) > 0.0)):
        return 0.0
    return min(_point_segment(a1, b1, b2), _point_segment(a2, b1, b2),
               _point_segment(b1, a1, a2), _point_segment(b2, a1, a2))


def _inside(p: Vec, poly: Sequence[Sequence[float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        (xi, yi), (xj, yj) = poly[i], poly[j]
        if (yi > p[1]) != (yj > p[1]):
            if p[0] < xi + (p[1] - yi) * (xj - xi) / (yj - yi):
                inside = not inside
        j = i
    return inside


def _overlap_depth(pa: Sequence[Sequence[float]],
                   pb: Sequence[Sequence[float]]) -> float:
    depth = math.inf
    for poly in (pa, pb):
        for a, b in _edges(poly):
            nx, ny = a[1] - b[1], b[0] - a[0]
            norm = math.hypot(nx, ny)
            if norm == 0.0:
                continue
            nx, ny = nx / norm, ny / norm
            onto_a = [nx * p[0] + ny * p[1] for p in pa]
            onto_b = [nx * p[0] + ny * p[1] for p in pb]
            depth = min(depth, max(onto_a) - min(onto_b),
                        max(onto_b) - min(onto_a))
    return max(0.0, depth)


def surface_gap(a: Dict, b: Dict) -> float:
    pa, pb = a.get("polygon"), b.get("polygon")
    if not pa and not pb:
        return (math.dist(body_centre(a), body_centre(b))
                - body_extent(a) - body_extent(b))
    if pa and pb:
        gap = min(_segment_segment(p, q, r, s)
                  for p, q in _edges(pa) for r, s in _edges(pb))
        if gap > 0.0 and not (_inside((pa[0][0], pa[0][1]), pb)
                              or _inside((pb[0][0], pb[0][1]), pa)):
            return gap
        return -_overlap_depth(pa, pb)
    disc_body, poly = (b, pa) if pa else (a, pb)
    centre, radius = body_centre(disc_body), body_extent(disc_body)
    to_edge = min(_point_segment(centre, p, q) for p, q in _edges(poly))
    return -(to_edge + radius) if _inside(centre, poly) else to_edge - radius


def corridor_of(spacing: float, radius: float) -> float:
    return max(0.0, float(spacing) - 2.0 * float(radius))


def clears(body: Dict, at: Vec, placed: Sequence[Seat], *, spacing: float,
           radius: float) -> bool:
    extent = body_extent(body, radius)
    is_disc = not body.get("polygon")
    corridor = corridor_of(spacing, radius)
    moved: Optional[Dict] = None
    for other, size, seated in placed:
        apart = math.dist(at, other)
        if is_disc and not seated.get("polygon"):
            if apart < max(spacing, extent + size):
                return False
        elif apart < extent + size + corridor:
            if moved is None:
                moved = move_body(body, at)
            if surface_gap(moved, seated) < corridor:
                return False
    return True


def free_position(rng: random.Random, body: Dict, *, bounds: Vec,
                  start_region: Region, goal: Vec,
                  placed: Sequence[Seat], spacing: float, radius: float,
                  goal_radius: float = 0.0, attempts: int = 400,
                  avoid: Optional[Callable[[Vec, float], bool]] = None
                  ) -> Optional[Vec]:
    extent = body_extent(body, radius)
    for _ in range(attempts):
        point = (rng.uniform(extent, bounds[0] - extent),
                 rng.uniform(extent, bounds[1] - extent))
        if keepout(point, extent, start_region=start_region, goal=goal,
                   bounds=bounds, goal_radius=goal_radius):
            continue
        if avoid is not None and avoid(point, extent):
            continue
        if not clears(body, point, placed, spacing=spacing, radius=radius):
            continue
        return point
    return None


def disc(centre: Vec, radius: float) -> Dict:
    return {"position": [round(centre[0], 3), round(centre[1], 3)],
            "radius": round(float(radius), 3)}


def mover(index: int, centre: Vec, radius: float, motion: Dict) -> Dict:
    return {"id": "mover_%d" % index, "body": disc(centre, radius),
            "motion": motion}


def random_body(rng: random.Random, centre: Vec, *, rect_prob: float,
                radius: float, rect_radius: Optional[float] = None) -> Dict:
    ratio = rng.uniform(*RADIUS_RATIO)
    if rng.random() < rect_prob:
        extent = ratio * float(radius if rect_radius is None else rect_radius)
        length = rng.uniform(*WALL_LENGTH_RATIO)
        thickness = rng.uniform(*WALL_THICKNESS_RATIO)
        onto = 2.0 * extent / math.hypot(length, thickness)
        return {"polygon": rectangle(centre, length * onto, thickness * onto,
                                     rng.uniform(0.0, math.pi))}
    return {"position": [centre[0], centre[1]],
            "radius": round(ratio * float(radius), 6)}


def random_motion(rng: random.Random, *, speed: float, standoff: float,
                  side: float) -> Dict:
    drawn = round(min(rng.uniform(*MOVER_SPEED_BAND), 1.0) * float(speed), 6)
    hold = (standoff if standoff > 0.0
            else rng.uniform(*STANDOFF_BAND) * float(side))
    return {"kind": "strategy", "strategy": rng.choice(list(STRATEGIES)),
            "speed": drawn, "params": {"min_distance": round(hold, 6)}}


def mission_defaults(config: FamilyConfig) -> Dict[str, Any]:
    mission = config.mission
    if mission is None:
        raise ValueError("%s declares no mission block" % config.path)
    world = config.world
    radius = float(mission.obstacle_radius)
    return {
        "bounds": (world.x.hi - world.x.lo, world.y.hi - world.y.lo),
        "agents": int(config.swarm_size),
        "ticks": int(mission.ticks),
        "dt": float(config.physics.dt),
        "goal_radius": float(mission.goal_radius),
        "arrive_fraction": float(mission.arrive_fraction),
        "radius": radius,
        "rect_prob": float(mission.rect_prob),
        "rect_radius": mission.rect_radius,
        "max_dynamic": int(mission.attackers.count),
        "mover_radius": float(mission.attackers.size or radius),
        "mover_speed": float(mission.attackers.max_speed),
        "mover_standoff": float(mission.attackers.min_gap),
        "agent_boundary": float(config.physics.agent_boundary),
        "goal_speed": (None if mission.moving_goal is None
                       else mission.moving_goal.speed),
        "goal_lead": (None if mission.moving_goal is None
                      else mission.moving_goal.lead),
    }


def count_range(density: float, bounds: Vec) -> Tuple[int, int]:
    ceiling = int(round(float(density) * bounds[0] * bounds[1]))
    return (int(math.ceil(STATIC_COUNT_RATIO[0] * ceiling)),
            int(math.floor(STATIC_COUNT_RATIO[1] * ceiling)))


def mover_seat(spacing: float, body_radius: float, radius: float) -> float:
    return min(spacing, max(body_radius + radius, 2.0 * body_radius))


def scatter(rng: random.Random, wanted: int, placed: List[Seat],
            static: List[Dict], *, bounds: Vec, start_region: Region,
            goal: Vec, goal_radius: float, spacing: float, radius: float,
            rect_prob: float = 0.0, rect_radius: Optional[float] = None,
            avoid: Optional[Callable[[Vec, float], bool]] = None) -> int:
    misses = added = 0
    for _ in range(wanted):
        body = random_body(rng, (0.0, 0.0), rect_prob=rect_prob,
                           radius=radius, rect_radius=rect_radius)
        extent = body_extent(body, radius)
        centre = free_position(rng, body, bounds=bounds,
                               start_region=start_region, goal=goal,
                               goal_radius=goal_radius, placed=placed,
                               spacing=spacing, radius=radius, avoid=avoid)
        if centre is None:
            misses += 1
            if misses > _MISS_BUDGET:
                break
            continue
        misses = 0
        seated = move_body(body, centre)
        placed.append((centre, extent, seated))
        static.append(seated)
        added += 1
    return added


def random_field(seed: int, *, bounds: Vec, agents: int, ticks: int,
                 dt: float, goal_radius: float, density: float,
                 min_spacing: float, radius: float, rect_prob: float = 0.0,
                 rect_radius: Optional[float] = None,
                 mover_speed: float = 0.0, mover_standoff: float = 0.0,
                 max_dynamic: int = 0, mover_radius: Optional[float] = None,
                 agent_boundary: float = 0.0, name: Optional[str] = None,
                 **_ignored) -> Field:
    if density < 0.0 or radius <= 0.0:
        raise ValueError("density must be non-negative and radius positive")
    spacing = float(min_spacing)
    if spacing < 2.0 * radius:
        raise ValueError("min_spacing %g overlaps bodies of radius %g -- the "
                         "pair is one concave lump, not a gate"
                         % (spacing, radius))
    start_region = start_region_for(bounds, agents=agents,
                                    agent_boundary=agent_boundary)
    goal = goal_for(bounds)
    rng = random.Random(int(seed))
    count_min, count_max = count_range(density, bounds)
    wanted = rng.randint(count_min, count_max) if count_max > 0 else 0

    placed: List[Seat] = []
    static: List[Dict] = []
    scatter(rng, wanted, placed, static, bounds=bounds,
            start_region=start_region, goal=goal, goal_radius=goal_radius,
            spacing=spacing, radius=radius, rect_prob=rect_prob,
            rect_radius=rect_radius)

    n_dynamic = rng.randint(0, max(0, int(max_dynamic)))
    dynamic: List[Dict] = []
    mover_top = float(radius if mover_radius is None else mover_radius)
    for index in range(n_dynamic):
        body_radius = round(rng.uniform(*[r * mover_top
                                          for r in MOVER_RADIUS_RATIO]), 6)
        centre = free_position(rng, disc((0.0, 0.0), body_radius),
                               bounds=bounds, start_region=start_region,
                               goal=goal, goal_radius=goal_radius,
                               placed=placed,
                               spacing=mover_seat(spacing, body_radius,
                                                  radius),
                               radius=radius)
        if centre is None:
            continue
        dynamic.append(mover(index, centre, body_radius, random_motion(
            rng, speed=mover_speed, standoff=mover_standoff,
            side=min(bounds))))

    note = ("Random field: %d static bodies (%d drawn from %d-%d at rho "
            "%.4g, hard core %.4g) and %d dynamic; seed %d."
            % (len(static), wanted, count_min, count_max, float(density),
               spacing, len(dynamic), int(seed)))
    if len(static) < wanted:
        note += " SATURATED -- the hard core admits no more."
    return compose(name=name or "random-%d" % int(seed), description=note,
                   bounds=bounds, start_region=start_region, goal=goal,
                   goal_radius=goal_radius, agents=agents, ticks=ticks, dt=dt,
                   seed=int(seed), static=static, dynamic=dynamic)


def compose(*, name: str, description: str, bounds: Vec,
            start_region: Region, goal: Vec, goal_radius: float,
            agents: int, ticks: int, dt: float, seed: int,
            static: Sequence[Dict], dynamic: Sequence[Dict]) -> Field:
    return {
        "name": name,
        "description": description,
        "bounds": [float(bounds[0]), float(bounds[1])],
        "start_region": [float(v) for v in start_region],
        "goal": [float(goal[0]), float(goal[1])],
        "goal_radius": float(goal_radius),
        "agents": int(agents),
        "ticks": int(ticks),
        "dt": float(dt),
        "seed": int(seed),
        "static": [dict(b) for b in static],
        "dynamic": [dict(d) for d in dynamic],
    }


MOVING_GOAL = "moving_goal"


def moving_goal(field: Field, speed: Optional[float],
                lead: Optional[float] = None) -> Field:
    out = {key: value for key, value in field.items() if key != MOVING_GOAL}
    if speed is None:
        return out
    region = [float(v) for v in field["start_region"]]
    centre = (0.5 * (region[0] + region[2]), 0.5 * (region[1] + region[3]))
    block: Dict[str, Any] = {"start": [centre[0], centre[1]],
                             "speed": float(speed)}
    if lead is not None:
        goal = (float(field["goal"][0]), float(field["goal"][1]))
        span = math.dist(centre, goal)
        if span > 0.0:
            along = ((goal[0] - centre[0]) / span,
                     (goal[1] - centre[1]) / span)
            front = (0.5 * (region[2] - region[0]) * abs(along[0])
                     + 0.5 * (region[3] - region[1]) * abs(along[1]))
            ahead = min(front + float(lead), span)
            block["start"] = [round(centre[0] + ahead * along[0], 9),
                              round(centre[1] + ahead * along[1], 9)]
        block["lead"] = float(lead)
    out[MOVING_GOAL] = block
    return out


def _body_errors(body: Dict, where: str, bounds: Vec) -> List[str]:
    out: List[str] = []
    poly = body.get("polygon")
    if poly is not None:
        if len(poly) < 3:
            out.append("%s: polygon needs 3+ vertices" % where)
            return out
        if any(len(p) != 2 for p in poly):
            out.append("%s: polygon vertices must be [x, y]" % where)
            return out
    elif body.get("position") is None:
        out.append("%s: body has neither position nor polygon" % where)
        return out
    radius = body.get("radius")
    if radius is not None and float(radius) <= 0.0:
        out.append("%s: radius must be positive" % where)
    centre, extent = body_centre(body), body_extent(body)
    if not inside_bounds(centre, extent, bounds):
        out.append("%s: body at (%.1f, %.1f) extent %.1f leaves the map"
                   % (where, centre[0], centre[1], extent))
    return out


def _motion_errors(motion: Dict, where: str) -> List[str]:
    out: List[str] = []
    kind = motion.get("kind", "static")
    if kind not in MOTION_KINDS:
        out.append("%s: unknown motion kind %r -- have %s"
                   % (where, kind, ", ".join(MOTION_KINDS)))
        return out
    if kind == "strategy":
        if motion.get("strategy") not in STRATEGIES:
            out.append("%s: strategy must be one of %s"
                       % (where, ", ".join(STRATEGIES)))
        if float(motion.get("speed", 0.0)) <= 0.0:
            out.append("%s: strategy motion needs a positive speed" % where)
    return out


def validate(field: Field) -> List[str]:
    out: List[str] = []
    for key in ("bounds", "start_region", "goal", "agents", "ticks", "dt"):
        if key not in field:
            out.append("missing required key %r" % key)
    if out:
        return out
    bounds = tuple(float(v) for v in field["bounds"])
    if bounds[0] <= 0.0 or bounds[1] <= 0.0:
        return ["bounds must be positive"]
    region = tuple(float(v) for v in field["start_region"])
    if region[0] >= region[2] or region[1] >= region[3]:
        out.append("start_region must be [x0, y0, x1, y1] with x0<x1, y0<y1")
    goal = tuple(float(v) for v in field["goal"])
    if not (0.0 <= goal[0] <= bounds[0] and 0.0 <= goal[1] <= bounds[1]):
        out.append("goal lies outside the map")
    if int(field["agents"]) < 1:
        out.append("a mission needs at least one agent")
    if int(field["ticks"]) < 1:
        out.append("a mission needs at least one tick")
    if float(field["dt"]) <= 0.0:
        out.append("dt must be positive")
    motion = field.get(MOVING_GOAL)
    if motion is not None:
        start = motion.get("start") if isinstance(motion, dict) else None
        if (start is None or len(start) != 2
                or float(motion.get("speed", 0.0)) <= 0.0):
            out.append("moving_goal needs a start [x, y] and a positive speed")
    for index, body in enumerate(field.get("static", [])):
        where = "static[%d]" % index
        out.extend(_body_errors(body, where, bounds))
        if keepout(body_centre(body), body_extent(body), start_region=region,
                   goal=goal, bounds=bounds,
                   goal_radius=float(field.get("goal_radius", 0.0))):
            out.append("%s: fouls the spawn box or the goal disc" % where)
    for index, spec in enumerate(field.get("dynamic", [])):
        where = "dynamic[%d]" % index
        body = spec.get("body")
        if not isinstance(body, dict):
            out.append("%s: needs a body" % where)
            continue
        out.extend(_body_errors(body, where, bounds))
        out.extend(_motion_errors(spec.get("motion") or {}, where))
    out.extend(_overlap_errors(field))
    return out


def _overlap_errors(field: Field) -> List[str]:
    out: List[str] = []
    statics = list(field.get("static", []))
    movers = [spec["body"] for spec in field.get("dynamic", [])
              if isinstance(spec.get("body"), dict)]
    for index, body in enumerate(movers):
        for other in statics:
            slack = surface_gap(body, other)
            if slack < -1e-6:
                out.append("dynamic[%d]: starts %.3f inside a static body"
                           % (index, -slack))
                break
    for i in range(len(movers)):
        for j in range(i + 1, len(movers)):
            slack = surface_gap(movers[i], movers[j])
            if slack < -1e-6:
                out.append("dynamic[%d] and dynamic[%d]: overlap by %.3f"
                           % (i, j, -slack))
    return out


__all__ = ["Field", "MOVING_GOAL", "body_centre", "body_extent", "clears",
           "compose", "goal_for", "keepout", "mission_defaults", "moving_goal",
           "random_field", "start_region_for", "surface_gap", "validate"]
