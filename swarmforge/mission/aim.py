from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..solve.config import FamilyConfig
from ..solve.window import Reaches
from . import movers as M
from .field import (GOAL_CLEARANCE, MOVER_RADIUS_RATIO, MOVER_SPEED_BAND,
                    RADIUS_RATIO, STRATEGIES, Field, Seat, Vec, body_centre,
                    body_extent, clears, compose, disc, free_position,
                    goal_for, inside_bounds, keepout, mission_defaults,
                    move_body, mover, mover_seat, moving_goal, random_body,
                    random_motion, scatter, start_region_for, validate)

OPERATORS = ("random", "aim", "transplant", "mutate", "steer", "climb")
ATTEMPTS = 25
SCATTER_ROUNDS = 40


@dataclass(frozen=True)
class Slot:
    index: int
    band: int
    target_success: float
    rho: float
    spacing: float
    statics: int
    movers: int


class Course:
    def __init__(self, config: FamilyConfig, dense: bool = False,
                 tactics: bool = False, tactic_blind: bool = False,
                 mover_draw: str = "random"):
        knobs = mission_defaults(config)
        self.config = config
        self.dense = bool(dense)
        self.levers = M.levers(config.name) if tactics else None
        self.tactic_blind = bool(tactic_blind)
        if mover_draw not in ("random", "tactic", "learned"):
            raise ValueError("no mover draw %r" % mover_draw)
        self.mover_draw = mover_draw
        self.sampler: Optional[M.TacticDraw] = None
        self.drawn: List[str] = []
        self.knobs = knobs
        self.bounds: Vec = knobs["bounds"]
        self.start = start_region_for(self.bounds, agents=knobs["agents"],
                                      agent_boundary=knobs["agent_boundary"])
        self.origin: Vec = (0.5 * (self.start[0] + self.start[2]),
                            0.5 * (self.start[1] + self.start[3]))
        self.goal: Vec = goal_for(self.bounds)
        dx, dy = self.goal[0] - self.origin[0], self.goal[1] - self.origin[1]
        self.length = math.hypot(dx, dy)
        self.along: Vec = (dx / self.length, dy / self.length)
        self.across: Vec = (-self.along[1], self.along[0])
        self.radius = float(knobs["radius"])
        self.rect_prob = float(knobs["rect_prob"])
        self.rect_radius = knobs["rect_radius"]
        self.goal_radius = float(knobs["goal_radius"])
        self.side = min(self.bounds)
        physics = config.physics
        self.agent_radius = float(physics.agent_radius)
        self.boundary = float(physics.agent_boundary)
        self.step = float(physics.plant.max_step)
        self.max_dynamic = int(knobs["max_dynamic"])
        self.mover_speed = float(knobs["mover_speed"])
        self.mover_radius = float(knobs["mover_radius"])
        self.mover_standoff = float(knobs["mover_standoff"])
        horizon = config.model("obstacle_collision").horizon
        self.reach = float(Reaches.of(config, horizon).static_bodies)
        free = [s / 400.0 for s in range(401)
                if not keepout(self.at(s / 400.0), self.radius,
                               start_region=self.start, goal=self.goal,
                               bounds=self.bounds,
                               goal_radius=self.goal_radius)]
        self.free = (min(free), max(free)) if free else (0.5, 0.5)

    def at(self, s: float, y: float = 0.0) -> Vec:
        return (self.origin[0] + s * self.length * self.along[0]
                + y * self.across[0],
                self.origin[1] + s * self.length * self.along[1]
                + y * self.across[1])

    def contact(self, radius: Optional[float] = None) -> float:
        return (self.radius if radius is None else radius) + self.agent_radius

    def spacing(self, slot: Slot) -> float:
        return max(float(slot.spacing), 2.0 * self.radius)


def _seat(body: Dict) -> Seat:
    return (body_centre(body), body_extent(body), body)


def _fits(body: Dict, course: Course, placed: Sequence[Seat],
          spacing: float) -> bool:
    centre, extent = body_centre(body), body_extent(body, course.radius)
    if not inside_bounds(centre, extent, course.bounds):
        return False
    if keepout(centre, extent, start_region=course.start, goal=course.goal,
               bounds=course.bounds, goal_radius=course.goal_radius):
        return False
    return clears(body, centre, placed, spacing=spacing, radius=course.radius)


def _mover_fits(spec: Dict, course: Course, placed: Sequence[Seat],
                movers: Sequence[Dict], spacing: float) -> bool:
    body = spec["body"]
    centre = body_centre(body)
    radius = float(body["radius"])
    if not inside_bounds(centre, radius, course.bounds) or keepout(
            centre, radius, start_region=course.start, goal=course.goal,
            bounds=course.bounds, goal_radius=course.goal_radius):
        return False
    if not clears(body, centre, placed,
                  spacing=mover_seat(spacing, radius, course.radius),
                  radius=course.radius):
        return False
    return all(math.dist(centre, body_centre(m["body"]))
               >= radius + float(m["body"]["radius"]) for m in movers)


def _random_mover(rng: random.Random, course: Course, placed: Sequence[Seat],
                  movers: Sequence[Dict], spacing: float) -> Optional[Dict]:
    name = M.RANDOM
    if course.mover_draw == "tactic":
        name = rng.choice(sorted(M.TACTICS))
    elif course.mover_draw == "learned" and course.sampler is not None:
        name = course.sampler.draw()
    if name != M.RANDOM:
        spec = M.place(rng, M.TACTICS[name], course, placed, spacing)
        if spec is not None and _mover_fits(spec, course, placed, movers,
                                            spacing):
            course.drawn.append(name)
            return spec
    course.drawn.append(M.RANDOM)
    for _ in range(ATTEMPTS):
        radius = round(rng.uniform(*[r * course.mover_radius
                                     for r in MOVER_RADIUS_RATIO]), 6)
        centre = free_position(rng, disc((0.0, 0.0), radius),
                               bounds=course.bounds, start_region=course.start,
                               goal=course.goal,
                               goal_radius=course.goal_radius, placed=placed,
                               spacing=mover_seat(spacing, radius,
                                                  course.radius),
                               radius=course.radius)
        if centre is None:
            continue
        spec = mover(0, centre, radius, random_motion(
            rng, speed=course.mover_speed, standoff=course.mover_standoff,
            side=course.side))
        if _mover_fits(spec, course, placed, movers, spacing):
            return spec
    return None


def compose_slot(rng: random.Random, course: Course, slot: Slot, *,
                 keep: Sequence[Dict] = (), movers: Sequence[Dict] = (),
                 name: str, note: str, seed: int) -> Optional[Field]:
    spacing = course.spacing(slot)
    placed: List[Seat] = []
    static: List[Dict] = []
    kept = 0
    for body in keep:
        if len(static) >= slot.statics:
            break
        if _fits(body, course, placed, spacing):
            placed.append(_seat(body))
            static.append(body)
            kept += 1
    dynamic: List[Dict] = []
    for spec in movers:
        if len(dynamic) >= slot.movers:
            break
        if _mover_fits(spec, course, placed, dynamic, spacing):
            dynamic.append(spec)
    held = [_seat(spec["body"]) for spec in dynamic]
    placed.extend(held)
    if course.dense:
        _corridor_fill(rng, course, slot, placed, static)
    for _ in range(SCATTER_ROUNDS):
        if len(static) >= slot.statics:
            break
        scatter(rng, slot.statics - len(static), placed, static,
                bounds=course.bounds, start_region=course.start,
                goal=course.goal, goal_radius=course.goal_radius,
                spacing=spacing, radius=course.radius,
                rect_prob=course.rect_prob, rect_radius=course.rect_radius)
    placed[:] = [seat for seat in placed
                 if not any(seat is h for h in held)]
    while len(dynamic) < slot.movers:
        spec = _random_mover(rng, course, placed, dynamic, spacing)
        if spec is None:
            break
        dynamic.append(spec)
    for index, spec in enumerate(dynamic):
        spec["id"] = "mover_%d" % index
    knobs = course.knobs
    document = compose(
        name=name, description=("%s: %d static bodies (%d placed by "
                                "guidance) at rho %.4g, hard core %.4g; %d "
                                "dynamic; seed %d." % (
                                    note, len(static), kept, slot.rho,
                                    spacing, len(dynamic), seed)),
        bounds=course.bounds, start_region=course.start, goal=course.goal,
        goal_radius=course.goal_radius, agents=knobs["agents"],
        ticks=knobs["ticks"], dt=knobs["dt"], seed=int(seed), static=static,
        dynamic=dynamic)
    document = moving_goal(document, knobs["goal_speed"], knobs["goal_lead"])
    if validate(document):
        return None
    return document


def random_mission(seed: int, course: Course, slot: Slot,
                   name: str) -> Optional[Field]:
    return compose_slot(random.Random(int(seed)), course, slot, name=name,
                        note="Random field", seed=seed)


CORRIDOR = 0.15


def _corridor_fill(rng: random.Random, course: Course, slot: Slot,
                   placed: List[Seat], static: List[Dict]) -> None:
    half = CORRIDOR * course.side
    lo, hi = course.free
    spacing = course.spacing(slot)
    misses = 0
    while len(static) < slot.statics and misses < 200:
        body = random_body(rng, course.at(rng.uniform(lo, hi),
                                          rng.uniform(-half, half)),
                           rect_prob=course.rect_prob, radius=course.radius,
                           rect_radius=course.rect_radius)
        if _fits(body, course, placed, spacing):
            placed.append(_seat(body))
            static.append(body)
            misses = 0
        else:
            misses += 1


def corridor_mission(seed: int, course: Course, slot: Slot,
                     name: str) -> Optional[Field]:
    rng = random.Random(int(seed))
    placed: List[Seat] = []
    static: List[Dict] = []
    _corridor_fill(rng, course, slot, placed, static)
    return compose_slot(rng, course, slot, keep=static, name=name,
                        note="Corridor field", seed=seed)


def parse_body(label: str) -> Tuple[str, int]:
    if not label:
        return "", -1
    kind, index = label.split(":")
    return kind, int(index)


def _surface(body: Dict, point: Vec) -> float:
    return math.dist(body_centre(body), point) - body_extent(body)


def local_scene(document: Field, where: Vec, reach: float,
                body: str = "") -> Tuple[List[Dict], List[Dict]]:
    kind, k = parse_body(body)
    static = list(document.get("static", []))
    near = sorted((b for b in static if _surface(b, where) <= reach),
                  key=lambda b: _surface(b, where))
    if kind == "static" and 0 <= k < len(static):
        anchor = static[k]
        near = [anchor] + [b for b in near if b is not anchor]
    movers = list(document.get("dynamic", []))
    mine = ([movers[k]] if kind == "dynamic" and 0 <= k < len(movers)
            else [])
    return copy.deepcopy(near), copy.deepcopy(mine)


def transplant(seed: int, course: Course, slot: Slot, parent: Field,
               where: Vec, body: str, name: str) -> Optional[Field]:
    scene, movers = local_scene(parent, where, course.reach, body)
    if not scene and not movers:
        return None
    if scene and slot.statics < 1 or movers and slot.movers < 1:
        return None
    return compose_slot(random.Random(int(seed)), course, slot, keep=scene,
                        movers=movers, name=name,
                        note="Transplant of %s round (%.4g, %.4g)"
                        % (parent.get("name", "?"), where[0], where[1]),
                        seed=seed)


MUTATE_SHIFT = 0.6
MUTATE_ADD = 0.25
MUTATE_REMOVE = 0.2
MUTATE_STRATEGY = 0.3
INPUT_NEAR = 3


def _by_distance(document: Field, where: Vec, first: Sequence[Dict]
                 ) -> List[Dict]:
    ids = {id(b) for b in first}
    rest = sorted((b for b in document.get("static", []) if id(b) not in ids),
                  key=lambda b: _surface(b, where))
    return list(first) + rest


def _nudge_mover(rng: random.Random, course: Course, spec: Dict,
                 shift: float) -> Dict:
    spec = copy.deepcopy(spec)
    body = spec["body"]
    c = body_centre(body)
    body["position"] = [round(c[0] + rng.gauss(0.0, shift), 6),
                        round(c[1] + rng.gauss(0.0, shift), 6)]
    motion = spec.setdefault("motion", {})
    if rng.random() < MUTATE_STRATEGY:
        motion["strategy"] = rng.choice(list(STRATEGIES))
    if motion.get("speed"):
        lo, hi = (r * course.mover_speed for r in MOVER_SPEED_BAND)
        motion["speed"] = round(min(hi, max(lo, float(motion["speed"])
                                            * rng.uniform(0.85, 1.15))), 6)
    return spec


def mutate(seed: int, course: Course, slot: Slot, parent: Field, where: Vec,
           body: str, name: str) -> Optional[Field]:
    rng = random.Random(int(seed))
    parent = copy.deepcopy(parent)
    shift = MUTATE_SHIFT * course.contact()
    kind, k = parse_body(body)
    near = [b for b in parent.get("static", [])
            if _surface(b, where) <= course.reach]
    moved = {id(b): move_body(b, (body_centre(b)[0] + rng.gauss(0.0, shift),
                                  body_centre(b)[1] + rng.gauss(0.0, shift)))
             for b in near}
    static = [moved.get(id(b), b) for b in parent.get("static", [])]
    parent["static"] = static
    near = [moved[id(b)] for b in near]
    if near and rng.random() < MUTATE_REMOVE:
        anchor = (static[k] if kind == "static" and 0 <= k < len(static)
                  else None)
        others = [b for b in near if b is not anchor]
        if others:
            drop = min(others, key=lambda b: _surface(b, where))
            near.remove(drop)
            static.remove(drop)
    if rng.random() < MUTATE_ADD:
        angle = rng.uniform(-math.pi, math.pi)
        r = course.reach * math.sqrt(rng.random())
        extra = random_body(rng, (where[0] + r * math.cos(angle),
                                  where[1] + r * math.sin(angle)),
                            rect_prob=course.rect_prob, radius=course.radius,
                            rect_radius=course.rect_radius)
        near.append(extra)
        static.append(extra)
    movers = [(_nudge_mover(rng, course, spec, shift)
               if kind == "dynamic" and j == k else spec)
              for j, spec in enumerate(parent.get("dynamic", []))]
    keep = _by_distance(parent, where, near)
    return compose_slot(rng, course, slot, keep=keep, movers=movers,
                        name=name, note="Mutant of %s round (%.4g, %.4g)"
                        % (parent.get("name", "?"), where[0], where[1]),
                        seed=seed)


def mutate_input(seed: int, course: Course, parent: Field, where: Vec,
                 body: str, name: str, *, rho: float, spacing: float
                 ) -> Optional[Field]:
    rng = random.Random(int(seed))
    parent = copy.deepcopy(parent)
    shift = MUTATE_SHIFT * course.contact()
    kind, k = parse_body(body)
    static = list(parent.get("static", []))
    anchor = static[k] if kind == "static" and 0 <= k < len(static) else None
    near = [b for b in static if _surface(b, where) <= course.reach]
    if len(near) < INPUT_NEAR:
        near = sorted(static, key=lambda b: _surface(b, where))[:INPUT_NEAR]
    moved = {id(b): move_body(b, (body_centre(b)[0] + rng.gauss(0.0, shift),
                                  body_centre(b)[1] + rng.gauss(0.0, shift)))
             for b in near}
    static = [moved.get(id(b), b) for b in static]
    near = [moved[id(b)] for b in near]
    anchor = moved.get(id(anchor), anchor) if anchor is not None else None
    if near and rng.random() < MUTATE_REMOVE:
        others = [b for b in near if b is not anchor]
        if others:
            drop = min(others, key=lambda b: _surface(b, where))
            near.remove(drop)
            static.remove(drop)
    if rng.random() < MUTATE_ADD:
        angle = rng.uniform(-math.pi, math.pi)
        r = course.reach * math.sqrt(rng.random())
        extra = random_body(rng, (where[0] + r * math.cos(angle),
                                  where[1] + r * math.sin(angle)),
                            rect_prob=course.rect_prob, radius=course.radius,
                            rect_radius=course.rect_radius)
        near.append(extra)
        static.append(extra)
    movers = [(_nudge_mover(rng, course, spec, shift)
               if kind == "dynamic" and j == k else spec)
              for j, spec in enumerate(parent.get("dynamic", []))]
    parent["static"] = static
    slot = Slot(index=-1, band=-1, target_success=0.0, rho=float(rho),
                spacing=float(spacing), statics=len(static),
                movers=len(movers))
    return compose_slot(rng, course, slot,
                        keep=_by_distance(parent, where, near), movers=movers,
                        name=name, note="Input mutant of %s round (%.4g, %.4g)"
                        % (parent.get("name", "?"), where[0], where[1]),
                        seed=seed)


EDITS = ("onto_path", "off_path", "closer", "farther", "pair", "cup",
         "thin", "squeeze", "goalward", "backward")

LEVERS: Dict[str, Dict[str, int]] = {
    "m_goal_away1": {"onto_path": -1, "off_path": +1, "closer": -1,
                     "farther": +1},
    "m_away_body": {"closer": -1, "farther": +1, "cup": -1},
    "peer_dist": {"squeeze": -1, "pair": -1},
    "coast_gap": {"onto_path": -1, "off_path": +1, "closer": -1,
                  "farther": +1},
    "goal_dist": {"goalward": -1, "backward": +1},
    "body_push_size": {"pair": +1, "cup": +1, "thin": -1, "closer": +1,
                       "farther": -1},
    "goal_bearing": {"onto_path": -1, "off_path": +1},
    "flock_now_size": {"squeeze": +1, "pair": +1},
    "m_goal_away1_flock_near": {"onto_path": -1, "off_path": +1,
                                "closer": -1, "farther": +1, "squeeze": -1},
    "m_goal_flock_near": {"onto_path": -1, "off_path": +1, "closer": -1,
                          "farther": +1, "squeeze": -1},
    "flock_near_size": {"squeeze": +1, "pair": +1},
    "brake_excess": {"closer": +1, "farther": -1, "onto_path": +1,
                     "off_path": -1},
    "stop_speed": {"closer": -1, "farther": +1},
    "probe_d_ws0.3_wb3": {"squeeze": -1, "pair": -1, "thin": +1},
    "probe_d_ws0.3_wb0.3": {"squeeze": -1, "pair": -1, "thin": +1},
    "probe_d_ws3_wb0.3": {"squeeze": -1, "pair": -1, "thin": +1},
    "probe_d_ws3_wb3": {"squeeze": -1, "pair": -1, "thin": +1},
    "probe_b_ws0.03_wb0.3": {"onto_path": -1, "off_path": +1, "closer": -1,
                             "farther": +1},
    "d_coast_gap": {"squeeze": -1, "pair": -1},
    "gap1": {"closer": -1, "farther": +1, "onto_path": -1, "off_path": +1},
    "d_gap2": {"squeeze": -1, "pair": -1, "thin": +1},
    "hz_body_gap": {"pair": -1, "closer": -1, "thin": +1, "farther": +1},
    "flock_size": {"squeeze": +1, "pair": +1},
    "m_body_push": {"onto_path": -1, "off_path": +1},
    "s_centroid_dist": {"goalward": -1, "backward": +1},
    "s_v_goal": {"cup": -1, "pair": -1, "onto_path": -1, "thin": +1},
    "t_body_push_along": {"onto_path": -1, "cup": -1, "off_path": +1},
    "peer_dist_now": {"squeeze": -1, "pair": -1},
    "coast_gap1": {"closer": -1, "farther": +1, "onto_path": -1},
    "t_free_best_over_need": {"cup": -1, "pair": -1, "thin": +1},
    "t_flock_along": {"squeeze": -1, "pair": -1},
    "t_coast_over_need": {"cup": -1, "onto_path": -1, "thin": +1},
    "t_v_cross": {"onto_path": +1, "off_path": -1},
    "centroid_dist": {"squeeze": -1, "pair": -1},
}


def lever_scores(delta: Mapping[str, float]) -> Dict[str, float]:
    out = {e: 0.0 for e in EDITS}
    for name, d in delta.items():
        for edit, sign in LEVERS.get(name, {}).items():
            out[edit] += sign * d
    return out


def _frame(where: Vec, heading: Vec, course: Course) -> Tuple[Vec, Vec]:
    hx, hy = float(heading[0]), float(heading[1])
    norm = math.hypot(hx, hy)
    if norm < 1e-9:
        hx, hy = course.goal[0] - where[0], course.goal[1] - where[1]
        norm = math.hypot(hx, hy) or 1.0
    u = (hx / norm, hy / norm)
    return u, (-u[1], u[0])


def edit_scene(rng: random.Random, course: Course, slot: Slot, edit: str,
               where: Vec, heading: Vec, anchor: Optional[Dict],
               scene: List[Dict], size: float) -> List[Dict]:
    u, n = _frame(where, heading, course)
    contact = course.contact()
    step = size * contact
    out = list(scene)

    def put(body: Dict, a: float, l: float) -> Dict:
        return move_body(body, (where[0] + a * u[0] + l * n[0],
                                where[1] + a * u[1] + l * n[1]))

    def coords(body: Dict) -> Tuple[float, float]:
        c = body_centre(body)
        dx, dy = c[0] - where[0], c[1] - where[1]
        return dx * u[0] + dy * u[1], dx * n[0] + dy * n[1]

    def fresh(at: Vec) -> Dict:
        return random_body(rng, at, rect_prob=0.0, radius=course.radius)

    spacing = course.spacing(slot)
    if anchor is None:
        anchor = fresh((where[0] + 2.0 * contact * u[0],
                        where[1] + 2.0 * contact * u[1]))
        out = [anchor] + out
    a, l = coords(anchor)
    if edit in ("onto_path", "off_path", "closer", "farther", "goalward",
                "backward"):
        if edit == "onto_path":
            l = l - math.copysign(min(abs(l), step), l)
        elif edit == "off_path":
            l = l + math.copysign(step, l if l else rng.choice((-1, 1)))
        elif edit == "closer":
            a = max(a - step, 0.5 * contact)
        elif edit == "farther":
            a = a + step
        moved = put(anchor, a, l)
        if edit in ("goalward", "backward"):
            c = body_centre(anchor)
            gx, gy = course.goal[0] - c[0], course.goal[1] - c[1]
            g = math.hypot(gx, gy) or 1.0
            sign = 1.0 if edit == "goalward" else -1.0
            moved = move_body(anchor, (c[0] + sign * step * 2.0 * gx / g,
                                       c[1] + sign * step * 2.0 * gy / g))
        out[0] = moved
    elif edit == "pair":
        side = -1.0 if l > 0 else 1.0
        out.append(fresh((where[0] + a * u[0] + (l + side * spacing) * n[0],
                          where[1] + a * u[1] + (l + side * spacing) * n[1])))
    elif edit == "cup":
        out.append(fresh((where[0] + (a + spacing) * u[0] + l * n[0],
                          where[1] + (a + spacing) * u[1] + l * n[1])))
    elif edit == "thin":
        others = [b for b in out[1:]]
        if others:
            out.remove(min(others, key=lambda b: _surface(b, where)))
    elif edit == "squeeze":
        back = 2.0 * spacing + step
        for side in (-0.5, 0.5):
            out.append(fresh((where[0] - back * u[0] + side * spacing * n[0],
                              where[1] - back * u[1] + side * spacing * n[1])))
    return out


MOVER_EDIT = "mover:"


def _retask(rng: random.Random, course: Course, slot: Slot, parent: Field,
            tactic: str, k: int, name: str, seed: int) -> Optional[Field]:
    static = list(parent.get("static", []))
    movers = list(parent.get("dynamic", []))
    radius = float(movers[k]["body"]["radius"])
    spec = M.place(rng, M.TACTICS[tactic], course, [_seat(b) for b in static],
                   course.spacing(slot), radius=radius)
    if spec is None:
        return None
    movers[k] = spec
    course.drawn.append(tactic)
    return compose_slot(rng, course, slot, keep=static, movers=movers,
                        name=name, note="Re-tasked mover of %s (%s)" % (
                            parent.get("name", "?"), tactic), seed=seed)


def steer(seed: int, course: Course, slot: Slot, parent: Field, where: Vec,
          body: str, heading: Vec, delta: Mapping[str, float], name: str,
          greedy: bool = True, model: str = "",
          edit: Optional[str] = None, size: Optional[float] = None
          ) -> Tuple[Optional[Field], str]:
    rng = random.Random(int(seed))
    parent = copy.deepcopy(parent)
    kind, k = parse_body(body)
    static = parent.get("static", [])
    anchor = (static[k] if kind == "static" and 0 <= k < len(static)
              else None)
    scores = lever_scores(delta)
    if (greedy and course.levers is not None and model
            and parent.get("dynamic") and slot.movers >= 1):
        scores.update({MOVER_EDIT + t: v for t, v in
                       M.scores(course.levers, model, delta).items()})
    if edit is not None:
        if edit.startswith(MOVER_EDIT) and not (parent.get("dynamic")
                                                and slot.movers >= 1):
            return None, edit
    elif greedy:
        positive = {e: s for e, s in scores.items() if s > 0.0}
        pool = positive or scores
        edits = sorted(pool, key=lambda e: -pool[e])[:3]
        weights = [max(pool[e], 1e-6) for e in edits]
        edit = rng.choices(edits, weights=weights)[0]
    else:
        edit = rng.choice(list(EDITS))
    if edit.startswith(MOVER_EDIT):
        tactic = (M.choose(rng, {}, blind=True) if course.tactic_blind
                  else edit[len(MOVER_EDIT):])
        mover_k = k if kind == "dynamic" and 0 <= k < len(
            parent["dynamic"]) else 0
        document = _retask(rng, course, slot, parent, tactic, mover_k, name,
                           seed)
        if document is not None:
            return document, MOVER_EDIT + tactic
        edit = max(EDITS, key=lambda e: scores[e])
    if size is None:
        size = float(np.clip(np.mean([abs(v) for v in delta.values()]) * 4.0
                             if delta else 0.5, 0.25, 1.5))
    near = [b for b in static if _surface(b, where) <= course.reach
            and b is not anchor]
    scene = ([anchor] if anchor is not None else []) + near
    edited = edit_scene(rng, course, slot, edit, where, heading, anchor,
                        scene, size * rng.uniform(0.7, 1.3))
    gone = {id(b) for b in scene} - {id(b) for b in edited}
    kept = [b for b in static if id(b) not in gone and id(b) not in
            {id(x) for x in scene}]
    keep = edited + sorted(kept, key=lambda b: _surface(b, where))
    movers = list(parent.get("dynamic", []))
    if kind == "dynamic" and 0 <= k < len(movers):
        movers[k] = _nudge_mover(rng, course, movers[k],
                                 MUTATE_SHIFT * course.contact())
    document = compose_slot(rng, course, slot, keep=keep, movers=movers,
                            name=name, note="Steered (%s) %s round "
                            "(%.4g, %.4g)" % (edit, parent.get("name", "?"),
                                              where[0], where[1]), seed=seed)
    return document, edit


def _raw(value: Optional[float]) -> Optional[float]:
    return None if value is None or not np.isfinite(value) else float(value)


def aim_scene(rng: random.Random, course: Course, slot: Slot, model: str,
              target: Mapping[str, float], quantile: Mapping[str, float],
              unit: float, goal_cap: float) -> Tuple[List[Dict], List[Dict],
                                                     str]:
    spacing = course.spacing(slot)
    contact = course.contact()
    lo, hi = course.free
    s = rng.uniform(lo + 0.1 * (hi - lo), hi)
    d = None
    for name in ("goal_dist", "s_centroid_dist"):
        v = _raw(target.get(name))
        if v is not None:
            scale = unit if name == "goal_dist" else (
                course.config.model("timeout").horizon * course.step)
            d = scale * (math.expm1(v))
            break
    if d is not None and d < goal_cap * 0.98:
        s = float(np.clip(1.0 - d / course.length, lo, hi))
    bearing = _raw(target.get("goal_bearing"))
    if bearing is None:
        lateral = rng.uniform(-0.5, 0.5) * contact
    else:
        lateral = math.tan(math.radians(min(abs(bearing), 80.0))) * contact \
            * rng.choice((-1.0, 1.0))
    crowd = max(quantile.get("body_push_size", 0.0),
                1.0 - quantile.get("t_free_best_over_need", 1.0),
                1.0 - quantile.get("hz_body_gap", 1.0),
                1.0 - quantile.get("t_body_push_along", 1.0))
    packed = max(1.0 - quantile.get("peer_dist", 1.0),
                 1.0 - quantile.get("peer_dist_now", 1.0),
                 quantile.get("flock_now_size", 0.0),
                 quantile.get("flock_size", 0.0),
                 1.0 - min((quantile.get(n, 1.0) for n in quantile
                            if n.startswith("probe_d_")), default=1.0))
    bodies: List[Dict] = []
    where = course.at(s)

    def fresh(at: Vec) -> Dict:
        return random_body(rng, at, rect_prob=0.0, radius=course.radius)

    account = ["at %.2f of the route" % s, "offset %.3g" % lateral]
    if slot.statics >= 1:
        bodies.append(fresh(course.at(s, lateral)))
        if crowd > 0.6 and slot.statics >= 2:
            bodies.append(fresh(course.at(s, lateral + math.copysign(
                spacing, -lateral if lateral else 1.0))))
            account.append("pair")
            if (1.0 - quantile.get("t_free_best_over_need", 1.0) > 0.6
                    and slot.statics >= 3):
                bodies.append(fresh(course.at(s + spacing / course.length,
                                              lateral * 0.5)))
                account.append("cup")
        if packed > 0.6 and slot.statics >= len(bodies) + 2:
            back = s - (2.0 * spacing + contact) / course.length
            if back > lo:
                bodies.append(fresh(course.at(back, 0.5 * spacing)))
                bodies.append(fresh(course.at(back, -0.5 * spacing)))
                account.append("squeeze upstream")
    movers: List[Dict] = []
    if slot.movers >= 1 and course.max_dynamic >= 1 and \
            course.levers is not None:
        ranked = M.scores(course.levers, model,
                          M.seed_delta(course.levers, model, quantile))
        name = M.choose(rng, ranked, course.tactic_blind)
        if name is not None:
            spec = M.place(rng, M.TACTICS[name], course,
                           [_seat(b) for b in bodies], course.spacing(slot))
            if spec is not None:
                movers.append(spec)
                account.append("mover %s" % name)
                course.drawn.append(name)
    elif slot.movers >= 1 and course.max_dynamic >= 1:
        fast = max(quantile.get("brake_excess", 0.0),
                   1.0 - quantile.get("coast_gap", 1.0),
                   1.0 - quantile.get("d_coast_gap", 1.0))
        radius = round(rng.uniform(*[r * course.mover_radius
                                     for r in MOVER_RADIUS_RATIO]), 6)
        motion = random_motion(rng, speed=course.mover_speed,
                               standoff=course.mover_standoff,
                               side=course.side)
        if fast > 0.6:
            motion["speed"] = round(course.mover_speed
                                    * rng.uniform(0.85, 1.0), 6)
        ahead = min(hi, s + rng.uniform(0.05, 0.2))
        movers.append(mover(0, course.at(ahead, rng.uniform(-1.0, 1.0)
                                         * 3.0 * contact), radius, motion))
        account.append("mover %s" % motion["strategy"])
    del where
    return bodies, movers, ", ".join(account)


def aim(seed: int, course: Course, slot: Slot, model: str,
        target: Mapping[str, float], quantile: Mapping[str, float],
        unit: float, goal_cap: float, name: str
        ) -> Tuple[Optional[Field], str]:
    rng = random.Random(int(seed))
    for _ in range(ATTEMPTS):
        bodies, movers, account = aim_scene(rng, course, slot, model, target,
                                            quantile, unit, goal_cap)
        if not bodies and not movers:
            return None, ""
        spacing = course.spacing(slot)
        placed: List[Seat] = []
        if all(_fits(b, course, placed, spacing)
               and placed.append(_seat(b)) is None for b in bodies):
            document = compose_slot(rng, course, slot, keep=bodies,
                                    movers=movers, name=name,
                                    note="Aimed field (%s)" % account,
                                    seed=seed)
            if document is not None:
                return document, account
    return None, ""


__all__ = ["ATTEMPTS", "Course", "EDITS", "LEVERS", "OPERATORS", "Slot",
           "aim", "aim_scene", "compose_slot", "edit_scene", "lever_scores",
           "local_scene", "mutate", "mutate_input", "parse_body", "random_mission", "steer",
           "transplant"]
