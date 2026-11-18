from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .field import (MOVER_RADIUS_RATIO, MOVER_SPEED_BAND, STRATEGIES, Seat,
                    Vec, mover)

NUDGES = 12


@dataclass(frozen=True)
class Leg:
    strategy: str
    share: float = 0.0
    knobs: Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class Tactic:
    name: str
    start: str
    s: float
    legs: Tuple[Leg, ...]
    speed: float
    intent: str


TACTICS: Dict[str, Tactic] = {t.name: t for t in (
    Tactic("chase", "behind", 0.0, (Leg("S2", knobs=(("bearing", 180.0),)),),
           1.0, "station astern of the flock at full speed: pushes it on, "
                "packs its rear into its front"),
    Tactic("flank", "beside", 0.25, (Leg("S2", knobs=(("bearing", "side90"),)),),
           1.0, "station abeam: pushes the flock across its route"),
    Tactic("headon", "ahead", 0.55, (Leg("S2", knobs=(("bearing", 0.0),)),),
           1.0, "station dead ahead of the flock's heading"),
    Tactic("block", "ahead", 0.7, (Leg("S6"),), 1.0,
           "station between the flock and its goal"),
    Tactic("slow_block", "ahead", 0.7, (Leg("S6"),), 0.4,
           "the same, slowly: arrives late and gives way slowly"),
    Tactic("split", "beside", 0.3, (Leg("S1"),), 1.0,
           "through the flock's waist along its thinnest direction"),
    Tactic("herd", "beside", 0.35, (Leg("S3"),), 1.0,
           "the far side of the flock from the nearest body: drives it on"),
    Tactic("gate", "beside", 0.35, (Leg("S4", knobs=(("spacing", "tight"),)),),
           1.0, "beside the body nearest the flock, a tight gate's other "
                "wall"),
    Tactic("shear", "beside", 0.25, (Leg("S5", knobs=(("side", "side"),)),),
           1.0, "leans on one flank only"),
    Tactic("chase_herd", "behind", 0.0,
           (Leg("S2", 0.35, (("bearing", 180.0),)), Leg("S3")), 1.0,
           "pushes the flock on, then onto the nearest body"),
    Tactic("chase_block", "behind", 0.0,
           (Leg("S2", 0.3, (("bearing", 180.0),)), Leg("S6")), 1.0,
           "pushes the flock on, then cuts across its way to the goal"),
    Tactic("block_early", "ahead", 0.45, (Leg("S6"),), 1.0,
           "blocks from early on the route"),
    Tactic("block_late", "ahead", 0.85, (Leg("S6"),), 1.0,
           "blocks short of the goal"),
    Tactic("headon_split", "ahead", 0.55,
           (Leg("S2", 0.4, (("bearing", 0.0),)), Leg("S1")), 1.0,
           "holds the flock up from ahead, then cuts through it"),
    Tactic("block_herd", "ahead", 0.6, (Leg("S6", 0.45), Leg("S3")), 1.0,
           "holds the flock up, then drives it onto the nearest body"),
    Tactic("split_deep", "beside", 0.3, (Leg("S1", knobs=(("approach", 2.0),)),),
           1.0, "through the waist and on, twice the stand-off past it"),
    Tactic("rear_quarter", "behind", 0.0,
           (Leg("S2", knobs=(("bearing", "side150"),)),), 1.0,
           "station off the flock's rear quarter: packs it from behind "
           "and the side"),
    Tactic("chase_split", "behind", 0.0,
           (Leg("S2", 0.3, (("bearing", 180.0),)), Leg("S1")), 1.0,
           "pushes the flock on, then cuts through it"),
    Tactic("gate_wide", "beside", 0.35,
           (Leg("S4", knobs=(("spacing", "core"),)),), 1.0,
           "beside the body nearest the flock, at the slot's hard core"),
)}


def route_ticks(course) -> int:
    return max(1, int(round(course.length / course.step)))


def _params(leg: Leg, course, side: float, radius: float,
            spacing: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {"min_distance": round(course.mover_standoff, 6)}
    for key, value in leg.knobs:
        if value == "side90":
            value = 90.0 * side
        elif value == "side150":
            value = 150.0 * side
        elif value == "side":
            value = side
        elif value == "tight":
            value = course.radius + radius + 3.0 * course.agent_radius
        elif value == "core":
            value = float(spacing)
        out[key] = round(value, 6) if isinstance(value, float) else value
    return out


def motion(tactic: Tactic, course, side: float, radius: float,
           spacing: float) -> Dict:
    for leg in tactic.legs:
        if leg.strategy not in STRATEGIES:
            raise ValueError("no strategy %r" % leg.strategy)
    lo, hi = MOVER_SPEED_BAND
    speed = round(min(hi, max(lo, tactic.speed)) * course.mover_speed, 6)
    first = tactic.legs[0]
    out = {"kind": "strategy", "strategy": first.strategy, "speed": speed,
           "params": _params(first, course, side, radius, spacing)}
    if len(tactic.legs) > 1:
        ticks = route_ticks(course)
        out["stages"] = [{"strategy": leg.strategy, "speed": speed,
                          "params": _params(leg, course, side, radius,
                                            spacing),
                          "duration": int(round(leg.share * ticks))}
                         for leg in tactic.legs]
    return out


def _wants(tactic: Tactic, course, side: float) -> List[Vec]:
    h = 0.15 * course.side
    if tactic.start == "behind":
        out = [course.at(-f) for f in (0.12, 0.16, 0.2, 0.08)]
        out += [course.at(0.0, side * f * h) for f in (1.2, 1.6, 2.0)]
        return out
    if tactic.start == "far":
        return [(course.bounds[0] * 0.95, course.bounds[1] * 0.05),
                (course.bounds[0] * 0.05, course.bounds[1] * 0.95)]
    if tactic.start == "ahead":
        return [course.at(tactic.s + d, 0.0) for d in (0.0, 0.05, -0.05,
                                                       0.1, -0.1)]
    return [course.at(tactic.s + d, side * f * h)
            for d in (0.0, 0.05, -0.05) for f in (0.6, 0.9, 0.4)]


LEVER_PATH = Path(__file__).with_name("mover_levers.json")
_LEVERS: Dict[str, Any] = {}


def levers(family: str) -> Optional[Dict[str, Any]]:
    if not _LEVERS:
        _LEVERS.update(json.loads(LEVER_PATH.read_text())
                       if LEVER_PATH.is_file() else {"_": {}})
    return _LEVERS.get(family)


def scores(table: Mapping[str, Any], model: str,
           delta: Mapping[str, float]) -> Dict[str, float]:
    out = {}
    for name, by_model in table["tactics"].items():
        effect = by_model.get(model, {})
        total = 0.0
        for feature, d in delta.items():
            e = effect.get(feature, 0)
            total += abs(d) if e == 2 else e * d
        out[name] = total
    return out


def seed_delta(table: Mapping[str, Any], model: str,
               quantile: Mapping[str, float]) -> Dict[str, float]:
    ref = table["reference"].get(model, {})
    return {f: float(q) - float(ref[f]) for f, q in quantile.items()
            if f in ref}


PRIOR_FLIGHTS = 10.0
RANDOM = "random"


class TacticDraw:
    def __init__(self, table: Mapping[str, Any], rng,
                 weight: float = PRIOR_FLIGHTS):
        self.rng = rng
        outcomes = table.get("outcomes", {})
        self.names = [n for n in [RANDOM] + sorted(TACTICS) if n in outcomes]
        self.post = {n: [1.0 + weight * outcomes[n]["fail"],
                         1.0 + weight * (1.0 - outcomes[n]["fail"])]
                     for n in self.names}
        self.pulls = {n: 0 for n in self.names}

    def draw(self) -> str:
        if not self.names:
            return RANDOM
        score = {n: self.rng.beta(*self.post[n]) for n in self.names}
        return max(self.names, key=lambda n: score[n])

    def update(self, name: str, failed: bool) -> None:
        if name in self.post:
            self.post[name][0 if failed else 1] += 1.0
            self.pulls[name] += 1

    def table(self) -> List[Dict[str, Any]]:
        return [{"tactic": n, "pulls": self.pulls[n],
                 "p_fail": round(self.post[n][0] / sum(self.post[n]), 4)}
                for n in self.names]


def choose(rng: random.Random, ranked: Mapping[str, float],
           blind: bool = False) -> Optional[str]:
    if blind:
        return rng.choice(sorted(TACTICS))
    positive = {t: s for t, s in ranked.items() if s > 0.0 and t in TACTICS}
    if not positive:
        return None
    best = sorted(positive, key=lambda t: -positive[t])[:3]
    return rng.choices(best, weights=[positive[t] for t in best])[0]


def place(rng: random.Random, tactic: Tactic, course, placed: Sequence[Seat],
          spacing: float, side: Optional[float] = None,
          radius: Optional[float] = None) -> Optional[Dict]:
    from .aim import _mover_fits

    side = side if side is not None else rng.choice((-1.0, 1.0))
    if radius is None:
        radius = round(rng.uniform(*[r * course.mover_radius
                                     for r in MOVER_RADIUS_RATIO]), 6)
    block = motion(tactic, course, side, radius, spacing)
    if tactic.start == "far":
        block = {"kind": "static"}
    jitter = 0.5 * radius + course.contact()
    for want in _wants(tactic, course, side):
        for k in range(NUDGES + 1):
            at = want if k == 0 else (want[0] + rng.gauss(0.0, jitter),
                                      want[1] + rng.gauss(0.0, jitter))
            spec = mover(0, at, radius, dict(block))
            if _mover_fits(spec, course, placed, (), spacing):
                return spec
    return None


__all__ = ["LEVER_PATH", "Leg", "PRIOR_FLIGHTS", "RANDOM", "TACTICS",
           "Tactic", "TacticDraw", "choose", "levers", "motion", "place",
           "route_ticks", "scores", "seed_delta"]
