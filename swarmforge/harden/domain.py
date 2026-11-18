from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..solve.config import FamilyConfig
from ..solve.record import Record, Term

DIST, CLEARANCE, REL_SPEED, SHILL_SPEED = ("dist", "clearance", "rel_speed",
                                           "shill_speed")


@dataclass(frozen=True)
class Domain:
    law: str
    spans: Mapping[str, Tuple[float, float]]
    reasons: Mapping[str, str]
    offset: Optional[float] = None
    extent: Optional[Mapping[str, Tuple[float, float]]] = None
    sensing: Optional[float] = None

    def reach_span(self, name: str) -> Optional[Tuple[float, float]]:
        pair = (self.extent or self.spans).get(name)
        return None if pair is None else (float(pair[0]), float(pair[1]))

    def span(self, name: str) -> Optional[Tuple[float, float]]:
        pair = self.spans.get(name)
        return None if pair is None else (float(pair[0]), float(pair[1]))

    def describe(self) -> str:
        return "%s: %s" % (self.law, "; ".join(
            "%s in [%.4g, %.4g] (%s)" % (name, lo, hi, self.reasons[name])
            for name, (lo, hi) in self.spans.items()))

    def to_dict(self) -> Dict[str, Any]:
        return {"law": self.law,
                "spans": {k: [v[0], v[1]] for k, v in self.spans.items()},
                "reasons": dict(self.reasons), "offset": self.offset,
                "extent": (None if self.extent is None else
                           {k: [v[0], v[1]] for k, v in self.extent.items()}),
                "sensing": self.sensing}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Domain":
        offset, extent = raw.get("offset"), raw.get("extent")
        return cls(str(raw["law"]),
                   {k: (float(v[0]), float(v[1]))
                    for k, v in raw["spans"].items()},
                   dict(raw.get("reasons") or {}),
                   None if offset is None else float(offset),
                   None if extent is None else
                   {k: (float(v[0]), float(v[1])) for k, v in extent.items()},
                   raw.get("sensing"))


@dataclass(frozen=True)
class MemberKind:
    what: str
    floor_lo: float
    floor_hi: float
    size_lo: float
    size_hi: float
    reason: str

    def clearance_floor(self) -> float:
        return min(self.floor_lo - self.size_lo, self.floor_hi - self.size_hi)


def _subtracted(config: FamilyConfig, radius: float, peer: bool) -> float:
    geometry = config.physics.geometry
    if geometry.clearance == "self":
        return config.physics.agent_radius
    if geometry.clearance == "other":
        return geometry.peer_size if peer else radius
    return 0.0


def member_kinds(source: str, config: FamilyConfig) -> List[MemberKind]:
    physics = config.physics
    geometry = physics.geometry
    if source == "target":
        size = _subtracted(config, 0.0, False)
        return [MemberKind("target", 0.0, 0.0, size, size,
                           "the target may be reached")]
    out: List[MemberKind] = []
    if source in ("agents", "bodies"):
        if geometry.peers == "square":
            floor = max(0.0, physics.agent_boundary
                        - math.sqrt(2.0) * geometry.peer_half_size)
            why = ("a peer's square seen across its corner at the agent "
                   "boundary %.4g" % physics.agent_boundary)
        else:
            floor = physics.agent_boundary
            why = "the agent boundary %.4g" % floor
        size = _subtracted(config, 0.0, True)
        out.append(MemberKind("peer", floor, floor, size, size, why))
    if source in ("obstacles", "bodies"):
        bodies = []
        if config.state.static is not None:
            bodies.append(("static body", config.state.static.radius))
            if config.state.static.wall_share > 0.0:
                bodies.append(("wall", None))
        if config.state.dynamic is not None:
            bodies.append(("moving body", config.state.dynamic.radius))
        if not bodies:
            raise ValueError("the config draws no obstacle, so a law reading "
                             "%r has no floor" % source)
        reach = physics.agent_radius + physics.obstacle_min_gap
        for what, radius in bodies:
            lo, hi = (0.0, 0.0) if radius is None else (radius.lo, radius.hi)
            if geometry.obstacles == "square" or radius is None:
                floor_lo = floor_hi = reach
                why = "contact with a %s's surface at %.4g" % (what, reach)
            else:
                floor_lo, floor_hi = reach + lo, reach + hi
                why = "contact with a %s of radius %.4g at %.4g" % (
                    what, lo, floor_lo)
            out.append(MemberKind(what, floor_lo, floor_hi,
                                  _subtracted(config, lo, False),
                                  _subtracted(config, hi, False), why))
    if not out:
        raise ValueError("source %r has no member kinds" % source)
    return out


def _constant(term: Term) -> Optional[float]:
    if term.gate is None or term.gate.threshold.free_symbols:
        return None
    return float(term.gate.threshold)


def top_speed(config: FamilyConfig) -> float:
    return config.physics.plant.max_step / config.physics.dt


def domain_of(term: Term, record: Record, config: FamilyConfig) -> Domain:
    kinds = member_kinds(term.source, config)
    world = config.world
    diagonal = math.hypot(world.x.hi - world.x.lo, world.y.hi - world.y.lo)
    size_min = min(min(k.size_lo, k.size_hi) for k in kinds)
    size_max = max(max(k.size_lo, k.size_hi) for k in kinds)

    nearest = min(kinds, key=lambda k: min(k.floor_lo, k.floor_hi))
    dist_lo = min(nearest.floor_lo, nearest.floor_hi)
    dist_lo_why = nearest.reason
    tight = min(kinds, key=MemberKind.clearance_floor)
    clear_lo = tight.clearance_floor()
    clear_lo_why = "%s, less %s" % (tight.reason, _subtract_text(config))

    dist_hi, dist_hi_why = diagonal, "the world's diagonal %.4g" % diagonal
    reach = (None if not term.range_param
             else record.params.get(term.range_param))
    sensing = None
    if reach is not None and reach < dist_hi:
        dist_hi, dist_hi_why = float(reach), "the sensing range %s = %.4g" % (
            term.range_param, reach)
        sensing = float(reach)
    clear_hi, clear_hi_why = (dist_hi - size_min,
                              "%s, less the least size %.4g"
                              % (dist_hi_why, size_min))
    extent = {DIST: (dist_lo, dist_hi), CLEARANCE: (clear_lo, clear_hi)}

    threshold = _constant(term)
    if threshold is not None:
        kind, feature = term.gate.kind, term.gate.feature
        why = "the %s gate on %s at %.4g" % (kind, feature, threshold)
        if kind == "below" and feature == DIST and threshold < dist_hi:
            dist_hi, dist_hi_why = threshold, why
            if threshold - size_min < clear_hi:
                clear_hi, clear_hi_why = threshold - size_min, why
        elif kind == "below" and feature == CLEARANCE:
            if threshold < clear_hi:
                clear_hi, clear_hi_why = threshold, why
            if threshold + size_max < dist_hi:
                dist_hi, dist_hi_why = threshold + size_max, why
        elif kind == "above" and feature == DIST:
            if threshold > dist_lo:
                dist_lo, dist_lo_why = threshold, why
            if threshold - size_max > clear_lo:
                clear_lo, clear_lo_why = threshold - size_max, why
        elif kind == "above" and feature == CLEARANCE:
            if threshold > clear_lo:
                clear_lo, clear_lo_why = threshold, why
            if threshold + size_min > dist_lo:
                dist_lo, dist_lo_why = threshold + size_min, why

    spans = {DIST: (dist_lo, dist_hi), CLEARANCE: (clear_lo, clear_hi)}
    reasons = {DIST: "%s up to %s" % (dist_lo_why, dist_hi_why),
               CLEARANCE: "%s up to %s" % (clear_lo_why, clear_hi_why)}
    speed = top_speed(config)
    spans[REL_SPEED] = (0.0, 2.0 * speed)
    reasons[REL_SPEED] = "two drones closing at the top speed %.4g" % speed
    if term.direction == "shill":
        spans[SHILL_SPEED] = (0.0, float(term.signal_speed or 0.0) + speed)
        reasons[SHILL_SPEED] = ("the virtual agent's %.4g plus the top speed"
                                % float(term.signal_speed or 0.0))
    for name in (DIST, CLEARANCE):
        lo, hi = spans[name]
        if not hi > lo:
            raise ValueError("law %r has an empty %s domain [%.4g, %.4g]: %s"
                             % (term.name, name, lo, hi, reasons[name]))
    for name in (REL_SPEED, SHILL_SPEED):
        if name in spans:
            extent[name] = spans[name]
    sizes = {size for kind in kinds for size in (kind.size_lo, kind.size_hi)}
    offset = sizes.pop() if len(sizes) == 1 else None
    return Domain(term.name, spans, reasons, offset, extent, sensing)


def _subtract_text(config: FamilyConfig) -> str:
    clearance = config.physics.geometry.clearance
    return {"self": "the drone's own radius",
            "other": "the other body's size",
            "surface": "nothing (the law reads the surface)"}[clearance]


def domains_for(record: Record, config: FamilyConfig) -> Dict[str, Domain]:
    return {term.name: domain_of(term, record, config)
            for term in record.terms}


__all__ = ["CLEARANCE", "DIST", "Domain", "MemberKind", "REL_SPEED",
           "SHILL_SPEED", "domain_of", "domains_for", "member_kinds",
           "top_speed"]
