from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import sympy as sp

from .expr import (AXIS, DIST, REL_SPEED, Expr, Params,
                   ParamSpec, sigma1, sigma_norm)


class Role(Enum):
    AGENT = "agent"
    LEADER = "leader"
    FOLLOWER = "follower"


class Source(Enum):
    AGENTS = "agents"
    OBSTACLES = "obstacles"
    TARGET = "target"
    SELF = "self"
    BODIES = "bodies"


@dataclass
class Membership:
    range_param: Optional[str] = None
    fov_param: Optional[str] = None
    cap_param: Optional[str] = None
    topology: str = "global"
    gate: str = "off"
    mutable: bool = True


class Signal(Enum):
    POSITION = "position"
    VELOCITY = "velocity"


class Anchor(Enum):
    SELF = "self"
    CENTROID = "centroid"


class Direction(Enum):
    TOWARD = "toward"
    MATCH = "match"
    AWAY_RAW = "away_raw"
    AWAY_UNIT = "away_unit"
    SIGMA_GRAD = "sigma_grad"
    AXIS = "axis"
    PERP = "perp"
    FRICTION = "friction"
    FORMATION = "formation"
    SHILL = "shill"


class ActivationKind(Enum):
    ALWAYS = "always"
    BELOW = "below"
    ABOVE = "above"
    BUMP = "bump"


@dataclass
class Activation:
    kind: ActivationKind = ActivationKind.ALWAYS
    threshold: Optional[Expr] = None
    feature: Optional[str] = None


class Aggregator(Enum):
    SUM = "sum"
    NEAREST = "nearest"
    MEAN = "mean"
    MEDIAN = "median"
    TRIMMED = "trimmed"
    HUBER = "huber"


class Bound(Enum):
    NONE = "none"
    CLAMP_SPEED = "clamp_speed"
    CLAMP_ACCEL = "clamp_accel"
    RENORMALIZE = "renormalize"


@dataclass
class BoundOp:
    kind: Bound
    params: Dict[str, ParamSpec] = field(default_factory=dict)

    def tunable(self) -> List[str]:
        return [n for n, s in self.params.items() if s.tunable]

    def values(self) -> Dict[str, float]:
        return {n: s.value for n, s in self.params.items()}

    def with_values(self, values: Dict[str, float]) -> "BoundOp":
        params = dict(self.params)
        for n, v in values.items():
            if n in params:
                old = params[n]
                params[n] = ParamSpec(float(v), old.tunable, old.bounds)
        return BoundOp(self.kind, params)

    def render(self) -> str:
        if not self.params:
            return self.kind.value
        return "%s(%s)" % (self.kind.value, ",".join(self.params))


class Stochastic(Enum):
    NONE = "none"
    PERP_JITTER = "perp_jitter"


class Base(Enum):
    ADD_VELOCITY = "add_velocity"
    VELOCITY_CMD = "velocity_cmd"
    ACCEL = "accel"


class Measures(Enum):
    CENTRE_TO_CENTRE = "centre_to_centre"
    CENTRE_TO_SURFACE = "centre_to_surface"
    SURFACE_TO_SURFACE = "surface_to_surface"

    def offset(self, other_radius: float, self_radius: float) -> float:
        if self is Measures.CENTRE_TO_CENTRE:
            return float(other_radius) + float(self_radius)
        if self is Measures.CENTRE_TO_SURFACE:
            return float(self_radius)
        return 0.0


@dataclass
class Term:
    name: str
    role: Role
    source: Source
    membership: Membership
    signal: Signal
    direction: Direction
    magnitude: Expr
    measures: Measures = Measures.CENTRE_TO_CENTRE
    anchor: Anchor = Anchor.SELF
    activation: Activation = field(default_factory=Activation)
    aggregator: Aggregator = Aggregator.SUM
    bound: Optional[BoundOp] = None
    stochastic: Stochastic = Stochastic.NONE
    weight: Optional[Expr] = None
    signal_speed: Optional[str] = None

    def tunable_params(self) -> List[str]:
        out: List[str] = []
        if self.weight is not None:
            out += self.weight.tunable_params()
        out += self.magnitude.tunable_params()
        if self.bound is not None:
            out += self.bound.tunable()
        if self.activation.threshold is not None:
            out += self.activation.threshold.tunable_params()
        return out

    def _exprs(self) -> List[Expr]:
        es = [self.magnitude]
        if self.weight is not None:
            es.append(self.weight)
        if self.activation.threshold is not None:
            es.append(self.activation.threshold)
        return es

    def param_values(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for e in self._exprs():
            for n, v in e.values().items():
                out.setdefault(n, v)
        if self.bound is not None:
            for n, v in self.bound.values().items():
                out.setdefault(n, v)
        return out

    def with_values(self, values: Dict[str, float]) -> "Term":
        t = copy.deepcopy(self)
        t.magnitude = self.magnitude.with_values(values)
        if self.weight is not None:
            t.weight = self.weight.with_values(values)
        if self.activation.threshold is not None:
            t.activation = Activation(self.activation.kind,
                                      self.activation.threshold.with_values(values),
                                      self.activation.feature)
        if self.bound is not None:
            t.bound = self.bound.with_values(values)
        return t


@dataclass
class Arbitration:
    role_rule: str
    planner: str
    target_selector: Dict[str, str]
    transition: Optional[str] = None
    state: Tuple[str, ...] = ()


@dataclass
class Algorithm:
    target_key: str
    roles: Tuple[Role, ...]
    terms: List[Term]
    base: Base
    global_bound: Tuple[BoundOp, ...] = ()
    arbitration: Optional[Arbitration] = None

    def tunable_params(self) -> List[str]:
        out: List[str] = []
        seen = set()
        pools = [t.tunable_params() for t in self.terms]
        pools += [b.tunable() for b in self.global_bound]
        for pool in pools:
            for n in pool:
                if n not in seen:
                    seen.add(n)
                    out.append(n)
        return out

    def param_values(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for t in self.terms:
            for n, v in t.param_values().items():
                out.setdefault(n, v)
        for b in self.global_bound:
            for n, v in b.values().items():
                out.setdefault(n, v)
        return out

    def tunable_values(self) -> Dict[str, float]:
        names = set(self.tunable_params())
        return {n: v for n, v in self.param_values().items() if n in names}

    def with_values(self, values: Dict[str, float]) -> "Algorithm":
        return Algorithm(
            target_key=self.target_key, roles=self.roles,
            terms=[t.with_values(values) for t in self.terms],
            base=self.base,
            global_bound=tuple(b.with_values(values) for b in self.global_bound),
            arbitration=self.arbitration)

    def copy(self) -> "Algorithm":
        return copy.deepcopy(self)

    def replace_term(self, index: int, term: "Term") -> "Algorithm":
        a = self.copy()
        a.terms[index] = term
        return a

    def signature(self) -> str:
        parts = [self.target_key, self.base.value,
                 "|".join(b.render() for b in self.global_bound)]
        for t in self.terms:
            m = t.membership
            w = t.weight.render() if t.weight is not None else "1"
            act = t.activation.kind.value
            if t.activation.threshold is not None:
                act += "[%s]" % t.activation.threshold.render()
            parts.append("%s:%s:%s:%s:%s:%s:agg=%s:bnd=%s:sto=%s:topo=%s:gate=%s:w=%s:m=%s:th=%s"
                         % (t.name, t.role.value, t.source.value, t.signal.value,
                            t.direction.value, t.anchor.value, t.aggregator.value,
                            t.bound.render() if t.bound else "none",
                            t.stochastic.value, m.topology, m.gate, w,
                            t.magnitude.render(), act))
        pv = self.param_values()
        parts.append("params=" + ",".join("%s=%.6g" % (n, pv[n]) for n in sorted(pv)))
        return "\n".join(parts)

    def summary(self) -> str:
        lines = ["%s  roles=%s  base=%s  bound=%s  arbitration=%s"
                 % (self.target_key, [r.value for r in self.roles],
                    self.base.value, [b.render() for b in self.global_bound] or ["none"],
                    "yes" if self.arbitration else "none")]
        for t in self.terms:
            m = t.membership
            gene = "  [topo=%s gate=%s]" % (m.topology, m.gate) if m.mutable else ""
            anc = "" if t.anchor is Anchor.SELF else " @%s" % t.anchor.value
            act = t.activation.kind.value
            if t.activation.threshold is not None:
                act += "(%s)" % t.activation.threshold.render()
            shape = t.magnitude.render()
            mag = "%s * (%s)" % (t.weight.render(), shape) if t.weight is not None else shape
            lines.append(
                "  %-16s %-8s <- %-9s dir=%-10s agg=%-6s act=%-16s%s%s\n"
                "        mag = %s"
                % (t.name, t.role.value, t.source.value, t.direction.value,
                   t.aggregator.value, act, anc, gene, mag))
        return "\n".join(lines)


IDENTITY_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "configs", "identity"))


def _identity_path(target_key: str) -> str:
    return os.path.join(IDENTITY_DIR, "%s.json" % target_key)


def identity(target_key: str) -> "Algorithm":
    from .serialize import load_algorithm
    path = _identity_path(target_key)
    if not os.path.exists(path):
        raise KeyError("no algorithm representation for %r (have: %s)"
                       % (target_key, ", ".join(names())))
    return load_algorithm(path)


def names() -> List[str]:
    if not os.path.isdir(IDENTITY_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(IDENTITY_DIR)
        if f.endswith(".json") and "." not in os.path.splitext(f)[0])


def variants(target_key: str) -> List[str]:
    if not os.path.isdir(IDENTITY_DIR):
        return []
    prefix = target_key + "."
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(IDENTITY_DIR)
        if f.endswith(".json") and os.path.splitext(f)[0].startswith(prefix))


def operator_families() -> Dict[str, List[str]]:
    return {
        "part1_relation": ["cohesion_floor*(add regroup coupling)",
                           "topology(knn/cap)", "gate(anomaly/quorum)",
                           "remove_coupling", "reassign_role"],
        "part2_expression": ["saturate*", "ceiling*", "soften*", "attenuate*",
                             "add_term*", "replace*", "graft*",
                             "widen_threshold*(activation deadband)",
                             "redirect(PERP/symmetry-break)", "swap_feature"],
        "part3_policy": ["robustify*(mean->median/trimmed/huber)",
                        "inject_stochastic*(perp jitter symmetry-break)",
                        "clamp_speed", "clamp_accel", "renormalize"],
        "part4_arbitration": ["stagnation_timeout", "mission_anchor",
                             "waypoint_guard", "planner_swap", "lookahead"],
    }


def _demo() -> None:
    for key in names():
        print(identity(key).summary())
        print()


if __name__ == "__main__":
    _demo()
