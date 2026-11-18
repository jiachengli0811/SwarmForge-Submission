from __future__ import annotations

import dataclasses
import json
import math
import re
from typing import Optional

from .classify import AlgorithmIR, TermIR

FEATURES = ("dist", "clearance", "rel_speed", "shill_speed", "axis")


def to_json(algo: AlgorithmIR) -> str:
    return json.dumps(dataclasses.asdict(algo), indent=2)


def summary(algo: AlgorithmIR) -> str:
    lines = ["%s  roles=%s  base=%s  bound=%s  sink=%s"
             % (algo.target_key, list(algo.roles), algo.base,
                [_render_bound_ir(b) for b in algo.global_bounds] or ["none"],
                algo.sink)]
    for t in algo.terms:
        act = t.activation.kind
        if t.activation.threshold:
            act += "(%s %s)" % (t.activation.feature or "", t.activation.threshold)
        mem = []
        if t.membership.range_param:
            mem.append("range=%s" % t.membership.range_param)
        if t.membership.fov_param:
            mem.append("fov=%s" % t.membership.fov_param)
        if t.membership.topology:
            mem.append("topo=%s" % t.membership.topology)
        if t.membership.cap_param:
            mem.append("cap=%s" % t.membership.cap_param)
        mem_s = "  [%s]" % " ".join(mem) if mem else ""
        anc = "" if t.anchor == "self" else " @%s" % t.anchor
        gat = "  gates=%s" % ",".join(t.gates) if t.gates else ""
        mag = "%s * (%s)" % (t.weight, t.magnitude) if t.weight else t.magnitude
        bnd = "  bnd=%s" % _render_bound_ir(t.bound) if t.bound else ""
        lines.append(
            "  %-14s %-7s <- %-10s dir=%-9s agg=%-5s act=%-24s%s%s%s%s\n"
            "        mag = %s"
            % (t.name, t.role, "%s(%s)" % (t.source, t.source_set) if t.source_set
               and t.source_set != t.source else t.source,
               t.direction, t.aggregator, act, anc, mem_s, bnd, gat, mag))
    if algo.arbitration:
        a = algo.arbitration
        sel = ", ".join("%s->%s" % kv for kv in
                        sorted(a.get("target_selector", {}).items()))
        lines.append("  arbitration: role_rule=%s planner=%s transition=%s"
                     "  targets: %s"
                     % (a.get("role_rule"), a.get("planner"),
                        a.get("transition"), sel))
    if algo.params:
        lines.append("  params: " + ", ".join(
            "%s=%s%s" % (n, d["value"], "*" if d["tunable"] else "")
            for n, d in sorted(algo.params.items())))
    for n in algo.notes:
        lines.append("  note: %s" % n)
    return "\n".join(lines)


def _render_bound_ir(b) -> str:
    if b is None:
        return "none"
    ps = ",".join("%s=%s" % (k, v) for k, v in b.params.items())
    return "%s(%s)" % (b.kind, ps)


SELF_SIZE = "self_size"

REL_SPEED = "rel_speed"


def uses_clearance(algo: AlgorithmIR) -> bool:
    for t in algo.terms:
        if getattr(t, "measures", "") == "centre_to_surface":
            continue
        for text in (t.magnitude, t.weight, t.activation.threshold):
            if text and "clearance" in text:
                return True
    return False


def to_swarmff(algo: AlgorithmIR, self_size: float = 0.0, *, strict: bool = False):
    import sympy as sp
    try:
        from swarmforge.algorithm.algorithm import (
            Activation, ActivationKind, Aggregator, Algorithm, Arbitration,
            Base, Bound, BoundOp, Direction, Measures, Membership, Role,
            Signal, Source, Term)
        from swarmforge.algorithm.expr import Expr, ParamSpec
    except ImportError:
        from swarmff.mutate.algorithm import (
            Activation, ActivationKind, Aggregator, Algorithm, Arbitration,
            Base, Bound, BoundOp, Direction, Measures, Membership, Role,
            Signal, Source, Term)
        from swarmff.mutate.expr import Expr, ParamSpec

    def pspec(name: str) -> ParamSpec:
        d = algo.params.get(name, {})
        v = d.get("value")
        if strict and v is None:
            raise ValueError(f"record export needs a value for parameter {name!r}")
        if strict and not math.isfinite(float(v)):
            raise ValueError(f"record export needs a finite value for parameter {name!r}")
        return ParamSpec(1.0 if v is None else float(v), bool(d.get("tunable", True)))

    if strict:
        if not algo.terms:
            raise ValueError("record export requires at least one classified term")
        for t in algo.terms:
            if t.gates:
                raise ValueError(f"term {t.name!r} has mode gates that v1 cannot serialize: {t.gates}")
            if t.activation.kind not in ("always", "below", "above"):
                raise ValueError(f"term {t.name!r} has an unsupported activation")
            if any(note.startswith("opaque ") or "unclassified guard" in note for note in t.notes):
                raise ValueError(f"term {t.name!r} has unresolved source expressions: {t.notes}")
        if any("dropped " in note or "no command sink" in note for note in algo.notes):
            raise ValueError(f"record export would omit source operations: {algo.notes}")

    size_value = float(algo.params.get(SELF_SIZE, {}).get("value", self_size)
                       if SELF_SIZE in algo.params else self_size)
    _surface = sp.Symbol("dist") - sp.Symbol(SELF_SIZE)

    def wrap(text: Optional[str], surface: bool = False) -> Optional[Expr]:
        if not text:
            return None
        called = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", text))
        names = set(re.findall(r"[A-Za-z_]\w*", text)) - called - {"True", "False", "true", "false"}
        e = sp.sympify(text, locals={n: sp.Symbol(n) for n in names})
        if strict:
            from sympy.core.function import AppliedUndef
            if e.atoms(AppliedUndef):
                raise ValueError(f"record export has unresolved functions in {text!r}")
        e = e.subs(sp.Symbol("clearance"),
                   sp.Symbol("dist") if surface else _surface)
        params = {}
        for s in e.free_symbols:
            if s.name == SELF_SIZE:
                params[s.name] = ParamSpec(size_value, False)
            elif s.name not in FEATURES:
                params[s.name] = pspec(s.name)
        return Expr(e, params)

    def bound_op(b) -> BoundOp:
        kind = {"clamp_speed": Bound.CLAMP_SPEED, "clamp_accel": Bound.CLAMP_ACCEL,
                "renormalize": Bound.RENORMALIZE}[b.kind]
        if strict and (not b.params or any(v is None or not math.isfinite(float(v)) for v in b.params.values())):
            raise ValueError(f"record export has an unresolved {b.kind} bound")
        return BoundOp(kind, {n: ParamSpec(1.0 if v is None else float(v),
                                           bool(algo.params.get(n, {}).get("tunable", True)))
                              for n, v in b.params.items()})

    terms = []
    for t in algo.terms:
        try:
            direction = Direction(t.direction)
            source = Source(t.source)
        except ValueError:
            if strict:
                raise ValueError(f"term {t.name!r} has unclassified direction/source: {t.direction}/{t.source}") from None
            continue
        membership = Membership(range_param=t.membership.range_param,
                                fov_param=t.membership.fov_param,
                                cap_param=t.membership.cap_param,
                                topology=t.membership.topology or "global")
        surface = getattr(t, "measures", "") == "centre_to_surface"
        act = Activation()
        if t.activation.kind in ("below", "above"):
            threshold = t.activation.threshold
            if t.activation.feature == "clearance" and threshold and not surface:
                threshold = "(%s) + %s" % (threshold, SELF_SIZE)
            act = Activation(ActivationKind(t.activation.kind),
                             wrap(threshold, surface),
                             feature=t.activation.feature)
        magnitude = wrap(t.magnitude, surface) or wrap("1")
        direction, magnitude = _explicit_velocity(direction, magnitude,
                                                  t.signal, Direction, Expr)
        speed_param = getattr(t, "signal_speed", None)
        if speed_param and speed_param not in magnitude.params:
            magnitude.params[speed_param] = pspec(speed_param)
        terms.append(Term(
            t.name, Role(t.role), source, membership,
            Signal(t.signal), direction, magnitude,
            measures=Measures(getattr(t, "measures", "centre_to_centre")),
            anchor=_anchor(t.anchor),
            activation=act,
            aggregator=Aggregator(t.aggregator),
            bound=bound_op(t.bound) if t.bound else None,
            weight=wrap(t.weight, surface),
            signal_speed=getattr(t, "signal_speed", None)))
    arb = None
    if algo.arbitration:
        a = algo.arbitration
        arb = Arbitration(role_rule=a.get("role_rule") or "",
                          planner=a.get("planner") or "",
                          target_selector=dict(a.get("target_selector") or {}),
                          transition=a.get("transition"),
                          state=tuple(a.get("state") or ()))
    return Algorithm(
        target_key=algo.target_key,
        roles=tuple(Role(r) for r in algo.roles),
        terms=terms,
        base=Base(algo.base),
        global_bound=tuple(bound_op(b) for b in algo.global_bounds),
        arbitration=arb)


def _explicit_velocity(direction, magnitude, signal: str, Direction, Expr):
    import sympy as sp

    if signal != "velocity" or direction is not Direction.MATCH:
        return direction, magnitude
    if magnitude is None:
        return direction, magnitude
    speed = sp.Symbol(REL_SPEED)
    if speed in magnitude.sympy.free_symbols:
        return direction, magnitude
    return Direction.FRICTION, Expr(sp.simplify(speed * magnitude.sympy),
                                    dict(magnitude.params))


def _anchor(name: str):
    try:
        from swarmforge.algorithm.algorithm import Anchor
    except ImportError:
        from swarmff.mutate.algorithm import Anchor
    return Anchor(name)
