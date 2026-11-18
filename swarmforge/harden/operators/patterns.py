from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

from ...solve.record import _is_size
from ..document import (Document, Param, Site, gate_of, namespace, params,
                        read_site, set_gate, set_param, sites,
                        tunable_params, write_site)
from .base import PATTERN, Intent, Mutation, Operator, OperatorContext
from .calibrate import Calibration
from .tree import broken, clean, laws_of, literal, walk

DIST, CLEARANCE, REL_SPEED = (sp.Symbol("dist"), sp.Symbol("clearance"),
                              sp.Symbol("rel_speed"))
EVERY = ("variables", "sign", "switching", "magnitude")


def _keeps(*changes: str) -> Tuple[str, ...]:
    return tuple(f for f in EVERY if f not in changes)


def _term(document: Document, law: str) -> Dict[str, Any]:
    return next(t for t in document["terms"] if t["name"] == law)


def _magnitude(document: Document, law: str) -> sp.Expr:
    return read_site(document, Site(law, "magnitude")).expr


def _reads_distance(expr: Any) -> bool:
    return bool({DIST, CLEARANCE} & expr.free_symbols)


def _reads_input(expr: Any) -> bool:
    return bool({DIST, CLEARANCE, REL_SPEED, sp.Symbol("shill_speed")}
                & expr.free_symbols)


def _write(document: Document, law: str, slot: str, expr: Any,
           cal: Calibration, params_: Optional[Dict[str, Param]] = None
           ) -> Optional[Document]:
    site = Site(law, slot)
    old = read_site(document, site).expr
    tree = clean(sp.sympify(expr))
    if tree == old:
        return None
    problem = broken(tree, cal.namespace)
    if problem:
        raise ValueError(problem)
    return write_site(document, site, tree, params_)


def _peak(cal: Calibration, law: str, expr: Any) -> Optional[float]:
    values = cal.values(law, "magnitude", expr)
    if values is None:
        return None
    peak = float(np.max(np.abs(values)))
    return peak if peak > 0.0 else None


def _sense(cal: Calibration, law: str, expr: Any) -> int:
    mean = cal.mean(law, "magnitude", expr)
    return -1 if mean is not None and mean < 0.0 else 1


class Pattern(Operator):
    family = PATTERN
    key = "k"
    grid: Tuple[Any, ...] = ()
    acts: Optional[int] = 0

    def direction(self, mutation: Mutation) -> int:
        if self.acts is not None:
            return self.acts
        factor = float(mutation.args.get(self.key, 1.0))
        return 1 if factor > 1.0 else -1 if factor < 1.0 else 0

    def applies(self, document: Document, law: str) -> bool:
        return True

    def site(self, law: str) -> str:
        return "%s/magnitude" % law

    def recipes(self, document: Document,
                context: OperatorContext) -> List[Mutation]:
        return [Mutation(self.name, site=self.site(law),
                         args={self.key: value})
                for law in laws_of(document) if self.applies(document, law)
                for value in self.grid]

    def transform(self, document: Document, law: str, value: Any,
                  cal: Calibration) -> Optional[Document]:
        raise NotImplementedError

    def apply(self, document: Document, mutation: Mutation,
              context: OperatorContext) -> Optional[Document]:
        cal = Calibration.of(document, context)
        return self.transform(document, mutation.law, mutation.args[self.key],
                              cal)


GAIN_BAND = (0.125, 8.0)
GAIN_STEPS = (0.5, 0.8, 1.25, 2.0)


class ScaleGain(Pattern):
    acts = None
    name = "scale_gain"
    description = "a tunable parameter's value times k, everywhere it is written"
    intent = Intent(("magnitude", "switching"),
                    ("variables", "sign"),
                    "the old loop's scale_gain: the record's tunable flags are "
                    "the contract of what may be re-valued")
    grid = GAIN_STEPS

    def recipes(self, document, context):
        original = namespace(context.original)
        out = []
        for name in tunable_params(document):
            value = namespace(document)[name]
            base = original.get(name) or value
            if not value or not base:
                continue
            for k in self.grid:
                if GAIN_BAND[0] <= value * k / base <= GAIN_BAND[1]:
                    out.append(Mutation(self.name, param=name,
                                        args={"k": k}))
        return out

    def apply(self, document, mutation, context):
        name = mutation.param
        value = namespace(document).get(name)
        if not value:
            return None
        new = float(literal(value * mutation.args["k"]))
        for _location, spec in params(document).get(name, ()):
            if spec.bounds and not spec.bounds[0] <= new <= spec.bounds[1]:
                return None
        return set_param(document, name, new)


def _source_gains(document: Document) -> Dict[str, Tuple[str, ...]]:
    tunable = set(tunable_params(document))
    out: Dict[str, List[str]] = {}
    for term in document["terms"]:
        weight = term.get("weight") or {}
        for name in sorted((weight.get("p") or {})):
            if name in tunable:
                names = out.setdefault(str(term.get("source")), [])
                if name not in names:
                    names.append(name)
    return {k: tuple(v) for k, v in out.items() if len(v) >= 2}


class ScaleSource(Pattern):
    acts = None
    name = "scale_source"
    description = "every tunable gain of one source's laws times k, together"
    intent = Intent(("magnitude", "switching"),
                    ("variables", "sign"),
                    "the old loop's scale_source: the balance between one "
                    "source's laws kept while their sum moves")
    grid = GAIN_STEPS

    def recipes(self, document, context):
        return [Mutation(self.name, param=",".join(names),
                         args={"k": k, "source": source})
                for source, names in sorted(_source_gains(document).items())
                for k in self.grid]

    def apply(self, document, mutation, context):
        out = document
        for name in mutation.param.split(","):
            value = namespace(out).get(name)
            if not value:
                return None
            out = set_param(out, name,
                            float(literal(value * mutation.args["k"])))
        return out


class ScaleLaw(Pattern):
    acts = None
    name = "scale_law"
    description = "the law's weight times k -- a law whose gain is fixed too"
    intent = Intent(("magnitude", "switching"),
                    _keeps("magnitude", "switching"),
                    "a law's authority is its weight; the record's fixed "
                    "parameters keep their values, the law its shape")
    grid = GAIN_STEPS

    def applies(self, document, law):
        return any(s.law == law and s.slot == "weight"
                   for s in sites(document))

    def site(self, law):
        return "%s/weight" % law

    def transform(self, document, law, k, cal):
        weight = read_site(document, Site(law, "weight")).expr
        return _write(document, law, "weight", literal(k) * weight, cal)


def _split_size(threshold: sp.Expr) -> Tuple[sp.Expr, sp.Expr]:
    if isinstance(threshold, sp.Add):
        sizes = [a for a in threshold.args
                 if isinstance(a, sp.Symbol) and _is_size(a.name)]
        if sizes:
            core = sp.Add(*[a for a in threshold.args if a not in sizes])
            return core, sp.Add(*sizes)
    return threshold, sp.Integer(0)


def _distance_gate(document: Document, law: str) -> bool:
    gate = gate_of(document, law)
    if not gate.gated or gate.threshold is None:
        return False
    if gate.feature not in (None, "dist", "clearance"):
        return False
    return not any(s.name in ("dist", "clearance", "rel_speed",
                              "shill_speed")
                   for s in gate.threshold.free_symbols)


class MoveCutoff(Pattern):
    acts = None
    name = "move_cutoff"
    description = "the gate's distance threshold times k"
    intent = Intent(("switching",),
                    ("variables", "sign"),
                    "the old loop's adjust_threshold: where a barrier starts "
                    "to act is its range")
    grid = (0.5, 0.75, 1.25, 1.5, 2.0)

    def applies(self, document, law):
        return (_distance_gate(document, law)
                and _split_size(gate_of(document, law).threshold)[0] != 0)

    def site(self, law):
        return "%s/gate" % law

    def transform(self, document, law, k, cal):
        gate = gate_of(document, law)
        core, size = _split_size(gate.threshold)
        return set_gate(document, law, gate.kind, gate.feature,
                        literal(k) * core + size, gate.params)


class Dilate(Pattern):
    acts = None
    name = "dilate"
    description = ("the law stretched by s along its distance about contact, "
                   "its gate with it: the same shape at another range")
    intent = Intent(("switching",),
                    ("variables", "sign"),
                    "an interaction's length scale (vasarhelyi's r0, a "
                    "barrier radius) sets its range and leaves its shape")
    key = "s"
    grid = (0.75, 1.25, 1.5)

    def applies(self, document, law):
        return DIST in _magnitude(document, law).free_symbols

    def site(self, law):
        return law

    def transform(self, document, law, s, cal):
        floor = cal.span(law, "dist", "magnitude", extent=True)[0]
        f = literal(floor)
        magnitude = _magnitude(document, law)
        stretched = magnitude.subs(DIST, f + (DIST - f) / literal(s))
        out = _write(document, law, "magnitude", stretched, cal)
        if out is None:
            return None
        if not _distance_gate(document, law):
            return out
        gate = gate_of(document, law)
        core, size = _split_size(gate.threshold)
        feature = "clearance" if size != 0 or gate.feature == "clearance" \
            else "dist"
        base = cal.span(law, feature, "gate", extent=True)[0]
        value = float(cal.flown(law, "threshold", core))
        new = literal(base + s * (value - base))
        return set_gate(out, law, gate.kind, gate.feature, new + size,
                        gate.params)


def _inverse_powers(expr: sp.Expr) -> List[sp.Pow]:
    return [node for _p, node, _parent, _i in walk(expr)
            if isinstance(node, sp.Pow) and node.args[1].is_number
            and node.args[1] < 0 and _reads_distance(node.args[0])]


class SoftenCore(Pattern):
    acts = -1
    name = "soften_core"
    description = ("every inverse power of a distance given a soft core: "
                   "b**-k -> (b**2 + e**2)**(-k/2), e a share of b at contact")
    intent = Intent(("switching", "magnitude"),
                    ("variables", "sign"),
                    "the old loop's soften; soft-core potentials, finite "
                    "at contact")
    key = "share"
    grid = (0.5, 1.0, 2.0)

    def applies(self, document, law):
        return bool(_inverse_powers(_magnitude(document, law)))

    def transform(self, document, law, share, cal):
        magnitude = _magnitude(document, law)
        swap = {}
        for power in _inverse_powers(magnitude):
            base, k = power.args
            values = cal.values(law, "magnitude", base)
            if values is None:
                continue
            gap = float(np.min(np.abs(values))) or float(
                np.mean(np.abs(values)))
            epsilon = literal(share * gap)
            swap[power] = (base ** 2 + epsilon ** 2) ** (k / 2)
        if not swap:
            return None
        return _write(document, law, "magnitude", magnitude.xreplace(swap),
                      cal)


class Stiffen(Pattern):
    acts = 1
    name = "stiffen"
    description = ("f -> f (1 + kappa f / s), s the law's mean size signed "
                   "as the law: sharper where it is strong")
    intent = Intent(("switching", "magnitude"),
                    ("variables", "sign"),
                    "the old loop's sharpen -- vasarhelyi's best single step "
                    "(kappa 16 on separation); a separation turned convex came "
                    "with every improved vasarhelyi candidate")
    key = "kappa"
    grid = (1.0, 4.0, 16.0)

    def applies(self, document, law):
        return _reads_distance(_magnitude(document, law))

    def transform(self, document, law, kappa, cal):
        f = _magnitude(document, law)
        size = cal.size(law, "magnitude", f)
        if size is None:
            return None
        factor = 1 + literal(_sense(cal, law, f) * kappa / size) * f
        values = cal.values(law, "magnitude", factor)
        if values is None or np.any(values < 0.0):
            return None
        return _write(document, law, "magnitude", f * factor, cal)


class Saturate(Pattern):
    acts = -1
    name = "saturate"
    description = "f -> C tanh(f / C), C a share of the law's peak"
    intent = Intent(("switching", "magnitude"),
                    ("variables", "sign"),
                    "the old loop's saturate: a smooth ceiling")
    key = "share"
    grid = (0.25, 0.5, 0.75)

    def applies(self, document, law):
        return _reads_input(_magnitude(document, law))

    def transform(self, document, law, share, cal):
        f = _magnitude(document, law)
        peak = _peak(cal, law, f)
        if peak is None:
            return None
        c = literal(share * peak)
        return _write(document, law, "magnitude", c * sp.tanh(f / c), cal)


class Cap(Pattern):
    acts = -1
    name = "cap"
    description = "f -> Min(f, C) (Max(f, -C) for a negative law)"
    intent = Intent(("switching", "magnitude"),
                    ("variables", "sign"),
                    "the old loop's ceiling: a hard limit on a law's push")
    key = "share"
    grid = (0.25, 0.5, 0.75)

    def transform(self, document, law, share, cal):
        f = _magnitude(document, law)
        peak = _peak(cal, law, f)
        if peak is None:
            return None
        c = literal(share * peak)
        capped = (sp.Min(f, c) if _sense(cal, law, f) > 0
                  else sp.Max(f, -c))
        return _write(document, law, "magnitude", capped, cal)


class Floor(Pattern):
    acts = 1
    name = "floor"
    description = "f -> Max(f, c): a strength the law never drops below"
    intent = Intent(("switching", "magnitude"),
                    ("variables", "sign"),
                    "the constant goal pull adaptive's and gradient's best "
                    "candidates moved toward")
    key = "share"
    grid = (0.1, 0.25, 0.5)

    def applies(self, document, law):
        return _reads_input(_magnitude(document, law))

    def transform(self, document, law, share, cal):
        f = _magnitude(document, law)
        peak = _peak(cal, law, f)
        if peak is None:
            return None
        c = literal(share * peak)
        floored = (sp.Max(f, c) if _sense(cal, law, f) > 0
                   else sp.Min(f, -c))
        return _write(document, law, "magnitude", floored, cal)


class Flatten(Pattern):
    name = "flatten"
    description = "f -> its mean over the law's span: a constant push or pull"
    intent = Intent(("variables", "magnitude"),
                    ("sign", "switching"),
                    "vasarhelyi's goal pull is a constant by design; the "
                    "adaptive and gradient fixes made theirs nearly one")
    key = "share"
    grid = (1.0,)

    def applies(self, document, law):
        return _reads_input(_magnitude(document, law))

    def transform(self, document, law, share, cal):
        f = _magnitude(document, law)
        mean = cal.mean(law, "magnitude", f)
        if mean is None or mean == 0.0:
            return None
        return _write(document, law, "magnitude", literal(share * mean), cal)


class Attenuate(Pattern):
    acts = -1
    name = "attenuate"
    description = "f -> f exp(-(x - x0)/lambda), lambda a share of its span"
    intent = Intent(("switching", "magnitude"),
                    ("variables", "sign"),
                    "the old loop's attenuate: a finite effective range")
    key = "share"
    grid = (0.5, 1.0, 2.0)

    def applies(self, document, law):
        return DIST in _magnitude(document, law).free_symbols

    def transform(self, document, law, share, cal):
        lo, hi = cal.span(law, "dist", "magnitude")
        if hi <= lo:
            return None
        f = _magnitude(document, law)
        decay = sp.exp(-(DIST - literal(lo)) / literal(share * (hi - lo)))
        return _write(document, law, "magnitude", f * decay, cal)


FAMILIES = {
    "inverse": lambda g, r: r / g,
    "inverse_square": lambda g, r: (r / g) ** 2,
    "log_barrier": lambda g, r: sp.Max(0, sp.log(r / g)),
    "linear_ramp": lambda g, r: sp.Max(0, r - g),
    "quadratic_ramp": lambda g, r: sp.Max(0, r - g) ** 2,
    "exponential": lambda g, r: sp.exp(-g / r),
}


def _gap(expr: sp.Expr) -> sp.Expr:
    for _p, node, _parent, _i in walk(expr):
        if isinstance(node, sp.Add) and DIST in node.free_symbols \
                and sp.diff(node, DIST) == 1 and not (
                    node - DIST).free_symbols & {DIST, CLEARANCE,
                                                 REL_SPEED}:
            return node
    return DIST


class TransformFamily(Pattern):
    name = "transform_family"
    description = ("a barrier rewritten as another family over its own gap "
                   "and length, matched to its mean strength")
    intent = Intent(("magnitude", "switching"),
                    ("variables", "sign"),
                    "the old loop's transform_family (SHAPE_FAMILIES), matched "
                    "to the law's mean strength")
    key = "family"
    grid = tuple(FAMILIES)

    def applies(self, document, law):
        f = _magnitude(document, law)
        return f.free_symbols & {DIST, CLEARANCE, REL_SPEED} == {DIST}

    def transform(self, document, law, family, cal):
        f = _magnitude(document, law)
        near, far = (cal.values(law, "magnitude", f.subs(DIST, literal(x)))
                     for x in cal.span(law, "dist", "magnitude"))
        if near is None or far is None or not abs(float(near[0])) > abs(
                float(far[0])):
            return None
        gap = _gap(f)
        hi = cal.span(law, "dist", "magnitude")[1]
        length = cal.values(law, "magnitude", gap.subs(DIST, literal(hi)))
        if length is None or float(length[0]) <= 0.0:
            return None
        shape = FAMILIES[family](gap, literal(float(length[0])))
        matched = cal.match(law, "magnitude", f, shape)
        if matched is None:
            return None
        return _write(document, law, "magnitude", matched, cal)


class ShiftSurface(Pattern):
    acts = 1
    name = "shift_surface"
    description = ("dist -> dist - share*r, r the other body's radius: the "
                   "law measured toward its surface")
    intent = Intent(("switching", "magnitude"),
                    ("variables", "sign"),
                    "the old loop's shift_input: a centre-to-centre law is "
                    "short by the other body's radius (all of it puts a "
                    "barrier's pole on contact itself)")
    key = "share"
    grid = (0.5, 0.9)

    def applies(self, document, law):
        term = _term(document, law)
        return (term.get("source") in ("agents", "obstacles", "bodies")
                and (term.get("measures") or "centre_to_centre")
                == "centre_to_centre"
                and DIST in _magnitude(document, law).free_symbols)

    def transform(self, document, law, share, cal):
        physics = cal.context.config.physics
        source = _term(document, law)["source"]
        radii = []
        if source in ("agents", "bodies"):
            radii.append(physics.agent_radius)
        state = cal.context.config.state
        if source in ("obstacles", "bodies"):
            for kind in (state.static, state.dynamic):
                if kind is not None:
                    radii.append(kind.radius.lo)
        r = share * min(radii) if radii else 0.0
        if r <= 0.0:
            return None
        name = "%s_surface" % law
        f = _magnitude(document, law)
        return _write(document, law, "magnitude",
                      f.subs(DIST, DIST - sp.Symbol(name)), cal,
                      {name: Param(float(literal(r)), True)})


class FadeOnset(Pattern):
    acts = -1
    name = "fade_onset"
    description = ("f -> f - f(cutoff): a law cut off at its gate or its "
                   "sensing range starts from zero there")
    intent = Intent(("switching",),
                    ("variables", "sign"),
                    "abrupt onsets the facets found: boids' gates switch on "
                    "mid-push, gradient's barrier ends at 37x the command; "
                    "vasarhelyi's and adaptive's laws fade in by design")
    key = "share"
    grid = (1.0,)

    def applies(self, document, law):
        f = _magnitude(document, law)
        return f.free_symbols & {DIST, CLEARANCE, REL_SPEED} == {DIST}

    def _cutoff(self, document, law, cal) -> Optional[float]:
        if _distance_gate(document, law):
            gate = gate_of(document, law)
            if gate.kind != "below":
                return None
            core, size = _split_size(gate.threshold)
            value = float(cal.flown(law, "threshold", core + size))
            if size == 0 and gate.feature == "clearance":
                offset = cal.context.domains[law].offset or 0.0
                value += offset
            return value
        sensing = cal.context.domains[law].sensing
        return None if sensing is None else float(sensing)

    def transform(self, document, law, share, cal):
        cutoff = self._cutoff(document, law, cal)
        if cutoff is None:
            return None
        f = _magnitude(document, law)
        at = cal.values(law, "magnitude", f.subs(DIST, literal(cutoff)))
        if at is None or float(at[0]) == 0.0:
            return None
        faded = f - literal(share * float(at[0]))
        before = cal.values(law, "magnitude", f)
        after = cal.values(law, "magnitude", faded)
        if before is None or after is None or np.any(
                np.sign(np.mean(before)) * after
                < -1e-9 * float(np.max(np.abs(before)))):
            return None
        return _write(document, law, "magnitude", faded, cal)


class Anticipate(Pattern):
    acts = 1
    name = "anticipate"
    description = ("dist -> dist - tau * rel_speed: the law reads the gap "
                   "it will have in tau")
    intent = Intent(("variables", "switching", "magnitude"),
                    ("sign",),
                    "time-to-contact and velocity-obstacle avoidance; the "
                    "speed-dependent reach of vasarhelyi's braking curves")
    key = "ticks"
    grid = (1, 2, 4)

    def applies(self, document, law):
        term = _term(document, law)
        return (term.get("direction") in ("away_unit", "away_raw", "around")
                and DIST in _magnitude(document, law).free_symbols)

    def transform(self, document, law, ticks, cal):
        tau = literal(ticks * cal.context.config.physics.dt)
        f = _magnitude(document, law)
        return _write(document, law, "magnitude",
                      f.subs(DIST, DIST - tau * REL_SPEED), cal)


class SpeedGain(Pattern):
    acts = 1
    name = "speed_gain"
    description = "f -> f (1 + rel_speed / v): stronger when closing fast"
    intent = Intent(("variables", "switching", "magnitude"),
                    ("sign",),
                    "velocity-coupled interactions (Cucker-Smale) and "
                    "vasarhelyi's speed-scaled braking")
    key = "share"
    grid = (0.5, 1.0)

    def applies(self, document, law):
        term = _term(document, law)
        return (term.get("source") != "target"
                and REL_SPEED not in _magnitude(document, law).free_symbols)

    def transform(self, document, law, share, cal):
        top = cal.reference(law, "rel_speed")
        f = _magnitude(document, law)
        return _write(document, law, "magnitude",
                      f * (1 + REL_SPEED / literal(share * top)), cal)


class Rectify(Pattern):
    name = "rectify"
    description = "Max(f, 0) (Min for a negative law): never against itself"
    intent = Intent(("sign", "switching"),
                    ("variables",),
                    "the cohesion that repels near (vasarhelyi's dispersal "
                    "exploit) and the avoid laws that changed sign: a law "
                    "kept to its own side")
    key = "share"
    grid = (1.0,)

    def applies(self, document, law):
        return _reads_input(_magnitude(document, law))

    def transform(self, document, law, share, cal):
        f = _magnitude(document, law)
        values = cal.values(law, "magnitude", f)
        if values is None:
            return None
        peak = float(np.max(np.abs(values)))
        sense = _sense(cal, law, f)
        if not np.any(sense * values < -1e-9 * peak):
            return None
        return _write(document, law, "magnitude",
                      sp.Max(f, 0) if sense > 0 else sp.Min(f, 0), cal)


class ScaleShared(Pattern):
    name = "scale_shared"
    description = ("a parameter written in several sites -- a length the "
                   "laws share -- times k in every one of them")
    intent = Intent(("switching", "sign", "magnitude"),
                    ("variables",),
                    "the length scales laws share: vasarhelyi's r0 is the "
                    "separation's cutoff, the cohesion's onset and the zero of "
                    "their spring; gradient's barrier radius")
    grid = (0.8, 1.25)

    def recipes(self, document, context):
        shared = []
        for name, places in sorted(params(document).items()):
            slots = {location for location, _p in places
                     if location.split("/")[-1] in ("magnitude", "threshold",
                                                    "weight")}
            if len(slots) >= 2 and places[-1][1].value:
                shared.append(name)
        return [Mutation(self.name, param=name, args={"k": k})
                for name in shared for k in self.grid]

    def apply(self, document, mutation, context):
        cal = Calibration.of(document, context)
        symbol = sp.Symbol(mutation.param)
        k = literal(mutation.args["k"])
        out = document
        touched = False
        for site in sites(document):
            expr = read_site(out, site).expr
            if symbol not in expr.free_symbols:
                continue
            tree = clean(expr.subs(symbol, k * symbol))
            problem = broken(tree, cal.namespace)
            if problem:
                raise ValueError(problem)
            out = write_site(out, site, tree)
            touched = True
        return out if touched else None


class Around(Pattern):
    acts = 1
    name = "around"
    description = ("direction away_unit -> around: a constant push along the "
                   "nearest map body within reach of it, and the nearest map "
                   "body and peer read apart")
    intent = Intent((), ("variables", "sign", "switching", "magnitude"),
                    "behind a wall the pull and the push lie on one line, so "
                    "no magnitude leaves it: the size only decides between "
                    "touching and stalling. Not a facet: the four read a "
                    "law's scalar response, this is its direction")
    key = "around"
    grid = ((0.5, 1.75), (0.5, 2.5), (1.0, 1.75), (1.0, 2.5))

    def applies(self, document, law):
        term = _term(document, law)
        return (term.get("direction") == "away_unit"
                and term.get("source") in ("obstacles", "bodies"))

    def transform(self, document, law, value, cal):
        share, steps = value
        peak = _peak(cal, law, _magnitude(document, law))
        weight = cal.values(law, "weight",
                            read_site(document, Site(law, "weight")).expr)
        if peak is None or weight is None:
            return None
        physics = cal.context.config.physics
        out = copy.deepcopy(document)
        term = _term(out, law)
        term["direction"] = "around"
        term["around"] = [
            float(literal(share * peak * abs(float(weight.max())))),
            float(physics.agent_radius + steps * physics.plant.max_step)]
        return out


PATTERNS: Tuple[Pattern, ...] = (
    ScaleGain(), ScaleSource(), ScaleLaw(), MoveCutoff(), Dilate(),
    SoftenCore(), Stiffen(), Saturate(), Cap(), Floor(), Flatten(),
    Attenuate(), TransformFamily(), ShiftSurface(), FadeOnset(),
    Anticipate(), SpeedGain(), Rectify(), ScaleShared(), Around(),
)


__all__ = ["FAMILIES", "PATTERNS", "Pattern"]
