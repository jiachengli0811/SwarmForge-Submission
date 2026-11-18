from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .sexpr import (S, Add, AndS, Clamp, Cmp, Cond, Const, Count, Dom, Fn,
                    Idx, Mul, NotS, Obj, OrS, Pow, Prop, Reduce, Sym, Unk,
                    Vec, add, cond, fn, is_vec, is_zero, mul, neg, sub, subst, walk, memo_map)
from . import canonical as rules
from .loop import SELF

def _has_role(prop: S, role: str) -> bool:
    return isinstance(prop, Prop) and prop.role == role


def _source(domain: S, names) -> str:
    name = _root_collection(domain)
    return names.sources.get(name) or "unknown:%s" % _domain_name(domain)


@dataclass
class MembershipIR:
    range_param: Optional[str] = None
    fov_param: Optional[str] = None
    cap_param: Optional[str] = None
    topology: Optional[str] = None


@dataclass
class ActivationIR:
    kind: str = "always"
    threshold: Optional[str] = None
    feature: Optional[str] = None


@dataclass
class BoundIR:
    kind: str
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class TermIR:
    name: str
    role: str = "agent"
    source: str = "unknown"
    source_set: str = ""
    signal: str = "position"
    direction: str = "unknown"
    signal_speed: Optional[str] = None
    anchor: str = "self"
    aggregator: str = "sum"
    measures: str = "centre_to_centre"
    membership: MembershipIR = field(default_factory=MembershipIR)
    activation: ActivationIR = field(default_factory=ActivationIR)
    magnitude: str = "1"
    weight: Optional[str] = None
    bound: Optional[BoundIR] = None
    gates: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class AlgorithmIR:
    target_key: str
    roles: Tuple[str, ...] = ("agent",)
    base: str = "velocity_cmd"
    terms: List[TermIR] = field(default_factory=list)
    global_bounds: List[BoundIR] = field(default_factory=list)
    params: Dict[str, Dict] = field(default_factory=dict)
    arbitration: Optional[Dict] = None
    sink: str = ""
    notes: List[str] = field(default_factory=list)
    plant: Optional[Dict] = None


def _peel_clamps_shallow(v: S) -> S:
    while isinstance(v, Clamp):
        v = v.x
    return v


@dataclass
class Context:
    params: Dict[str, float]
    notes: List[str]
    names: object
    heap: Dict[S, S] = field(default_factory=dict)


def with_roles(expr: S, names) -> S:
    def tag(node):
        if isinstance(node, Prop):
            role = names.fields.get(node.name)
            if role is not None and node.role != role:
                return Prop(node.base, node.name, role)
        return None
    return memo_map(expr, tag)


def classify(loop, names, ctx: Context, target_key: str) -> AlgorithmIR:
    algo = AlgorithmIR(target_key=target_key)
    algo.sink = loop.desc
    if loop.sink is None:
        algo.notes.append("no command found" + (": " + loop.reason if loop.reason else ""))
        return algo
    used_params: Dict[str, None] = {}
    role = names.fields.get(loop.written) if loop.written else None
    stepping = loop.kind == "state" and role == "position" and loop.step is not None
    if stepping:
        algo.sink = "state update self.%s += (command)" % loop.written
    sink = rules.rewrite(with_roles(loop.step if stepping else loop.sink, names), algo.notes)
    sink, slew = _slew(sink, names)
    if slew is not None:
        limit, step = slew
        algo.plant = {"max_accel": _param_name(limit) or repr(limit), "time_step": repr(step)}
        used_params.update(dict.fromkeys(n.name for n in walk(limit)
                                         if isinstance(n, Sym) and n.kind == "param"))

    sel = _planner_select(sink)
    if sel is not None and _extract_energy_planner(algo, sel, ctx, used_params):
        if not _refused(algo, ctx):
            _finish(algo, ctx, used_params)
        return algo

    while isinstance(sink, Clamp):
        b = _bound_ir(sink, ctx.params)
        algo.global_bounds.append(b)
        for n in b.params:
            used_params[n] = None
        sink = sink.x
        sink = rules.rewrite(sink, algo.notes)

    if isinstance(sink, Mul):
        adds = [p for p in sink.parts if isinstance(p, Add) and is_vec(p)]
        if len(adds) == 1 and all(not is_vec(p) for p in sink.parts
                                  if p is not adds[0]):
            rest = [p for p in sink.parts if p is not adds[0]]
            sink = add(*[mul(*rest, q) for q in adds[0].parts])
            sink = rules.rewrite(sink, algo.notes)

    parts = list(sink.parts) if isinstance(sink, Add) else [sink]
    algo.base = "velocity_cmd"
    previous = Prop(SELF, loop.written) if loop.written else None
    if (loop.kind == "rate" or (loop.kind == "state" and role == "velocity")) \
            and previous in parts:
        algo.base = "add_velocity"
        parts = [p for p in parts if p != previous]
    elif stepping:
        algo.notes.append("the command is a position step (p += v*dt): base "
                          "velocity_cmd, the constant integration step rides in "
                          "the bound/weights")
    for part in parts:
        part = _single_body(part, names)
        for t in _split_conds(part):
            term = _classify_term(t, ctx, algo.notes, used_params)
            if term is not None:
                algo.terms.append(term)

    if _refused(algo, ctx):
        return algo

    _exogenous_sources(algo, names)
    _finish(algo, ctx, used_params)
    return algo


def _refused(algo: AlgorithmIR, ctx: Context) -> bool:
    unknown = sum(t.direction == "unknown" for t in algo.terms)
    if not unknown:
        return False
    algo.notes.append("refused: the command is not a sum of force terms "
                      "(%d part(s) without a direction)" % unknown)
    algo.terms, algo.global_bounds, algo.arbitration = [], [], None
    _finish(algo, ctx, {})
    return True


def _slew(sink: S, names):
    if not isinstance(sink, Cond):
        return sink, None
    tests = sink.test.parts if isinstance(sink.test, OrS) else (sink.test,)
    greater = next((t for t in tests if isinstance(t, Cmp) and t.op == ">"), None)
    if greater is None or not (isinstance(greater.left, Fn) and greater.left.name == "norm"):
        return sink, None
    accel, limit, command = greater.left.args[0], greater.right, sink.orelse
    velocity = next((p for p in walk(sink.then) if isinstance(p, Prop) and p.base == SELF
                     and names.fields.get(p.name) == "velocity"), None)
    if velocity is None:
        return sink, None
    canon = lambda x: rules.rewrite(x, [])
    lower = canon(Cmp("<", greater.left, neg(limit)))
    if not all(t == greater or canon(t) == lower for t in tests):
        return sink, None
    for step in {x for x in walk(accel) if isinstance(x, Sym)}:
        expected = canon(mul(sub(command, velocity), Pow(step, Const(-1.0))))
        if canon(accel) != expected:
            continue
        then = canon(sink.then)
        raw = mul(sub(command, velocity), Pow(step, Const(-1.0)))
        if then in (canon(add(velocity, mul(step, limit, fn("unit", expected)))),
                    canon(add(velocity, mul(step, limit, raw, Pow(fn("norm", raw), Const(-1.0)))))):
            return command, (limit, step)
    return sink, None


def _single_body(part: S, names) -> S:
    if any(isinstance(x, Reduce) for x in walk(part)):
        return part
    bodies = {p for p in walk(part) if isinstance(p, Prop) and isinstance(p.base, Sym)
              and p.base.kind == "free" and p.name in names.collections}
    if len(bodies) != 1:
        return part
    body = bodies.pop()
    elem = Obj("_" + body.name, domain=body.name)
    return Reduce("sum", elem.name, Dom(body.name),
                  subst(part, {body: Prop(elem, body.name, body.role)}))


def membership_profile(sink: S, names, ctx: "Context") -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for red in walk(with_roles(sink, names)):
        if not isinstance(red, Reduce) or not isinstance(red.domain, Dom):
            continue
        source = names.sources.get(_root_collection(red.domain))
        if source is None or source in out:
            continue
        term = TermIR(name="_")
        guard = red.domain.guard
        predicate = _render(cond(guard, Const(1.0), Const(0.0)), term, {}, ctx) if guard is not None else "1"
        cap = red.domain.cap
        number = cap.v if isinstance(cap, Const) else ctx.params.get(cap.name) \
            if isinstance(cap, Sym) else None
        key = red.domain.key
        order = "stable_distance" if key is not None and isinstance(key, Fn) and key.name == "norm" \
            else "stable_distance" if key is None else "stable_" + _feature_name(key, term)
        out[source] = {"predicate": predicate, "cap": int(number) if number is not None else None,
                       "order": order}
    return out


ATTRACTIVE = frozenset(("toward",))
REPULSIVE = frozenset(("away_raw", "away_unit", "shill"))


def _root_collection(domain: S) -> Optional[str]:
    while isinstance(domain, Dom) and isinstance(domain.base, Dom):
        domain = domain.base
    return domain.name if isinstance(domain, Dom) else None


def _exogenous_sources(algo: AlgorithmIR, names) -> None:
    if "agents" not in names.sources.values():
        return
    directions: Dict[str, set] = {}
    for t in algo.terms:
        name = getattr(t, "_collection", None)
        if name in names.collections and name not in names.sources \
                and t.source.startswith("unknown"):
            directions.setdefault(name, set()).add(t.direction)
    for name, found in directions.items():
        if found <= ATTRACTIVE:
            source, how = "target", "pulls the agent toward"
        elif found <= REPULSIVE:
            source, how = "obstacles", "pushes the agent away from"
        else:
            continue
        names.sources[name] = source
        names.reasons["source:" + name] = \
            "collection beside the peers that the law only %s" % how
        for t in algo.terms:
            if getattr(t, "_collection", None) == name:
                t.source = source


def _finish(algo: AlgorithmIR, ctx: Context, used_params: Dict[str, None]):
    _name_terms(algo.terms)
    algo.params = {n: {"value": ctx.params.get(n), "tunable": True}
                   for n in used_params
                   if n in ctx.params}
    for n in ctx.notes:
        if any(k in n for k in ("rasterisation", "distance transform",
                                "grid field", "derivative", "world anchor",
                                "sampled symbolically", "selection loop")):
            algo.notes.append("dataflow: " + n)
    gates = sorted({g for t in algo.terms for g in t.gates})
    if gates:
        algo.notes.append("mode gates found (arbitration candidates): %s"
                          % ", ".join(gates))


def _planner_select(sink: S) -> Optional[Tuple[Fn, str]]:
    x = sink
    if isinstance(x, Idx):
        x = x.base
    if isinstance(x, Prop) and isinstance(x.base, Fn) \
            and x.base.name in ("select_argmax", "select_argmin"):
        return x.base, x.name
    return None


def _extract_energy_planner(algo: AlgorithmIR, sel_attr: Tuple[Fn, str],
                            ctx: Context, used_params: Dict[str, None]
                            ) -> bool:
    sel, attr = sel_attr
    X, Q = sel.args
    if not isinstance(Q, Dom) or Q.elt is None or not isinstance(X, Dom):
        algo.notes.append("planner sink found but the score set is opaque")
        return False
    A = _softmax_exponent(Q.elt)
    if A is None:
        algo.notes.append("planner sink found but no softmax structure: %r"
                          % Q.elt)
        return False
    minimise = sel.name == "select_argmax"
    parts = [neg(p) for p in (A.parts if isinstance(A, Add) else [A])] \
        if minimise else list(A.parts if isinstance(A, Add) else [A])
    algo.notes.append(
        "optimisation-based controller: the command '%s' is the element of "
        "the sampled control set '%s' minimising an energy (softmax argmax); "
        "terms are the energy's additive parts as force laws (-dE/d dist)"
        % (attr, X.name))

    for part in parts:
        t = _energy_term(part, ctx, used_params, algo.notes)
        if t is not None:
            algo.terms.append(t)

    algo.base = "accel"
    _control_notes(X, ctx, algo)
    _label_filter_bound(X, algo)
    algo.arbitration = {
        "role_rule": "all_equal",
        "planner": "mean_field_argmax",
        "target_selector": {"agent": _anchor_name(parts) or "target"},
        "transition": "variational_convergence",
        "state": ["Qx (per-label distribution)"],
    }
    return True


def _softmax_exponent(elt: S) -> Optional[S]:
    if not isinstance(elt, Mul):
        return None
    exps = [p for p in elt.parts if isinstance(p, Fn) and p.name == "exp"]
    zs = [p for p in elt.parts
          if isinstance(p, Pow) and p.exp == Const(-1.0)
          and isinstance(p.base, Reduce)
          and any(isinstance(w, Fn) and w.name == "exp" for w in walk(p.base.body))]
    if len(exps) == 1 and len(zs) == 1 and len(elt.parts) == 2:
        return exps[0].args[0]
    return None


def _energy_term(part: S, ctx: Context, used_params: Dict[str, None],
                 notes: List[str]) -> Optional[TermIR]:
    if is_zero(part):
        return None
    term = TermIR(name="term", raw=repr(part))
    part, info = _canon_predicted(part, term)
    part = rules.rewrite(part, term.notes)

    core = part
    if isinstance(core, Reduce):
        term.aggregator = core.op
        term.source_set = _domain_name(core.domain)
        term.source = _source(core.domain, ctx.names)
        for g in _conjuncts(_unnot(core.guard)):
            _classify_energy_guard(g, term, used_params, notes, ctx.names)
        core = core.body
    while isinstance(core, Cond) and is_zero(core.then):
        _classify_energy_guard(_unnot(NotS(core.test)), term, used_params, notes, ctx.names)
        core = core.orelse
    while isinstance(core, Cond) and is_zero(core.orelse):
        _classify_energy_guard(_unnot(core.test), term, used_params, notes, ctx.names)
        core = core.then
    core = _strip_expectation(core, term)

    factors = list(core.parts) if isinstance(core, Mul) else [core]
    gates = [f for f in factors if isinstance(f, (Cmp, NotS))]
    rest = [f for f in factors if f not in gates]
    for g in gates:
        _classify_energy_guard(_unnot(g), term, used_params, notes, ctx.names)
        term.notes.append("comparison used as a multiplicative (0/1) factor "
                          "in the source: %r" % g)
    energy = mul(*rest) if rest else Const(1.0)

    if term.source == "unknown" and info["anchor"]:
        term.source = "target"
        term.source_set = term.source_set or info["anchor"]

    dist = Sym("dist", "input")
    from . import fields
    force = neg(fields.deriv(energy, dist, term.notes))
    force = rules.rewrite(force, [])
    if any(isinstance(w, Unk) for w in walk(force)):
        term.direction = "away_unit"
        term.magnitude = _render(energy, term, used_params, ctx)
        term.notes.append("energy kept as the magnitude (derivative was "
                          "opaque); the planner minimises it")
        return term
    if is_zero(force):
        if any(isinstance(w, (Prop, Obj)) or (isinstance(w, Sym) and w.kind == "free")
               for w in walk(energy)):
            notes.append("energy part over state with no named position/velocity: %r" % energy)
            return term
        notes.append("dropped constant energy offset %r" % energy)
        return None
    term.signal = "position"
    if term.source == "target":
        sign, force = _extract_sign(force)
        term.direction = "toward" if sign < 0 else "away_unit"
    else:
        term.direction = "away_unit"
    weight, mag = _weight_split(force)
    term.weight = _render(weight, term, used_params, ctx) \
        if weight is not None else None
    term.magnitude = _render(mag, term, used_params, ctx)
    term.notes.append("force form of the energy term %s (direction from "
                      "-dE/d dist along unit(self - source))"
                      % _render(energy, term, used_params, ctx))
    return term


def _canon_predicted(x: S, term: TermIR) -> Tuple[S, Dict]:
    from .sexpr import memo_map
    info = {"anchor": None, "predicted": False}

    def scan(arg: S):
        pos_roots, vel = set(), False
        anchor = None
        for w in walk(arg):
            if isinstance(w, Prop):
                root = w
                while isinstance(root, Prop):
                    root = root.base
                if _has_role(w, "position"):
                    pos_roots.add(root)
                    if isinstance(root, Obj) and root.domain == "anchor":
                        anchor = root.name
                elif _has_role(w, "velocity"):
                    vel = True
        return pos_roots, vel, anchor

    def step(s: S):
        if isinstance(s, Fn) and s.name == "norm" and len(s.args) == 1:
            pos_roots, vel, anchor = scan(s.args[0])
            if len(pos_roots) >= 2 or (pos_roots and anchor):
                if vel:
                    info["predicted"] = True
                if anchor:
                    info["anchor"] = anchor
                return Sym("dist", "input")
            if vel and not pos_roots:
                return Sym("rel_speed", "input")
        return None

    out = memo_map(x, step)
    if info["predicted"]:
        term.notes.append("distances are between tau-lookahead predicted "
                          "positions (p + v*tau + u*tau^2/2)")
    return out, info


def _strip_expectation(core: S, term: TermIR) -> S:
    if not isinstance(core, Reduce) or not isinstance(core.body, Mul):
        return core
    parts = list(core.body.parts)
    exps = [p for p in parts if isinstance(p, Fn) and p.name == "exp"]
    for z in parts:
        if isinstance(z, Pow) and z.exp == Const(-1.0) \
                and isinstance(z.base, Reduce):
            for w in exps:
                if z.base.body == w:
                    rest = [p for p in parts if p is not w and p is not z]
                    term.aggregator = "sum"
                    term.notes.append(
                        "pairwise energy is an expectation under the "
                        "neighbour's label distribution (mean-field); "
                        "collapsed to the representative label")
                    return mul(*rest) if rest else Const(1.0)
    return core


def _classify_energy_guard(g: S, term: TermIR, used_params: Dict[str, None],
                           notes: List[str], names=None):
    if isinstance(g, Cmp) and _same_up_to_namespace(g.left, g.right) \
            and g.op in ("==", "!="):
        return
    if isinstance(g, Cmp):
        feat, param, op = _feature_side(g)
        if feat is not None:
            fname = _feature_name(feat, term)
            if fname not in ("dist", "clearance", "angle"):
                pname = _param_name(param)
                if pname:
                    used_params[pname] = None
                term.activation = ActivationIR(
                    "below" if op in ("<", "<=") else "above",
                    pname or repr(param), fname)
                return
    _classify_guard(g, term, used_params, notes, names)


def _unnot(g: Optional[S]) -> Optional[S]:
    while isinstance(g, NotS) and isinstance(g.x, NotS):
        g = g.x.x
    return g


def _same_up_to_namespace(a: S, b: S) -> bool:
    def canon(x: S) -> S:
        from .sexpr import memo_map

        def step(s: S):
            if isinstance(s, Obj) and "@" in s.name:
                return Obj(s.name.split("@")[0], s.domain, s.source)
            return None
        return memo_map(x, step)
    return canon(a) == canon(b)


def _anchor_name(parts: List[S]) -> Optional[str]:
    for p in parts:
        for w in walk(p):
            if isinstance(w, Obj) and w.domain == "anchor":
                return w.name
    return None


def _control_notes(X: Dom, ctx: Context, algo: AlgorithmIR):
    scale = 1.0
    if isinstance(X.elt, Obj):
        u = ctx.heap.get(Prop(X.elt, "u"))
        ks = set()
        comps = u.items if isinstance(u, Vec) else (u,) if u is not None else ()
        for c in comps:
            if isinstance(c, Mul):
                k = 1.0
                for p in c.parts:
                    if isinstance(p, Const):
                        k *= abs(p.v)
                ks.add(k)
        if len(ks) == 1:
            scale = ks.pop()
    bound = None
    d = X
    while isinstance(d, Dom):
        for g in _conjuncts(_unnot(d.guard)):
            if isinstance(g, Cmp) and g.op in ("<", "<="):
                for side in (g.left, g.right):
                    if isinstance(side, Const):
                        bound = max(bound or 0.0, abs(side.v))
        d = d.base
    if bound is not None:
        algo.global_bounds.append(BoundIR(
            "clamp_accel", {"max_accel": bound * scale}))
        algo.notes.append(
            "control space: accelerations sampled per-axis on a grid with "
            "|u| <= %g%s" % (bound, "" if scale == 1.0 else
                             ", commanded as %g * sample" % scale))


def _label_filter_bound(X: Dom, algo: AlgorithmIR):
    for g in _conjuncts(_unnot(X.guard)):
        if isinstance(g, Cmp) and g.op in ("<", "<=") \
                and isinstance(g.right, Const):
            has_vel = any(_has_role(w, "velocity")
                          for w in walk(g.left))
            if has_vel:
                algo.global_bounds.append(BoundIR(
                    "clamp_speed", {"max_speed": g.right.v}))
                algo.notes.append(
                    "candidate controls whose predicted per-axis velocity "
                    "exceeds %g are filtered out (speed clamp by "
                    "construction)" % g.right.v)
                return


def _split_conds(part: S) -> List[Tuple[S, List[S]]]:
    out: List[Tuple[S, List[S]]] = []

    def go(p: S, gates: List[S]):
        if isinstance(p, Cond) and not _is_threshold_test(p.test):
            if is_zero(p.orelse):
                go(p.then, gates + [p.test])
                return
            if is_zero(p.then):
                go(p.orelse, gates + [NotS(p.test)])
                return
            go(p.then, gates + [p.test])
            go(p.orelse, gates + [NotS(p.test)])
            return
        if isinstance(p, Clamp) and isinstance(p.x, Cond) \
                and not _is_threshold_test(p.x.test):
            c = p.x
            for branch, gate in ((c.then, c.test), (c.orelse, NotS(c.test))):
                if not is_zero(branch):
                    go(Clamp(branch, p.hi, p.lo, p.exact), gates + [gate])
            return
        out.append((p, gates))

    go(part, [])
    return out


def _is_threshold_test(t: S) -> bool:
    if isinstance(t, Cmp):
        return any(isinstance(x, (Fn, Add, Mul, Pow)) or
                   (isinstance(x, Sym) and x.kind == "input")
                   for x in (t.left, t.right))
    return False


def _classify_term(part_gates: Tuple[S, List[S]], ctx: Context,
                   notes: List[str], used_params: Dict[str, None]
                   ) -> Optional[TermIR]:
    part, gate_exprs = part_gates
    if is_zero(part) or isinstance(part, Unk):
        if isinstance(part, Unk):
            notes.append("dropped opaque sink part %r" % part)
        return None
    term = TermIR(name="term", raw=repr(part))
    for g in gate_exprs:
        term.gates.append(_gate_label(g))

    while isinstance(part, Cond) and is_zero(part.orelse):
        _classify_guard(part.test, term, used_params, notes, ctx.names)
        part = part.then

    scalars: List[S] = []
    core = part
    if isinstance(part, Mul):
        vecs = [p for p in part.parts if is_vec(p) or isinstance(p, (Reduce, Clamp, Unk, Idx))]
        scalars = [p for p in part.parts if p not in vecs]
        if len(vecs) == 1:
            core = vecs[0]
        else:
            core = mul(*vecs) if vecs else part
    if isinstance(core, Clamp):
        term.bound = _bound_ir(core, ctx.params)
        for n in term.bound.params:
            used_params[n] = None
        core = core.x

    body = core
    pending_guards: List[S] = []
    if isinstance(core, Reduce):
        term.aggregator = core.op
        term.source_set = _domain_name(core.domain)
        term.source = _source(core.domain, ctx.names)
        term._collection = _root_collection(core.domain)
        if isinstance(core.domain, Dom) and core.domain.cap is not None:
            term.membership.cap_param = _param_name(core.domain.cap) or repr(core.domain.cap)
            used_params.update(dict.fromkeys(n.name for n in walk(core.domain.cap)
                                             if isinstance(n, Sym) and n.kind == "param"))
        pending_guards.extend(_conjuncts(core.guard))
        pending_guards.extend(_conjuncts(
            core.domain.guard if isinstance(core.domain, Dom) else None))
        body = core.body
        var = core.var
    else:
        var = None

    shape, vecpart = _split_scalar_vector(body)
    scalars.append(shape)
    if vecpart is None:
        notes.append("term with no vector part: %r" % body)
        return None

    inner_reduce = _match_direction(vecpart, var, term, notes)
    if term.signal_speed:
        used_params[term.signal_speed] = None
    for g in pending_guards:
        _classify_guard(g, term, used_params, notes, ctx.names)
    if inner_reduce is not None and not isinstance(core, Reduce):
        term.aggregator = inner_reduce.op
        term.source_set = _domain_name(inner_reduce.domain)
        term.source = _source(inner_reduce.domain, ctx.names)
        term._collection = _root_collection(inner_reduce.domain)
        for g in _conjuncts(inner_reduce.guard):
            _classify_guard(g, term, used_params, notes, ctx.names)
        for g in _conjuncts(inner_reduce.domain.guard
                            if isinstance(inner_reduce.domain, Dom) else None):
            _classify_guard(g, term, used_params, notes, ctx.names)
    if term.source == "unknown" and inner_reduce is None \
            and not isinstance(core, Reduce):
        free = next((w for w in walk(body)
                     if isinstance(w, Obj) and w.domain), None)
        if free is not None:
            term.source_set = free.domain
            term.source = ctx.names.sources.get(free.domain) or "unknown:%s" % free.domain
            term.aggregator = "nearest"
            term.measures = "centre_to_surface"
            term.notes.append("nearest element of '%s': a distance-transform "
                              "field couples to the closest body only, and "
                              "measures to its surface" % free.domain)

    shape_expr = mul(*scalars) if scalars else Const(1.0)
    sign, shape_expr = _extract_sign(shape_expr)
    if sign < 0:
        flipped = _flip(term.direction, _was_unit(vecpart))
        if flipped is not None:
            term.direction = flipped
        else:
            shape_expr = neg(shape_expr)
    if term.direction in ("toward", "away_raw") and not _was_unit(vecpart):
        if term.direction == "toward":
            shape_expr = mul(shape_expr, Sym("dist", "input"))
    weight, mag = _weight_split(shape_expr)
    term.weight = _render(weight, term, used_params, ctx) if weight is not None else None
    term.magnitude = _render(mag, term, used_params, ctx)
    return term


def _match_direction(v: S, var: Optional[str], term: TermIR,
                     notes: List[str]) -> Optional[Reduce]:
    unit = False
    if isinstance(v, Fn) and v.name == "unit":
        unit = True
        v = v.args[0]
    atoms = _vector_atoms(v)
    kinds = {a[0] for a in atoms}

    def has(k):
        return k in kinds

    inner_reduce = None
    for k, a in atoms:
        if k in ("centroid_pos", "centroid_vel") and isinstance(a[0], Reduce):
            inner_reduce = a[0]

    if has("elem_vel") and has("self_vel") and _signs(atoms, "elem_vel") > 0:
        term.signal = "velocity"
        term.direction = "friction" if unit else "match"
        return inner_reduce
    if has("centroid_vel") and has("self_vel"):
        term.signal = "velocity"
        term.direction = "friction" if unit else "match"
        return inner_reduce
    if has("self_vel") and has("elem_pos") and has("self_pos") \
            and _signs(atoms, "self_vel") < 0:
        speed = _shill_speed(v)
        if speed is not None:
            term.signal = "velocity"
            term.direction = "shill"
            term.signal_speed = speed
            return inner_reduce
    if has("elem_pos") and has("self_pos"):
        term.signal = "position"
        toward = _signs(atoms, "elem_pos") > 0
        term.direction = ("toward" if toward else "away_raw") if not unit else \
            ("toward" if toward else "away_unit")
        if unit and toward:
            term.direction = "toward"
        return inner_reduce
    if has("centroid_pos") and has("self_pos"):
        term.signal = "position"
        term.direction = "toward" if _signs(atoms, "centroid_pos") > 0 else "away_raw"
        return inner_reduce
    if has("target") and has("centroid_pos"):
        term.signal = "position"
        term.direction = "toward" if _signs(atoms, "target") > 0 else "away_raw"
        term.anchor = "centroid"
        term.source = "target"
        term.source_set = term.source_set or "target"
        return None
    if has("target") and has("self_pos"):
        term.signal = "position"
        term.direction = "toward" if _signs(atoms, "target") > 0 else "away_raw"
        term.source = "target"
        term.source_set = term.source_set or "target"
        return inner_reduce
    if has("target") and len(kinds) == 1:
        term.signal = "position"
        term.direction = "toward"
        term.source = "target"
        return inner_reduce
    if has("self_vel") and len(kinds) == 1:
        term.signal = "velocity"
        term.direction = "axis"
        term.source = "self"
        term.source_set = "self"
        return inner_reduce
    if unit:
        notes.append("unit direction with unrecognised operand: %r" % v)
        term.direction = "away_unit" if _leading_sign(v) < 0 else "toward"
        return inner_reduce
    notes.append("unclassified direction: %r" % v)
    return inner_reduce


def _shill_speed(v: S) -> Optional[str]:
    for part in (v.parts if isinstance(v, Add) else (v,)):
        props = [w for w in walk(part) if isinstance(w, Prop)]
        if not any(_atom_kind(w) == "elem_pos" for w in props):
            continue
        if any(_atom_kind(w) == "self_vel" for w in props):
            continue
        names = {w.name for w in walk(part)
                 if isinstance(w, Sym) and w.kind == "param"}
        if len(names) == 1:
            return names.pop()
        return None
    return None


def _vector_atoms(v: S) -> List[Tuple[str, Tuple[S, float]]]:
    out: List[Tuple[str, Tuple[S, float]]] = []

    def go(x: S, sign: float):
        if isinstance(x, Fn) and x.name == "unit":
            go(x.args[0], sign)
            return
        if isinstance(x, Add):
            for p in x.parts:
                go(p, sign)
            return
        if isinstance(x, Mul):
            c = 1.0
            rest = []
            for p in x.parts:
                if isinstance(p, Const):
                    c *= p.v
                else:
                    rest.append(p)
            if len(rest) == 1:
                go(rest[0], sign * (1.0 if c >= 0 else -1.0))
                return
            vecs = [q for q in rest if is_vec(q)]
            if len(vecs) == 1:
                go(vecs[0], sign * (1.0 if c >= 0 else -1.0))
                return
        out.append((_atom_kind(x), (x, sign)))

    go(v, 1.0)
    return out


def _atom_kind(x: S) -> str:
    if isinstance(x, Prop):
        if isinstance(x.base, Obj) and x.base.domain == "anchor":
            return "target"
        base_self = isinstance(x.base, Obj) and x.base.name == "self"
        if _has_role(x, "position"):
            return "self_pos" if base_self else "elem_pos"
        if _has_role(x, "velocity"):
            return "self_vel" if base_self else "elem_vel"
        if _has_role(x, "target"):
            return "target"
        return "prop"
    if isinstance(x, Reduce):
        k = _atom_kind_of_body(x.body)
        if k in ("elem_pos", "self_pos"):
            return "centroid_pos"
        if k in ("elem_vel", "self_vel"):
            return "centroid_vel"
        return "reduce"
    if isinstance(x, Idx):
        return _atom_kind(x.base)
    return "other"


def _atom_kind_of_body(b: S) -> str:
    if isinstance(b, Prop):
        return _atom_kind(b)
    for w in walk(b):
        if isinstance(w, Prop):
            return _atom_kind(w)
    return "other"


def _signs(atoms, kind: str) -> float:
    for k, (x, sign) in atoms:
        if k == kind:
            return sign
    return 0.0


def _leading_sign(v: S) -> float:
    if isinstance(v, Mul):
        for p in v.parts:
            if isinstance(p, Const):
                return 1.0 if p.v >= 0 else -1.0
    return 1.0


def _was_unit(v: S) -> bool:
    return isinstance(v, Fn) and v.name == "unit"


def _flip(direction: str, unit: bool = False) -> Optional[str]:
    if direction == "toward":
        return "away_unit" if unit else "away_raw"
    if direction in ("away_raw", "away_unit"):
        return "toward"
    return None


def _conjuncts(g: Optional[S]) -> List[S]:
    if g is None:
        return []
    if isinstance(g, AndS):
        return list(g.parts)
    return [g]


def _classify_guard(g: S, term: TermIR, used_params: Dict[str, None],
                    notes: List[str], names=None):
    if isinstance(g, Cmp):
        if isinstance(g.left, (Obj, Prop)) and isinstance(g.right, (Obj, Prop)) \
                and g.op in ("!=", "=="):
            return
        if g.op == "!=" and is_zero(g.right) and isinstance(g.left, Fn) and g.left.name == "norm":
            return
        feat, param, op = _feature_side(g)
        if feat is not None:
            fname = _feature_name(feat, term)
            pname = _param_name(param)
            is_range = names is not None and names.membership.get(pname) == "range"
            if pname:
                used_params[pname] = None
            if fname == "angle":
                term.membership.fov_param = pname or str(param)
                return
            below = op in ("<", "<=")
            if fname == "clearance":
                term.activation = ActivationIR("below" if below else "above",
                                               pname or repr(param), "clearance")
                if pname and is_range:
                    term.membership.range_param = pname
                return
            if fname == "dist":
                if pname and is_range:
                    term.membership.range_param = pname
                    return
                term.activation = ActivationIR("below" if below else "above",
                                               pname or repr(param), "dist")
                return
            if fname in CANONICAL_FEATURES:
                term.activation = ActivationIR(
                    "below" if below else "above",
                    pname or _render(param, term, used_params, None), fname)
                return
            term.notes.append("guard on %s %s %r" % (fname, op, param))
            return
    if isinstance(g, Sym):
        term.gates.append(g.name)
        return
    if isinstance(g, Prop):
        term.gates.append(g.name)
        return
    if isinstance(g, NotS):
        inner = TermIR(name="_")
        _classify_guard(g.x, inner, used_params, notes, names)
        term.gates.extend("not " + x for x in inner.gates)
        if inner.activation.kind != "always":
            flip = {"below": "above", "above": "below"}[inner.activation.kind]
            term.activation = ActivationIR(flip, inner.activation.threshold,
                                           inner.activation.feature)
        term.membership = inner.membership if inner.membership != MembershipIR() \
            else term.membership
        return
    term.notes.append("unclassified guard: %r" % g)


def _feature_side(g: Cmp):
    def is_feature(x: S) -> bool:
        return any(isinstance(w, Fn) and w.name in ("norm", "angle", "acos", "degrees")
                   for w in walk(x)) or \
            any(isinstance(w, Sym) and w.kind == "input" for w in walk(x))

    def is_atomic(x: S) -> bool:
        return (isinstance(x, Fn) and x.name in ("norm", "angle")) or \
            (isinstance(x, Sym) and x.kind == "input")

    if is_feature(g.left) and not is_feature(g.right):
        return g.left, g.right, g.op
    if is_feature(g.right) and not is_feature(g.left):
        flipped = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(g.op, g.op)
        return g.right, g.left, flipped
    if is_feature(g.left) and is_feature(g.right):
        if is_atomic(g.left) and not is_atomic(g.right):
            return g.left, g.right, g.op
        if is_atomic(g.right) and not is_atomic(g.left):
            flipped = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(g.op, g.op)
            return g.right, g.left, flipped
    return None, None, g.op


CANONICAL_FEATURES = ("rel_speed", "shill_speed")


def _feature_name(feat: S, term: Optional[TermIR] = None) -> str:
    if term is not None:
        canon = _canon_features(feat, term)
        if isinstance(canon, Sym) and canon.kind == "input":
            return canon.name
    has_norm = any(isinstance(w, Fn) and w.name == "norm" for w in walk(feat))
    has_angle = any(isinstance(w, Fn) and w.name in ("angle", "acos") for w in walk(feat))
    has_size = any(_has_role(w, "size") for w in walk(feat))
    has_vel = any(_has_role(w, "velocity") for w in walk(feat))
    if has_angle:
        return "angle"
    if has_norm and has_vel and not has_size:
        return "rel_speed"
    if has_norm and has_size:
        return "clearance"
    if has_norm:
        return "dist"
    for w in walk(feat):
        if isinstance(w, Sym) and w.kind == "input":
            return w.name
    return "expr"


def _param_name(p: Optional[S]) -> Optional[str]:
    if isinstance(p, Sym):
        return p.name
    return None


def _gate_label(g: S) -> str:
    if isinstance(g, NotS):
        return "not " + _gate_label(g.x)
    if isinstance(g, (Sym,)):
        return g.name
    if isinstance(g, Prop):
        return g.name
    from .sexpr import srepr
    text = srepr(g, 4)
    return text if len(text) <= 80 else text[:77] + "..."


def _split_scalar_vector(body: S) -> Tuple[S, Optional[S]]:
    if isinstance(body, Mul):
        vecs = [p for p in body.parts if is_vec(p) or isinstance(p, (Idx, Unk))
                or _looks_vector(p)]
        scalars = [p for p in body.parts if p not in vecs]
        if len(vecs) == 1:
            return (mul(*scalars) if scalars else Const(1.0)), vecs[0]
        if not vecs:
            return body, None
        return (mul(*scalars) if scalars else Const(1.0)), mul(*vecs)
    if is_vec(body) or _looks_vector(body) or isinstance(body, (Unk, Idx)):
        return Const(1.0), body
    return body, None


def _looks_vector(p: S) -> bool:
    if isinstance(p, Prop):
        return is_vec(p)
    if isinstance(p, (Add, Reduce, Cond)):
        return is_vec(p) or any(_looks_vector(c) for c in
                                (p.parts if isinstance(p, Add) else [p.body] if isinstance(p, Reduce)
                                 else [p.then, p.orelse]))
    if isinstance(p, Fn) and p.name in ("unit", "dot"):
        return p.name == "unit"
    return False


def _extract_sign(shape: S) -> Tuple[float, S]:
    if isinstance(shape, Const) and shape.v < 0:
        return -1.0, Const(-shape.v)
    if isinstance(shape, Mul):
        for p in shape.parts:
            if isinstance(p, Const) and p.v < 0:
                rest = [q for q in shape.parts if q is not p]
                return -1.0, mul(Const(-p.v), *rest)
    return 1.0, shape


def _weight_split(shape: S) -> Tuple[Optional[S], S]:
    parts = list(shape.parts) if isinstance(shape, Mul) else [shape]
    gains = [p for p in parts if _is_pure_param(p)]
    rest = [p for p in parts if not _is_pure_param(p)]
    if not gains:
        return None, shape
    if not rest:
        return mul(*gains) if len(gains) > 1 else gains[0], Const(1.0)
    return (mul(*gains) if len(gains) > 1 else gains[0]), mul(*rest)


def _is_pure_param(p: S) -> bool:
    return isinstance(p, Sym) and p.kind == "param"


def _render(x: S, term: TermIR, used_params: Dict[str, None],
            ctx: Context) -> str:
    x = _canon_features(x, term)
    for w in walk(x):
        if isinstance(w, Sym) and w.kind == "param":
            used_params[w.name] = None
    return _print_sympy(x, term)


def _canon_features(x: S, term: TermIR) -> S:
    from .sexpr import memo_map

    def step(s: S):
        if isinstance(s, Fn) and s.name == "norm" and len(s.args) == 1:
            inner = s.args[0]
            k = {_atom_kind(w) for w in walk(inner) if isinstance(w, Prop)}
            if k and k <= {"self_pos", "elem_pos", "target", "centroid_pos"}:
                return Sym("dist", "input")
            if k and k <= {"self_vel", "elem_vel", "centroid_vel"}:
                return Sym("rel_speed", "input")
            if term.direction == "shill" and k and "self_vel" in k \
                    and k <= {"self_vel", "self_pos", "elem_pos", "prop"}:
                return Sym("shill_speed", "input")
        if isinstance(s, Add):
            parts = list(s.parts)
            dist_i = next((i for i, p in enumerate(parts)
                           if p == Sym("dist", "input")), None)
            size_i = next((i for i, p in enumerate(parts)
                           if isinstance(p, Mul) and Const(-1.0) in p.parts
                           and any(_has_role(q, "size")
                                   for q in p.parts)), None)
            if dist_i is not None and size_i is not None:
                if any(_has_role(q, "size")
                       and not (isinstance(q.base, Obj) and q.base.name == "self")
                       for q in walk(parts[size_i])):
                    term.measures = "centre_to_surface"
                rest = [p for i, p in enumerate(parts) if i not in (dist_i, size_i)]
                return add(Sym("clearance", "input"), *rest) if rest \
                    else Sym("clearance", "input")
        return None

    return memo_map(x, step)


_FN_SYMPY = {"exp": "exp", "sqrt": "sqrt", "abs": "Abs", "min": "Min",
             "max": "Max", "cos": "cos", "sin": "sin", "dot": "dot",
             "norm": "norm", "unit": "unit", "angle": "angle", "mod": "Mod",
             "acos": "acos", "degrees": "deg", "len": "len"}


_CMP_SYMPY = {"<": "<", "<=": "<=", ">": ">", ">=": ">=",
              "==": "Eq", "!=": "Ne"}


def _print_sympy(x: S, term: TermIR, _u=[0]) -> str:
    def p(s: S, prec: int = 0) -> str:
        if isinstance(s, Const):
            v = "%g" % s.v
            return "(%s)" % v if s.v < 0 and prec > 0 else v
        if isinstance(s, Sym):
            return s.name
        if isinstance(s, Add):
            inner = " + ".join(p(q, 1) for q in s.parts).replace("+ -", "- ")
            return "(%s)" % inner if prec > 1 else inner
        if isinstance(s, Mul):
            inner = "*".join(p(q, 2) for q in s.parts)
            return "(%s)" % inner if prec > 2 else inner
        if isinstance(s, Pow):
            return "%s**%s" % (p(s.base, 3), p(s.exp, 3))
        if isinstance(s, Fn) and s.name in _FN_SYMPY:
            return "%s(%s)" % (_FN_SYMPY[s.name], ", ".join(p(a) for a in s.args))
        if isinstance(s, Cond):
            test = q(s.test)
            if test is not None:
                return "Piecewise((%s, %s), (%s, True))" % (
                    p(s.then), test, p(s.orelse))
        _u[0] += 1
        name = "u%d" % _u[0]
        term.notes.append("opaque %s = %r" % (name, s))
        return name

    def q(s: S):
        if isinstance(s, Cmp):
            op = _CMP_SYMPY.get(s.op)
            if op is None:
                return None
            left, right = p(s.left), p(s.right)
            if op.isalpha():
                return "%s(%s, %s)" % (op, left, right)
            return "(%s %s %s)" % (left, op, right)
        if isinstance(s, (AndS, OrS)):
            parts = [q(a) for a in s.parts]
            if any(a is None for a in parts):
                return None
            return "%s(%s)" % ("And" if isinstance(s, AndS) else "Or",
                               ", ".join(parts))
        if isinstance(s, NotS):
            inner = q(s.x)
            return None if inner is None else "Not(%s)" % inner
        return None

    return p(x)


def _bound_ir(c: Clamp, params: Dict[str, float]) -> BoundIR:
    kind = "renormalize" if c.exact else "clamp_speed"
    out: Dict[str, float] = {}
    lims = ((("max", c.hi),) if c.exact else (("max", c.hi), ("min", c.lo)))
    for label, lim in lims:
        if lim is None:
            continue
        if isinstance(lim, Sym):
            out[lim.name] = params.get(lim.name)
        elif isinstance(lim, Const):
            out["%s_speed" % label] = lim.v
        elif isinstance(lim, Mul):
            syms = [p for p in lim.parts if isinstance(p, Sym)]
            consts = [p for p in lim.parts if isinstance(p, Const)]
            if len(syms) == 1 and len(syms) + len(consts) == len(lim.parts):
                out[syms[0].name] = params.get(syms[0].name)
                k = 1.0
                for q in consts:
                    k *= q.v
                if k != 1.0:
                    out["dt"] = k
    return BoundIR(kind, out)


def _name_terms(terms: List[TermIR]):
    counts: Dict[str, int] = {}
    for t in terms:
        base = _term_base_name(t)
        n = counts.get(base, 0)
        counts[base] = n + 1
        t.name = base if n == 0 else "%s_%d" % (base, n + 1)


def _term_base_name(t: TermIR) -> str:
    if t.source == "agents":
        if t.direction in ("match", "friction") and t.signal == "velocity":
            return "alignment"
        if t.direction in ("away_raw", "away_unit"):
            return "separation"
        if t.direction == "toward":
            return "cohesion"
        return "coupling"
    if t.source == "obstacles":
        return "obstacle"
    if t.source == "target":
        return "migration" if t.anchor == "centroid" else "attraction"
    if t.source == "self":
        return "damping"
    return "term"


def _domain_name(d: S) -> str:
    return d.name if isinstance(d, Dom) else repr(d)


def fix_params(algo: AlgorithmIR, names) -> List[str]:
    missing = []
    for name in names:
        if name in algo.params:
            algo.params[name]["tunable"] = False
        else:
            missing.append(name)
    return missing


def augment_driver(algo: AlgorithmIR, prog, entry_qualname: str, names) -> AlgorithmIR:
    import copy as _copy

    driver = _Driver(prog, entry_qualname, names)

    injected = driver.injected_peers() if driver else None
    if injected:
        population, helper = injected
        clones = []
        for t in algo.terms:
            if t.source == "obstacles":
                c = _copy.deepcopy(t)
                c.source = "agents"
                c.source_set = "%s (driver-injected)" % population
                c.notes = list(c.notes) + [
                    "driver passes peer %s as obstacles (%s)" % (population, helper)]
                clones.append(c)
        if clones:
            algo.terms.extend(clones)
            _name_terms(algo.terms)
            algo.notes.append("driver scan: peer %s are injected into "
                              "the obstacle set -> mirrored agents-source term"
                              % population)

    names = _call_names(prog)
    if (algo.arbitration or {}).get("planner") == "mean_field_argmax" \
            and "sort" in names:
        for t in algo.terms:
            if t.source == "agents":
                t.membership.topology = "knn"
                t.membership.cap_param = "k_neighbors"
        if "k_neighbors" in getattr(prog, "params", {}):
            algo.params.setdefault(
                "k_neighbors", {"value": prog.params["k_neighbors"],
                                "tunable": True})
        algo.notes.append(
            "driver scan: neighbour nodes are the K nearest (distance-sorted "
            "list truncated to K+1 incl. self) -> knn membership; K is the "
            "'k_neighbors' launch argument")

    roles = driver.target_roles() if driver else None
    if roles:
        planner, target, formation, first, advances = roles
        algo.roles = ("leader", "follower")
        algo.arbitration = {
            "role_rule": "first_is_leader" if first else "unknown",
            "planner": planner,
            "target_selector": {"leader": "path_waypoint",
                                "follower": "formation_slot"},
            "transition": "advance_waypoint" if advances else None,
            "state": ["leader_path", "leader_wp"],
        }
        algo.notes.append(
            "driver scan: global path planner (%s) feeds the leader's %s; "
            "followers' %s are formation slots (%s); TARGET resolves per role"
            % (planner, target, target, formation))
    return algo


class _Driver:
    def __init__(self, prog, entry, names):
        from . import ir as _ir
        self.ir = _ir
        function, _ = prog.functions[entry]
        self.method = function.name if function.cls else None
        self.params = function.params
        self.positions = {n for n, r in names.fields.items() if r == "position"}
        self.targets = {n for n, r in names.fields.items() if r == "target"}
        self.obstacle_args = [i - 1 for i, p in enumerate(function.params)
                              if i > 0 and names.sources.get(p) == "obstacles"]
        unique = len(prog.methods.get(self.method, ())) == 1 if self.method else False
        self.scopes = []
        if unique:
            bodies = [m.body for m in prog.modules.values()]
            bodies += [f.body for f, _ in prog.functions.values()]
            for body in bodies:
                stmts = list(self._statements(body))
                calls = [n for s in stmts for e in _stmt_exprs(s) for n in _walk_expr(e)
                         if isinstance(n, _ir.Call) and isinstance(n.fn, _ir.Attr)
                         and n.fn.name == self.method]
                if calls:
                    self.scopes.append((stmts, self._defs(stmts), calls))

    def _statements(self, body):
        ir = self.ir
        for s in body:
            if isinstance(s, (ir.FuncDef, ir.ClassDef)):
                continue
            yield s
            if isinstance(s, ir.If):
                yield from self._statements(s.body)
                yield from self._statements(s.orelse)
            elif isinstance(s, (ir.For, ir.While)):
                yield from self._statements(s.body)

    def _defs(self, stmts):
        ir, defs = self.ir, {}
        for s in stmts:
            if isinstance(s, (ir.Assign, ir.AugAssign, ir.For)):
                value = s.iter if isinstance(s, ir.For) else s.value
                target = s.target
                for t in target.items if isinstance(target, (ir.TupleE, ir.ListE)) else [target]:
                    if isinstance(t, ir.NameE):
                        defs.setdefault(t.id, []).append(value)
        return defs

    def _slice(self, expr, defs, values=False):
        seen, out, pending = set(), [], [expr]
        while pending and len(out) < 256:
            e = pending.pop(0)
            out.append(e)
            for n in (self._data(e) if values else _walk_expr(e)):
                if isinstance(n, self.ir.NameE) and n.id not in seen:
                    seen.add(n.id)
                    pending.extend(defs.get(n.id, ()))
        return out

    def _data(self, e):
        ir, stack = self.ir, [e]
        while stack:
            x = stack.pop()
            yield x
            if isinstance(x, ir.Index):
                stack.append(x.base)
            elif isinstance(x, ir.Comp):
                stack.extend((x.elt, x.iter))
            else:
                stack.extend(_children_of(x))

    def _root(self, e):
        while isinstance(e, (self.ir.Attr, self.ir.Index)):
            e = e.base
        return e.id if isinstance(e, self.ir.NameE) else None

    def _receivers(self):
        return {self._root(c.fn.base) for _, _, calls in self.scopes for c in calls} - {None}

    def _collection(self, defs):
        ir = self.ir
        lists = {self._root(c.fn.base) for _, _, calls in self.scopes for c in calls
                 if isinstance(c.fn.base, ir.Index)}
        return next(iter(lists)) if len(lists) == 1 else None

    def _defining_call(self, expr, defs, reads):
        ir = self.ir
        for e in self._slice(expr, defs, values=True):
            if isinstance(e, ir.Call) and isinstance(e.fn, ir.NameE) and any(
                    isinstance(a, ir.Attr) and reads(a)
                    for x in list(e.args) + list(e.kwargs.values())
                    for y in self._slice(x, defs, values=True) for a in self._data(y)):
                return e
        return None

    def injected_peers(self):
        receivers = self._receivers()

        def peer_position(a):
            return a.name in self.positions and self._root(a.base) in receivers
        for stmts, defs, calls in self.scopes:
            for call in calls:
                for k in self.obstacle_args:
                    if k < len(call.args):
                        helper = self._defining_call(call.args[k], defs, peer_position)
                        if helper is not None:
                            return (self._collection(defs) or "peers"), helper.fn.id
        return None

    def target_roles(self):
        ir = self.ir
        receivers = self._receivers()
        for stmts, defs, calls in self.scopes:
            obstacle_names = {self._root(c.args[k]) for c in calls
                              for k in self.obstacle_args if k < len(c.args)} - {None}
            counters = {s.target.id for s in stmts if isinstance(s, ir.AugAssign)
                        and s.op == "+" and isinstance(s.target, ir.NameE)}
            writes = [s for s in stmts if isinstance(s, ir.Assign)
                      and isinstance(s.target, ir.Attr) and s.target.name in self.targets
                      and self._root(s.target.base) in receivers]
            for lead in writes:
                path, index_names = lead.value, set()
                while isinstance(path, ir.Index):
                    index_names |= {n.id for n in _walk_expr(path.index)
                                    if isinstance(n, ir.NameE)}
                    path = path.base
                if path is lead.value or not index_names & counters:
                    continue
                scene = {id(n) for name in obstacle_names for d in defs.get(name, ())
                         for e in self._slice(d, defs) for n in _walk_expr(e)}
                planners = []
                for e in self._slice(path, defs):
                    for n in _walk_expr(e):
                        if isinstance(n, ir.Call) and isinstance(n.fn, ir.NameE) \
                                and id(n) not in scene \
                                and any(self._root(a) in obstacle_names for a in n.args):
                            planners.append(n)
                def inputs(call):
                    return {id(m) for x in call.args if self._root(x) not in obstacle_names
                            for y in self._slice(x, defs) for m in _walk_expr(y)}
                sources = [p for p in planners
                           if not any(id(q) in inputs(p) for q in planners if q is not p)]
                if len(sources) != 1:
                    continue
                leader, field = self._root(lead.target.base), lead.target.name

                def leader_target(a):
                    return a.name == field and self._root(a.base) == leader
                follows = [self._defining_call(other.value, defs, leader_target)
                           for other in writes if self._root(other.target.base) != leader]
                follows = [c for c in follows if c is not None]
                if not follows:
                    continue
                formation = follows[0].fn.id
                first = any(isinstance(d, ir.Index) and isinstance(d.index, ir.Num)
                            and d.index.value == 0 for d in defs.get(leader, ())) or any(
                    isinstance(d, ir.Index) and isinstance(d.index, ir.Num) and d.index.value == 0
                    for _, other_defs, _ in self.scopes for d in other_defs.get(leader, ()))
                return sources[0].fn.id, field, formation, first, True
        return None


def _call_names(prog) -> set:
    from . import ir as _ir
    names = set()
    for mod in prog.modules.values():
        for s in _ir.walk_stmts(mod.body):
            for e in _stmt_exprs(s):
                for x in _walk_expr(e):
                    if isinstance(x, _ir.Call):
                        f = x.fn
                        if isinstance(f, _ir.NameE):
                            names.add(f.id)
                        elif isinstance(f, _ir.Attr):
                            names.add(f.name)
    return names


def _children_of(x):
    from . import ir as _ir
    import dataclasses as _dc
    out = []
    for f in _dc.fields(x):
        v = getattr(x, f.name)
        if isinstance(v, _ir.Expr):
            out.append(v)
        elif isinstance(v, list):
            out.extend(q for q in v if isinstance(q, _ir.Expr))
        elif isinstance(v, dict):
            out.extend(q for q in v.values() if isinstance(q, _ir.Expr))
    return out


def _stmt_exprs(s):
    from . import ir as _ir
    for f in ("target", "value", "cond", "iter"):
        v = getattr(s, f, None)
        if isinstance(v, _ir.Expr):
            yield v
    for d in getattr(s, "defaults", {}).values():
        if isinstance(d, _ir.Expr):
            yield d


def _walk_expr(e):
    from . import ir as _ir
    import dataclasses as _dc
    stack = [e]
    while stack:
        x = stack.pop()
        if not isinstance(x, _ir.Expr):
            continue
        yield x
        for f in _dc.fields(x):
            v = getattr(x, f.name)
            if isinstance(v, _ir.Expr):
                stack.append(v)
            elif isinstance(v, list):
                stack.extend(q for q in v if isinstance(q, _ir.Expr))
            elif isinstance(v, dict):
                stack.extend(q for q in v.values() if isinstance(q, _ir.Expr))
