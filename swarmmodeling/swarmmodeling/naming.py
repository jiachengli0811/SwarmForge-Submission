from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from . import fields as fieldcalc, ir
from .sexpr import (S, Add, Cmp, Cond, Const, Dom, Fn, Mul, NotS, Obj, Pow, Prop, Reduce, Sym,
                    contains, is_zero, walk)

SELF = Obj("self")


@dataclass
class Names:
    fields: Dict[str, str] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    membership: Dict[str, str] = field(default_factory=dict)
    collections: Set[str] = field(default_factory=set)
    agent: Optional[str] = None
    time_step: Optional[S] = None
    body_radius: Optional[float] = None
    reasons: Dict[str, str] = field(default_factory=dict)

    def role(self, name, value, reason):
        if name not in self.fields:
            self.fields[name] = value
            self.reasons["field:" + name] = reason

    def source(self, name, value, reason):
        if name not in self.sources:
            self.sources[name] = value
            self.reasons["source:" + name] = reason


def _root(domain) -> Optional[str]:
    while isinstance(domain, Dom) and isinstance(domain.base, Dom):
        domain = domain.base
    return domain.name if isinstance(domain, Dom) else None


def collections(roots) -> Set[str]:
    found = set()
    for expr in roots:
        for node in walk(expr):
            if isinstance(node, Reduce):
                name = _root(node.domain)
                if name:
                    found.add(name)
    return found


def _difference(expr):
    if not isinstance(expr, Add) or len(expr.parts) != 2:
        return None
    plus = [p for p in expr.parts if isinstance(p, Prop)]
    minus = [p.parts[1] for p in expr.parts if isinstance(p, Mul) and len(p.parts) == 2
             and p.parts[0] == Const(-1.0) and isinstance(p.parts[1], Prop)]
    return (plus[0], minus[0]) if len(plus) == 1 and len(minus) == 1 else None


def _reference(base):
    return base == SELF or (isinstance(base, Sym) and base.kind == "free")


def _pairings(roots, known):
    found = set()
    for expr in roots:
        for node in walk(expr):
            if isinstance(node, Add):
                aggregates = [p for p in node.parts if isinstance(p, Reduce)
                              and isinstance(p.body, Prop) and isinstance(p.body.base, Obj)
                              and p.body.base.name == p.var and _root(p.domain) in known]
                for red in aggregates:
                    for part in node.parts:
                        ref = part.parts[1] if isinstance(part, Mul) and len(part.parts) == 2 \
                            and part.parts[0] == Const(-1.0) else None
                        if isinstance(ref, Prop) and ref.name == red.body.name and _reference(ref.base):
                            found.add((ref.base, ref.name, _root(red.domain), red.body.name))
            if isinstance(node, Fn) and node.name in ("norm", "unit") and len(node.args) == 1:
                pair = _difference(node.args[0])
                if pair is not None:
                    for other, ref in (pair, pair[::-1]):
                        if ref.base == SELF and isinstance(other.base, Sym) and other.base.kind == "free":
                            found.add((SELF, ref.name, other.name, other.name))
            if not isinstance(node, Reduce):
                continue
            collection = _root(node.domain)
            if collection not in known:
                continue
            for inner in walk(node.body):
                if not (isinstance(inner, Fn) and inner.name in ("norm", "unit") and len(inner.args) == 1):
                    continue
                pair = _difference(inner.args[0])
                if pair is None:
                    continue
                for elem, ref in (pair, pair[::-1]):
                    if isinstance(elem.base, Obj) and elem.base.name == node.var and _reference(ref.base):
                        found.add((ref.base, ref.name, collection, elem.name))
    return found


def singletons(roots):
    return {collection for _, _, collection, accessor in _pairings(roots, set())
            if collection == accessor}


def _position(prop, names) -> bool:
    return prop.role == "position" or names.fields.get(prop.name) == "position"


def _distance(expr, names) -> bool:
    for node in walk(expr):
        if isinstance(node, Fn) and node.name == "norm":
            props = [p for p in walk(node) if isinstance(p, Prop)]
            if props and all(_position(p, names) for p in props):
                return True
    return False


def _params(expr):
    return {x.name for x in walk(expr) if isinstance(x, Sym) and x.kind == "param"}


def _numeric(expr, parameters):
    if isinstance(expr, Const):
        return expr.v
    if isinstance(expr, Sym) and expr.kind == "param":
        return parameters.get(expr.name)
    return None


def _quadratic_terms(expr, parameters):
    pending, seen = [(expr, 1.0)], set()
    while pending:
        node, coefficient = pending.pop()
        key = (id(node), coefficient)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(node, Add):
            pending.extend((p, coefficient) for p in node.parts)
        elif isinstance(node, Mul):
            variable = []
            for part in node.parts:
                number = _numeric(part, parameters)
                if number is None:
                    variable.append(part)
                else:
                    coefficient *= number
            if len(variable) == 1:
                pending.append((variable[0], coefficient))
        elif isinstance(node, Pow) and node.exp == Const(2):
            yield node, coefficient


def _compact_ranges(expr, names):
    found = set()
    for node in walk(expr):
        if isinstance(node, Cond):
            test, yes, no = node.test, node.then, node.orelse
        elif isinstance(node, Reduce) and node.guard is not None:
            test, yes, no = node.guard, node.body, Const(0.0)
        else:
            continue
        while isinstance(test, NotS):
            test, yes, no = test.x, no, yes
        if not isinstance(test, Cmp):
            continue
        if test.op in ("<", "<=") and is_zero(no):
            distance, cutoff, body = test.left, test.right, yes
        elif test.op in (">", ">=") and is_zero(yes):
            distance, cutoff, body = test.left, test.right, no
        else:
            continue
        if not _distance(distance, names):
            continue
        params = _params(cutoff)
        if len(params) != 1 or any(isinstance(x, (Prop, Obj, Dom)) for x in walk(cutoff)):
            continue
        reciprocal = any(isinstance(x, Pow) and x.base == distance and x.exp == Const(-1)
                         for x in walk(body)) and any(
            isinstance(x, Pow) and x.base == cutoff and x.exp == Const(-1) for x in walk(body))
        gaussian = any(isinstance(x, Fn) and x.name == "exp" and contains(x, distance)
                       and any(isinstance(p, Pow) and p.base == cutoff
                               and isinstance(p.exp, Const) and p.exp.v < 0 for p in walk(x))
                       for x in walk(body))
        if reciprocal or gaussian:
            found.update(params)
    return found


def _extents(roots, names):
    found = set()
    for expr in roots:
        for node in walk(expr):
            if not isinstance(node, Add):
                continue
            norms = [p for p in node.parts if isinstance(p, Fn) and p.name == "norm"
                     and _distance(p, names)]
            for norm in norms:
                bodies = {p.base for p in walk(norm) if isinstance(p, Prop)}
                for part in node.parts:
                    if isinstance(part, Mul) and Const(-1.0) in part.parts:
                        for prop in part.parts:
                            if isinstance(prop, Prop) and prop.base in bodies \
                                    and names.fields.get(prop.name) is None:
                                found.add(prop.name)
    return found


def controlled_extent(prog, entry, names) -> Optional[float]:
    function, _ = prog.functions[entry]
    extents = [name for name, role in names.fields.items() if role == "size"]
    init = prog.functions.get(entry.rsplit(".", 1)[0] + ".__init__") if function.cls else None
    if not init or len(extents) != 1:
        return None
    ctor, _ = init
    stored = [s.value for s in ir.walk_stmts(ctor.body)
              if isinstance(s, ir.Assign) and isinstance(s.target, ir.Attr)
              and isinstance(s.target.base, ir.NameE) and s.target.base.id == ctor.params[0]
              and s.target.name.lstrip("_") == extents[0]]
    if len(stored) != 1:
        return None
    if ir._const_value(stored[0]) is not None:
        return ir._const_value(stored[0])
    if not isinstance(stored[0], ir.NameE) or stored[0].id not in ctor.params:
        return None
    param = stored[0].id
    position = ctor.params.index(param) - 1
    values = set()
    default = ctor.defaults.get(param)
    for body, _, mod in prog.bodies():
        for stmt in ir.walk_stmts(body):
            for name in ("value", "target", "cond", "iter"):
                expr = getattr(stmt, name, None)
                if not isinstance(expr, ir.Expr):
                    continue
                for node in ir.walk_exprs(expr):
                    if not (isinstance(node, ir.Call) and isinstance(node.fn, ir.NameE)
                            and prog.find_ctor(mod, node.fn.id) == init):
                        continue
                    passed = node.kwargs.get(param) if param in node.kwargs else (
                        node.args[position] if position < len(node.args) else default)
                    values.add(None if passed is None else ir._const_value(passed))
    return values.pop() if len(values) == 1 and None not in values else None


def name_variables(prog, entry, heap, ret, calls, trace, params, loop, drawn_from=None,
                   free_params=(), peers=None) -> Names:
    names = Names(time_step=loop.time_step)
    if peers:
        names.source(peers, "agents", "the user's binding")
    roots = list(heap.values()) + [a for _, args in calls for a in args]
    if ret is not None:
        roots.append(ret)
    names.collections = collections(roots)

    for integrating, integrated in loop.integrates:
        names.role(integrating, "position", "the plant integrates the command into it")
        names.role(integrated, "velocity", "the command, a rate the plant integrates")

    if drawn_from in names.collections:
        names.source(drawn_from, "agents", "the collection the controlled drone is drawn from")

    names.collections |= singletons(roots)
    pairs = _pairings(roots, names.collections)
    if len({ref for ref, _, _, _ in pairs}) == 1:
        by_field = {}
        for _, accessor, collection, _ in pairs:
            by_field.setdefault(accessor, set()).add(collection)
        coupled = set().union(*by_field.values())
        if "position" not in names.fields.values():
            shared = [f for f, cs in by_field.items() if cs == coupled]
            if len(coupled) >= 2 and len(shared) == 1:
                names.role(shared[0], "position", "the one accessor compared with the reference "
                           "body across every coupled collection")
        position = next((f for f, r in names.fields.items() if r == "position"), None)
        for _, accessor, collection, element in pairs:
            if accessor == position and element != position:
                names.role(element, "position", "compared with the drone's position: the "
                           "position of %s" % collection)
        if "position" in names.fields.values() and "velocity" not in names.fields.values():
            others = [f for f in by_field if names.fields.get(f) is None]
            if len(others) == 1 and len(by_field) == 2:
                names.role(others[0], "velocity", "the second state accessor compared pairwise "
                           "with the reference body")
        if len(by_field) >= 2 and "agents" not in names.sources.values() \
                and all(names.fields.get(f) in ("position", "velocity") for f in by_field):
            peers = [c for c in coupled if all(c in cs for cs in by_field.values())]
            if len(peers) == 1:
                names.source(peers[0], "agents", "the only collection compared with the "
                             "reference body on every state accessor")

    if free_params and ret is not None and not any(
            isinstance(p, Prop) and p.base == SELF for p in walk(ret)):
        refs, paired = set(), set()
        for node in walk(ret):
            if isinstance(node, Prop) and names.fields.get(node.name) == "position" \
                    and isinstance(node.base, Sym) and node.base.name in free_params:
                refs.add(node.base.name)
            if isinstance(node, Fn) and node.name == "norm":
                local, domains = set(), False
                for prop in walk(node):
                    if isinstance(prop, Prop) and names.fields.get(prop.name) == "position":
                        if isinstance(prop.base, Sym) and prop.base.name in free_params:
                            local.add(prop.base.name)
                        elif isinstance(prop.base, Obj) and prop.base.domain:
                            domains = True
                if domains:
                    paired.update(local)
        if len(refs) == 1 and refs <= paired:
            names.agent = next(iter(refs))
            names.reasons["agent"] = "unique non-iterated reference in returned pairwise geometry"

    for extent in _extents(roots, names):
        names.role(extent, "size", "body extent subtracted from Euclidean position distance")

    updated = {p.name: value for p, value in heap.items()
               if isinstance(p, Prop) and p.base == SELF and value != p and contains(value, p)}
    sampled = set()
    for index, value in trace.samples:
        dependency_ids = {id(x) for x in walk(value) if isinstance(x, (Fn, Pow))}
        for prop in walk(index):
            if isinstance(prop, Prop) and prop.base == SELF and prop.name in updated and any(
                    id(x) in dependency_ids for x in walk(updated[prop.name])):
                sampled.add(prop.name)
    for name in sampled:
        names.role(name, "position", "state update receives a field gradient sampled at this accessor")
    if sampled:
        from .sexpr import Vec
        updates = Vec(tuple(updated[name] for name in sampled))
        centers = {}
        for potential in trace.potentials:
            for node, coefficient in _quadratic_terms(potential, params):
                if coefficient >= 0:
                    continue
                axes = {x for x in walk(node.base) if x in (fieldcalc.GR, fieldcalc.GC)}
                props = {p for p in walk(node.base) if isinstance(p, Prop) and p.base == SELF}
                if len(axes) == 1 and len(props) == 1:
                    prop = next(iter(props))
                    if prop.name not in updated and contains(updates, prop):
                        centers.setdefault(prop.name, set()).update(axes)
        for name, axes in centers.items():
            if axes == {fieldcalc.GR, fieldcalc.GC}:
                names.role(name, "target", "center of a quadratic field whose gradient drives "
                           "the position update")
        domains = {x.domain for x in walk(updates) if isinstance(x, Obj) and x.domain}
        for occupancy in trace.occupancies:
            for node in walk(occupancy):
                if isinstance(node, Fn) and node.name == "occupancy":
                    domain = node.args[0]
                    if isinstance(domain, Dom) and domain.name in domains:
                        names.collections.add(domain.name)
                        names.source(domain.name, "obstacles", "rasterized occupancy distance "
                                     "field drives the position update")

    for expr in roots + trace.potentials:
        for name in _compact_ranges(expr, names):
            if name not in names.membership:
                names.membership[name] = "range"
                names.reasons["membership:" + name] = \
                    "compact-support radial profile uses its cutoff as the distance scale"
    groups = {}
    for expr in roots:
        for node in walk(expr):
            if isinstance(node, Reduce) and isinstance(node.domain, Dom):
                signals = frozenset(names.fields.get(p.name) for p in walk(node.body)
                                    if isinstance(p, Prop))
                groups.setdefault(node.domain, set()).add(signals)
    for domain, signals in groups.items():
        if not any("position" in s for s in signals) or not any("velocity" in s for s in signals):
            continue
        for test in walk(domain.guard) if domain.guard is not None else ():
            if isinstance(test, Cmp) and test.op in ("<", "<=") and _distance(test.left, names) \
                    and isinstance(test.right, Sym) and test.right.kind == "param" \
                    and test.right.name not in names.membership:
                names.membership[test.right.name] = "range"
                names.reasons["membership:" + test.right.name] = \
                    "shared distance-filtered domain feeds position and velocity aggregates"

    names.body_radius = controlled_extent(prog, entry, names)
    return names
