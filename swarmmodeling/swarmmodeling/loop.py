from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .sexpr import (S, Add, Cond, Count, Dom, Fn, Idx, Mul, Obj, Prop, Reduce, Sym, Unk,
                    Vec, children, is_vec, is_zero, mul, walk)

SELF = Obj("self")


@dataclass
class Loop:
    sink: Optional[S] = None
    kind: str = "none"
    desc: str = "no command found"
    written: Optional[str] = None
    fields: Tuple[str, ...] = ()
    plant: Tuple[str, ...] = ()
    integrates: List[Tuple[str, str]] = field(default_factory=list)
    step: Optional[S] = None
    time_step: Optional[S] = None
    inputs: Dict[str, List[str]] = field(default_factory=dict)
    reason: str = ""


def _own_writes(heap) -> Dict[str, S]:
    return {key.name: value for key, value in heap.items()
            if isinstance(key, Prop) and key.base == SELF and value != key
            and value != Sym(key.name, "param")}


def _outside(value: S, through=frozenset()) -> bool:
    seen, pending = set(), [value]
    while pending:
        node = pending.pop()
        if id(node) in seen or id(node) in through:
            continue
        seen.add(id(node))
        if isinstance(node, (Dom, Reduce, Count)) or (isinstance(node, Obj) and node != SELF) \
                or (isinstance(node, Prop) and isinstance(node.base, Sym)):
            return True
        pending.extend(children(node))
    return False


def _opaque(value: S) -> bool:
    if isinstance(value, Cond):
        return _opaque(value.then) and _opaque(value.orelse)
    return isinstance(value, (Unk, Dom)) or (isinstance(value, Fn) and value.name.startswith("_"))


def _parts(value: S):
    return value.parts if isinstance(value, Add) else (value,)


def _inputs(value: S) -> Dict[str, List[str]]:
    own, bodies, params, scalars, bases = set(), {}, set(), set(), set()
    for node in walk(value):
        if isinstance(node, Dom):
            bases.add(node.name)
        if isinstance(node, Prop):
            if node.base == SELF:
                own.add(node.name)
            elif isinstance(node.base, Obj):
                bodies.setdefault(node.base.domain or node.base.name, set()).add(node.name)
            elif isinstance(node.base, Sym):
                bodies.setdefault(node.base.name, set()).add(node.name)
                bases.add(node.base.name)
        elif isinstance(node, Sym) and node.kind == "param":
            params.add(node.name)
        elif isinstance(node, Sym) and node.kind == "free":
            scalars.add(node.name)
    out = {"own state": sorted(own)}
    out.update(("fields of %s" % name, sorted(found)) for name, found in sorted(bodies.items()))
    out["parameters"] = sorted(params)
    out["other arguments"] = sorted(scalars - bases)
    return out


def _vector_argument(args) -> Optional[S]:
    if len(args) == 1:
        return args[0]
    comps = [a for a in args if isinstance(a, Idx)]
    if len(comps) >= 2 and len(comps) == len(args) and all(c.base == comps[0].base for c in comps):
        return comps[0].base
    vecs = [a for a in args if is_vec(a)]
    return vecs[0] if len(vecs) == 1 else None


def _time_step(plant_value: S, field_name: str, command: S, written: str) -> Optional[S]:
    from .execute import _delta_of
    delta = _delta_of(plant_value, Prop(SELF, field_name))
    if not isinstance(delta, Mul):
        return None
    ids = {id(p) for p in _parts(command)}
    carrier = [p for p in delta.parts if p == Prop(SELF, written) or id(p) in ids
               or any(id(x) in ids for x in walk(p))]
    rest = [p for p in delta.parts if p not in carrier]
    return mul(*rest) if carrier and rest and not any(is_vec(p) for p in rest) else None


def find_loop(heap, ret: Optional[S], calls) -> Loop:
    from .execute import _delta_of
    writes = _own_writes(heap)
    exits = [value for value in writes.values() if not _opaque(value)] \
        + [a for _, args in calls for a in args]
    if ret is not None:
        exits.append(ret)
    read = {p.name for value in exits for p in walk(value)
            if isinstance(p, Prop) and p.base == SELF}
    nodes = {name: {id(node) for node in walk(value)} for name, value in writes.items()}
    uses = {w: {x for x in writes if x != w and all(id(p) in nodes[w] for p in _parts(writes[x]))}
            for w in writes}
    on_loop = {name for name in writes if name in read}
    grown = True
    while grown:
        grown = False
        for name in writes:
            if name not in on_loop and any(name in uses[w] for w in on_loop):
                on_loop.add(name)
                grown = True
    loop = {name: value for name, value in writes.items() if name in on_loop}
    movers = [name for name, value in loop.items() if not _opaque(value)]
    coupled = [name for name in movers
               if _outside(loop[name], {id(p) for x in uses[name] for p in _parts(writes[x])})]
    plant = tuple(name for name in movers if name not in coupled)
    found = Loop(fields=tuple(sorted(loop)), plant=plant)
    if len(coupled) > 1:
        found.reason = ("several state writes on the feedback loop bring in outside "
                        "information: %s" % ", ".join(sorted(coupled)))
        return found
    if coupled:
        name = coupled[0]
        value = loop[name]
        found.written = name
        integrating = [other for other in plant if name in uses[other]
                       or any(x == Prop(SELF, name) for x in walk(loop[other]))]
        if integrating:
            found.sink, found.kind, found.desc = value, "rate", "state update self.%s" % name
            found.integrates = [(other, name) for other in integrating]
            found.time_step = _time_step(loop[integrating[0]], integrating[0], value, name)
            found.reason = "the coupled write on the feedback loop, a rate: %s integrates it" \
                % ", ".join(integrating)
        else:
            delta = _delta_of(value, Prop(SELF, name))
            found.step = delta if delta is not None and not is_zero(delta) else None
            found.sink, found.kind, found.desc = value, "state", "state update self.%s" % name
            found.reason = "the coupled write on the feedback loop" + (
                "; it adds to its own old value" if found.step is not None else "")
    elif ret is not None and not is_zero(ret):
        found.sink, found.kind, found.desc = ret, "return", "return value"
        found.reason = ("no coupled state write on a feedback loop; the returned value "
                        "is the law's output, and its loop closes in the caller")
    else:
        handed = [(name, vec) for name, args in calls
                  for vec in [_vector_argument(list(args))] if vec is not None and _outside(vec)]
        if len(handed) == 1:
            found.sink, found.kind, found.desc = handed[0][1], "call", "actuation %s(...)" % handed[0][0]
            found.reason = "the one external call handed a coupled vector; the loop closes outside the code"
        elif handed:
            found.reason = "several external calls are handed coupled vectors: %s" % \
                ", ".join(sorted(n for n, _ in handed))
    if found.sink is not None:
        found.inputs = _inputs(found.sink)
    return found
