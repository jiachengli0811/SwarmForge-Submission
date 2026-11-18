from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple


class S:
    __slots__ = ()


@dataclass(frozen=True)
class Const(S):
    v: float

    def __repr__(self):
        return "%g" % self.v


@dataclass(frozen=True)
class Sym(S):
    name: str
    kind: str = "free"

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Obj(S):
    name: str
    domain: Optional[str] = None
    source: Optional[str] = field(default=None, compare=False)

    def __repr__(self):
        return "@" + self.name


@dataclass(frozen=True)
class Prop(S):
    base: S
    name: str
    role: Optional[str] = field(default=None, compare=False)

    def __repr__(self):
        return "%r.%s" % (self.base, self.name)


@dataclass(frozen=True)
class Unk(S):
    tag: str
    args: Tuple[S, ...] = ()

    def __repr__(self):
        return "?%s%r" % (self.tag, list(self.args)) if self.args else "?" + self.tag


@dataclass(frozen=True)
class Vec(S):
    items: Tuple[S, ...]

    def __repr__(self):
        return "[%s]" % ", ".join(repr(i) for i in self.items)


@dataclass(frozen=True)
class Idx(S):
    base: S
    i: S

    def __repr__(self):
        return "%r[%r]" % (self.base, self.i)


@dataclass(frozen=True)
class Add(S):
    parts: Tuple[S, ...]

    def __repr__(self):
        return "(" + " + ".join(repr(p) for p in self.parts) + ")"


@dataclass(frozen=True)
class Mul(S):
    parts: Tuple[S, ...]

    def __repr__(self):
        return "(" + "*".join(repr(p) for p in self.parts) + ")"


@dataclass(frozen=True)
class Pow(S):
    base: S
    exp: S

    def __repr__(self):
        return "%r**%r" % (self.base, self.exp)


@dataclass(frozen=True)
class Fn(S):
    name: str
    args: Tuple[S, ...]

    def __repr__(self):
        return "%s(%s)" % (self.name, ", ".join(repr(a) for a in self.args))


@dataclass(frozen=True)
class Cmp(S):
    op: str
    left: S
    right: S

    def __repr__(self):
        return "(%r %s %r)" % (self.left, self.op, self.right)


@dataclass(frozen=True)
class AndS(S):
    parts: Tuple[S, ...]

    def __repr__(self):
        return "(" + " & ".join(repr(p) for p in self.parts) + ")"


@dataclass(frozen=True)
class OrS(S):
    parts: Tuple[S, ...]

    def __repr__(self):
        return "(" + " | ".join(repr(p) for p in self.parts) + ")"


@dataclass(frozen=True)
class NotS(S):
    x: S

    def __repr__(self):
        return "!%r" % self.x


@dataclass(frozen=True)
class Cond(S):
    test: S
    then: S
    orelse: S

    def __repr__(self):
        return "ite(%r, %r, %r)" % (self.test, self.then, self.orelse)


@dataclass(frozen=True)
class Dom(S):
    name: str
    base: Optional[S] = None
    var: Optional[str] = None
    guard: Optional[S] = None
    elt: Optional[S] = None
    source: Optional[str] = field(default=None, compare=False)
    cap: Optional[S] = None
    key: Optional[S] = None

    def __repr__(self):
        s = "{%s" % self.name
        if self.guard is not None:
            s += " | %r" % self.guard
        if self.cap is not None:
            s += " | first %r by %r" % (self.cap, self.key)
        return s + "}"


@dataclass(frozen=True)
class Reduce(S):
    op: str
    var: str
    domain: S
    body: S
    guard: Optional[S] = None

    def __repr__(self):
        g = " if %r" % self.guard if self.guard is not None else ""
        return "%s{%s in %r%s}(%r)" % (self.op.upper(), self.var, self.domain, g, self.body)


@dataclass(frozen=True)
class Count(S):
    var: str
    domain: S
    guard: Optional[S] = None

    def __repr__(self):
        g = " if %r" % self.guard if self.guard is not None else ""
        return "COUNT{%s in %r%s}" % (self.var, self.domain, g)


@dataclass(frozen=True)
class Clamp(S):
    x: S
    hi: Optional[S] = None
    lo: Optional[S] = None
    exact: bool = False

    def __repr__(self):
        kind = "setmag" if self.exact else "clamp"
        return "%s(%r, hi=%r, lo=%r)" % (kind, self.x, self.hi, self.lo)


ZERO = Const(0.0)
ONE = Const(1.0)


def _vec_len(parts: Iterable[S]) -> Optional[int]:
    for p in parts:
        if isinstance(p, Vec):
            return len(p.items)
    return None


def _broadcast(p: S, n: int) -> Tuple[S, ...]:
    if isinstance(p, Vec):
        return p.items
    return tuple([p] * n)


def add(*parts: S) -> S:
    flat: List[S] = []
    for p in parts:
        if isinstance(p, Add):
            flat.extend(p.parts)
        else:
            flat.append(p)
    if any(isinstance(p, Vec) for p in flat):
        nonzero = [p for p in flat if not is_zero(p)]
        if not nonzero:
            return ZERO
        flat = nonzero
    n = _vec_len(flat)
    if n is not None and not any(is_vec(p) and not isinstance(p, Vec) for p in flat):
        comps = [add(*[_broadcast(p, n)[i] for p in flat]) for i in range(n)]
        return Vec(tuple(comps))
    c = 0.0
    rest: List[S] = []
    for p in flat:
        if isinstance(p, Const):
            c += p.v
        else:
            rest.append(p)
    if c != 0.0 or not rest:
        rest.append(Const(c))
    if len(rest) == 1:
        return rest[0]
    return Add(tuple(rest))


def mul(*parts: S) -> S:
    flat: List[S] = []
    for p in parts:
        if isinstance(p, Mul):
            flat.extend(p.parts)
        else:
            flat.append(p)
    n = _vec_len(flat)
    if n is not None and not any(is_vec(p) and not isinstance(p, Vec) for p in flat):
        comps = [mul(*[_broadcast(p, n)[i] for p in flat]) for i in range(n)]
        return Vec(tuple(comps))
    c = 1.0
    rest: List[S] = []
    for p in flat:
        if isinstance(p, Const):
            c *= p.v
        else:
            rest.append(p)
    if c == 0.0:
        return ZERO
    if c != 1.0 or not rest:
        rest.insert(0, Const(c))
    if len(rest) == 1:
        return rest[0]
    return Mul(tuple(rest))


def neg(x: S) -> S:
    return mul(Const(-1.0), x)


def sub(a: S, b: S) -> S:
    return add(a, neg(b))


def pow_(b: S, e: S) -> S:
    if isinstance(b, Const) and isinstance(e, Const):
        try:
            return Const(float(b.v ** e.v))
        except (OverflowError, ValueError, ZeroDivisionError):
            pass
    if isinstance(e, Const):
        if e.v == 1.0:
            return b
        if e.v == 0.0:
            return ONE
    if isinstance(b, Pow) and isinstance(b.exp, Const) and isinstance(e, Const):
        return pow_(b.base, Const(b.exp.v * e.v))
    if isinstance(b, Fn) and b.name == "sqrt" and len(b.args) == 1 \
            and isinstance(e, Const) and e.v == 2.0:
        return b.args[0]
    if isinstance(b, Vec):
        return Vec(tuple(pow_(i, e) for i in b.items))
    return Pow(b, e)


def div(a: S, b: S) -> S:
    if isinstance(a, Vec) or isinstance(b, Vec):
        n = _vec_len([a, b])
        return Vec(tuple(div(_broadcast(a, n)[i], _broadcast(b, n)[i]) for i in range(n)))
    if isinstance(a, Const) and isinstance(b, Const) and b.v != 0:
        return Const(a.v / b.v)
    return mul(a, pow_(b, Const(-1.0)))


def idx(base: S, i: S) -> S:
    if isinstance(base, Vec) and isinstance(i, Const):
        k = int(i.v)
        if 0 <= k < len(base.items):
            return base.items[k]
        if -len(base.items) <= k < 0:
            return base.items[k]
    return Idx(base, i)


def fn(name: str, *args: S) -> S:
    if name == "sqrt" and len(args) == 1 and isinstance(args[0], Const) and args[0].v >= 0:
        return Const(math.sqrt(args[0].v))
    if name in ("norm", "abs") and len(args) == 1 and isinstance(args[0], Const):
        return Const(abs(args[0].v))
    return Fn(name, tuple(args))


def and_(*parts: S) -> S:
    flat: List[S] = []
    for p in parts:
        if isinstance(p, AndS):
            flat.extend(p.parts)
        elif isinstance(p, Const):
            if p.v == 0.0:
                return Const(0.0)
        else:
            flat.append(p)
    if not flat:
        return Const(1.0)
    if len(flat) == 1:
        return flat[0]
    return AndS(tuple(flat))


def cond(test: S, then: S, orelse: S) -> S:
    if isinstance(test, Const):
        return then if test.v else orelse
    if then == orelse:
        return then
    return Cond(test, then, orelse)


def cmp(op: str, l: S, r: S) -> S:
    if isinstance(l, Const) and isinstance(r, Const):
        f = {"<": l.v < r.v, "<=": l.v <= r.v, ">": l.v > r.v, ">=": l.v >= r.v,
             "==": l.v == r.v, "!=": l.v != r.v}.get(op)
        if f is not None:
            return Const(1.0 if f else 0.0)
    return Cmp(op, l, r)


def children(s: S) -> Tuple[S, ...]:
    if isinstance(s, (Add, Mul, AndS, OrS)):
        return s.parts
    if isinstance(s, Vec):
        return s.items
    if isinstance(s, Pow):
        return (s.base, s.exp)
    if isinstance(s, Fn):
        return s.args
    if isinstance(s, Idx):
        return (s.base, s.i)
    if isinstance(s, Prop):
        return (s.base,)
    if isinstance(s, Cmp):
        return (s.left, s.right)
    if isinstance(s, NotS):
        return (s.x,)
    if isinstance(s, Cond):
        return (s.test, s.then, s.orelse)
    if isinstance(s, Reduce):
        return (s.domain, s.body) + ((s.guard,) if s.guard is not None else ())
    if isinstance(s, Count):
        return (s.domain,) + ((s.guard,) if s.guard is not None else ())
    if isinstance(s, Clamp):
        return (s.x,) + tuple(x for x in (s.hi, s.lo) if x is not None)
    if isinstance(s, Dom):
        return tuple(x for x in (s.base, s.guard, s.elt, s.cap, s.key) if x is not None)
    if isinstance(s, Unk):
        return s.args
    return ()


def walk(s: S):
    seen = set()
    stack = [s]
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        yield x
        stack.extend(children(x))


def contains(s: S, target: S) -> bool:
    return any(x == target for x in walk(s))


def rebuild(s: S, f: Callable[[S], S]) -> S:
    def m(*cs):
        new = tuple(f(c) if c is not None else None for c in cs)
        same = all(a is b for a, b in zip(new, cs))
        return new, same

    if isinstance(s, Add):
        new, same = m(*s.parts)
        return s if same else add(*new)
    if isinstance(s, Mul):
        new, same = m(*s.parts)
        return s if same else mul(*new)
    if isinstance(s, AndS):
        new, same = m(*s.parts)
        return s if same else and_(*new)
    if isinstance(s, OrS):
        new, same = m(*s.parts)
        return s if same else OrS(new)
    if isinstance(s, Vec):
        new, same = m(*s.items)
        return s if same else Vec(new)
    if isinstance(s, Pow):
        new, same = m(s.base, s.exp)
        return s if same else pow_(*new)
    if isinstance(s, Fn):
        new, same = m(*s.args)
        return s if same else fn(s.name, *new)
    if isinstance(s, Idx):
        new, same = m(s.base, s.i)
        return s if same else idx(*new)
    if isinstance(s, Prop):
        new, same = m(s.base)
        return s if same else Prop(new[0], s.name, s.role)
    if isinstance(s, Cmp):
        new, same = m(s.left, s.right)
        return s if same else cmp(s.op, *new)
    if isinstance(s, NotS):
        new, same = m(s.x)
        return s if same else NotS(new[0])
    if isinstance(s, Cond):
        new, same = m(s.test, s.then, s.orelse)
        return s if same else cond(*new)
    if isinstance(s, Reduce):
        new, same = m(s.domain, s.body, s.guard)
        return s if same else Reduce(s.op, s.var, new[0], new[1], new[2])
    if isinstance(s, Count):
        new, same = m(s.domain, s.guard)
        return s if same else Count(s.var, new[0], new[1])
    if isinstance(s, Clamp):
        new, same = m(s.x, s.hi, s.lo)
        return s if same else Clamp(new[0], new[1], new[2], s.exact)
    if isinstance(s, Dom):
        new, same = m(s.base, s.guard, s.elt, s.cap, s.key)
        return s if same else Dom(s.name, new[0], s.var, new[1], new[2], s.source, new[3], new[4])
    if isinstance(s, Unk):
        new, same = m(*s.args)
        return s if same else Unk(s.tag, new)
    return s


def subst(s: S, mapping: Dict[S, S], _memo: Optional[Dict[int, S]] = None) -> S:
    if _memo is None:
        _memo = {}
    hit = _memo.get(id(s))
    if hit is not None:
        return hit
    if s in mapping:
        r = mapping[s]
    else:
        r = rebuild(s, lambda c: subst(c, mapping, _memo))
    _memo[id(s)] = r
    return r


def memo_map(s: S, f: Callable[[S], Optional[S]],
             _memo: Optional[Dict[int, S]] = None) -> S:
    if _memo is None:
        _memo = {}
    hit = _memo.get(id(s))
    if hit is not None:
        return hit
    t = rebuild(s, lambda c: memo_map(c, f, _memo))
    r = f(t)
    out = t if r is None else r
    _memo[id(s)] = out
    return out


def is_zero(s: S) -> bool:
    if isinstance(s, Const):
        return s.v == 0.0
    if isinstance(s, Vec):
        return all(is_zero(i) for i in s.items)
    return False


_REPR_DEPTH = 8
_REPR_WIDTH = 8


def srepr(s: S, d: int = _REPR_DEPTH) -> str:
    if d <= 0:
        return "..."
    if isinstance(s, Const):
        return "%g" % s.v
    if isinstance(s, Sym):
        return s.name
    if isinstance(s, Obj):
        return "@" + s.name
    if isinstance(s, Prop):
        return "%s.%s" % (srepr(s.base, d - 1), s.name)
    if isinstance(s, Unk):
        return "?" + s.tag
    kids = children(s)[:_REPR_WIDTH]
    ks = [srepr(c, d - 1) for c in kids]
    if isinstance(s, Add):
        return "(" + " + ".join(ks) + ")"
    if isinstance(s, Mul):
        return "(" + "*".join(ks) + ")"
    if isinstance(s, Vec):
        return "[" + ", ".join(ks) + "]"
    if isinstance(s, Pow):
        return "%s**%s" % (ks[0], ks[1])
    if isinstance(s, Fn):
        return "%s(%s)" % (s.name, ", ".join(ks))
    if isinstance(s, Idx):
        return "%s[%s]" % (ks[0], ks[1])
    if isinstance(s, Cmp):
        return "(%s %s %s)" % (ks[0], s.op, ks[1])
    if isinstance(s, AndS):
        return "(" + " & ".join(ks) + ")"
    if isinstance(s, OrS):
        return "(" + " | ".join(ks) + ")"
    if isinstance(s, NotS):
        return "!" + ks[0]
    if isinstance(s, Cond):
        return "ite(%s, %s, %s)" % tuple(ks[:3])
    if isinstance(s, Reduce):
        g = " if " + srepr(s.guard, d - 1) if s.guard is not None else ""
        return "%s{%s in %s%s}(%s)" % (s.op.upper(), s.var,
                                       srepr(s.domain, d - 1), g,
                                       srepr(s.body, d - 1))
    if isinstance(s, Count):
        g = " if " + srepr(s.guard, d - 1) if s.guard is not None else ""
        return "COUNT{%s in %s%s}" % (s.var, srepr(s.domain, d - 1), g)
    if isinstance(s, Clamp):
        return "%s(%s, hi=%s, lo=%s)" % (
            "setmag" if s.exact else "clamp", srepr(s.x, d - 1),
            srepr(s.hi, d - 1) if s.hi is not None else "None",
            srepr(s.lo, d - 1) if s.lo is not None else "None")
    if isinstance(s, Dom):
        g = " | " + srepr(s.guard, d - 1) if s.guard is not None else ""
        return "{%s%s}" % (s.name, g)
    return object.__repr__(s)


for _cls in (Const, Sym, Obj, Prop, Unk, Vec, Idx, Add, Mul, Pow, Fn, Cmp,
             AndS, OrS, NotS, Cond, Dom, Reduce, Count, Clamp):
    _cls.__repr__ = srepr


SCALAR_FNS = {"norm", "dot", "sqrt", "exp", "abs", "angle", "len", "cos",
              "sin", "acos", "degrees", "radians", "min", "max"}


def is_vec(s: S) -> bool:
    if isinstance(s, Vec):
        return True
    if isinstance(s, Prop):
        return s.role in ("position", "velocity", "target", "vector")
    if isinstance(s, (Add, Mul)):
        return any(is_vec(p) for p in s.parts)
    if isinstance(s, Cond):
        return is_vec(s.then) or is_vec(s.orelse)
    if isinstance(s, Reduce):
        return is_vec(s.body)
    if isinstance(s, Clamp):
        return True
    if isinstance(s, Fn):
        if s.name == "unit":
            return True
        if s.name in SCALAR_FNS:
            return False
    return False


def source_of(s: S) -> Optional[str]:
    while isinstance(s, Dom):
        if s.source is not None:
            return s.source
        s = s.base
    return s.source if isinstance(s, Obj) else None
