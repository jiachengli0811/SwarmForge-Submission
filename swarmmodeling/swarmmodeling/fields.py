from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .sexpr import (S, Add, Clamp, Cmp, Cond, Const, Count, Dom, Fn, Idx,
                    Mul, NotS, Obj, Pow, Prop, Reduce, Sym, Unk, Vec, ZERO,
                    ONE, add, cond, fn, idx, mul, neg, pow_, source_of, sub, walk)

GR = Sym("_gr", "field")
GC = Sym("_gc", "field")


def has_field(s: S) -> bool:
    return any(x == GR or x == GC for x in walk(s))


def next_field_sym(s: S) -> Optional[S]:
    seen_gr = seen_gc = False
    for x in walk(s):
        if x == GR:
            seen_gr = True
        elif x == GC:
            seen_gc = True
    if seen_gr:
        return GR
    if seen_gc:
        return GC
    return None


def deriv(e: S, s: S, notes: List[str]) -> S:
    memo: Dict[int, S] = {}
    have: Dict[int, bool] = {}

    def contains(x: S) -> bool:
        h = have.get(id(x))
        if h is None:
            h = any(w == s for w in walk(x))
            have[id(x)] = h
        return h

    def d(x: S) -> S:
        r = memo.get(id(x))
        if r is not None:
            return r
        if not contains(x):
            out = ZERO
        elif x == s:
            out = ONE
        elif isinstance(x, Add):
            out = add(*[d(p) for p in x.parts])
        elif isinstance(x, Mul):
            terms = []
            for i, p in enumerate(x.parts):
                if not contains(p):
                    continue
                rest = [q for j, q in enumerate(x.parts) if j != i]
                terms.append(mul(*rest, d(p)) if rest else d(p))
            out = add(*terms) if terms else ZERO
        elif isinstance(x, Pow) and isinstance(x.exp, Const):
            out = mul(x.exp, pow_(x.base, Const(x.exp.v - 1.0)), d(x.base))
        elif isinstance(x, Vec):
            out = Vec(tuple(d(i) for i in x.items))
        elif isinstance(x, Cond):
            out = cond(x.test, d(x.then), d(x.orelse))
        elif isinstance(x, Fn) and x.name == "norm" and len(x.args) == 1 \
                and isinstance(x.args[0], Vec):
            v = x.args[0]
            num = add(*[mul(c, d(c)) for c in v.items])
            out = mul(num, pow_(fn("norm", v), Const(-1.0)))
        elif isinstance(x, Fn) and x.name == "sqrt" and len(x.args) == 1:
            out = mul(Const(0.5), pow_(x.args[0], Const(-0.5)), d(x.args[0]))
        elif isinstance(x, Fn) and x.name == "exp" and len(x.args) == 1:
            out = mul(x, d(x.args[0]))
        elif isinstance(x, Idx):
            out = idx(d(x.base), x.i)
        else:
            notes.append("field derivative left opaque: %r" % x)
            out = Unk("deriv", (x,))
        memo[id(x)] = out
        return out

    return d(e)


def gradient(f: S, notes: List[str]) -> Vec:
    return Vec((deriv(f, GR, notes), deriv(f, GC, notes)))


def affine_of(e: S) -> Tuple[S, S]:
    for x in walk(e):
        if not isinstance(x, Add):
            continue
        k = None
        offset_parts = []
        ok = True
        for p in x.parts:
            if isinstance(p, Mul) and k is None:
                cs = [q for q in p.parts if isinstance(q, Const)]
                if len(cs) == 1 and len(p.parts) == 2:
                    k = cs[0]
                    continue
                ok = False
                break
            elif isinstance(p, (Const, Sym)):
                offset_parts.append(p)
            else:
                ok = False
                break
        if ok and k is not None:
            return k, add(*offset_parts) if offset_parts else ZERO
    return ONE, ZERO


def occupancy(dom: S, scale: S, offset: S) -> Fn:
    return Fn("occupancy", (dom, scale, offset))


def bwdist(arg: S, notes: List[str]) -> S:
    occ = arg
    if isinstance(occ, Cmp):
        occ = occ.left if _is_occ(occ.left) else occ.right
    if not _is_occ(occ):
        notes.append("bwdist of unrecognised grid: %r" % arg)
        return Unk("bwdist", (arg,))
    dom, scale, offset = occ.args
    name = dom.name if isinstance(dom, Dom) else "set"
    var = (dom.var or "_o") if isinstance(dom, Dom) else "_o"
    o = Obj(var, domain=name, source=source_of(dom))
    pos = Prop(o, "pos", "position")
    comps = (add(GC, mul(Const(-1.0), scale, idx(pos, Const(0.0))), neg(offset)),
             add(GR, mul(Const(-1.0), scale, idx(pos, Const(1.0))), neg(offset)))
    d = sub(fn("norm", Vec(comps)), mul(scale, Prop(o, "size", "size")))
    notes.append(
        "distance transform over rasterised '%s' -> surface distance to the "
        "nearest element (dist - size = clearance); per-element sum with "
        "compact support approximates the nearest-only field" % name)
    return d


def _is_occ(x: S) -> bool:
    return isinstance(x, Fn) and x.name == "occupancy"


def slice_center(iv: S) -> S:
    if isinstance(iv, Fn) and iv.name == "_slice":
        lo, hi = iv.args
        bl, cl = _split_const(lo)
        bh, ch = _split_const(hi)
        if bl is not None and bl == bh:
            return add(bl, Const((cl + ch) / 2.0))
        if not isinstance(lo, Unk):
            return lo
        return hi
    return iv


def _split_const(x: S) -> Tuple[Optional[S], float]:
    if isinstance(x, Unk):
        return None, 0.0
    if isinstance(x, Const):
        return ZERO, x.v
    if isinstance(x, Add):
        c = 0.0
        rest = []
        for p in x.parts:
            if isinstance(p, Const):
                c += p.v
            else:
                rest.append(p)
        return (add(*rest) if rest else ZERO), c
    return x, 0.0
