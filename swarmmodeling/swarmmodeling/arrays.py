from __future__ import annotations

from dataclasses import replace
from typing import Optional

from . import ir
from .sexpr import (S, Add, AndS, Cmp, Cond, Const, Count, Dom, Fn, Idx, Mul, NotS, Obj, OrS,
                    Pow, Prop, Reduce, Sym, Unk, Vec, ZERO, add, and_, cmp, cond, fn, idx,
                    is_vec, is_zero, subst, walk)


def is_map(v) -> bool:
    return isinstance(v, Dom) and v.elt is not None


def is_list(v) -> bool:
    return isinstance(v, Dom) and v.elt is None


def element(d: Dom) -> Obj:
    return Obj(d.var or "_e", domain=d.name)


def token(v) -> Optional[Obj]:
    if isinstance(v, Obj) and v.domain:
        return v
    if isinstance(v, Add):
        objs = [p for p in v.parts if isinstance(p, Obj) and p.domain]
        rest = [p for p in v.parts if p not in objs]
        if len(objs) == 1 and all(isinstance(p, Const) for p in rest):
            return objs[0]
    return None


def at(m: Dom, tok: S) -> S:
    return subst(m.elt, {element(m): tok}) if m.var else m.elt


def _over(m: Dom, x: S) -> S:
    return x


def _slice(v) -> Optional[tuple]:
    if isinstance(v, Fn) and v.name == "_slice":
        lo, hi = v.args
        lo = None if isinstance(lo, Unk) else lo
        hi = None if isinstance(hi, Unk) else hi
        return lo, hi
    return None


def _number(v) -> Optional[float]:
    return v.v if isinstance(v, Const) else None


def rows(value: S, sel: S, dims: int) -> S:
    if is_zero(value):
        return ZERO
    rng = _slice(sel)
    if rng is not None:
        lo, hi = (_number(rng[0]) if rng[0] is not None else 0.0), \
            (_number(rng[1]) if rng[1] is not None else None)
        if isinstance(value, Vec):
            return Vec(value.items[int(lo or 0):None if hi is None else int(hi)])
        if (lo or 0) == 0 and (hi is None or hi == dims):
            return value
        return Fn("_rows", (value, Const(lo or 0), Const(hi) if hi is not None else Unk("end")))
    r = _number(sel)
    if r is not None:
        if isinstance(value, Vec):
            return value.items[int(r)] if 0 <= int(r) < len(value.items) else Unk("row")
        return ZERO if r >= dims else idx(value, Const(r))
    return idx(value, sel)


def _accessor(matrix: S, tok: Obj) -> Optional[S]:
    if isinstance(matrix, Prop):
        return Prop(tok, matrix.name)
    if isinstance(matrix, Idx) and isinstance(matrix.base, Prop) and isinstance(matrix.i, Const):
        return Prop(tok, "%s[%d]" % (matrix.base.name, int(matrix.i.v)))
    if isinstance(matrix, Fn) and matrix.name == "_rows" and isinstance(matrix.args[0], Prop):
        lo, hi = matrix.args[1], matrix.args[2]
        return Prop(tok, "%s[%s:%s]" % (matrix.args[0].name, int(lo.v),
                                          int(hi.v) if isinstance(hi, Const) else ""), "vector")
    return None


def collection_of(value: S) -> Optional[str]:
    if isinstance(value, Dom):
        d = value
        while isinstance(d.base, Dom):
            d = d.base
        return d.name
    if isinstance(value, Prop) and value.role != "vector":
        return value.name
    if isinstance(value, Sym) and value.kind == "free":
        return value.name
    if isinstance(value, Idx) and isinstance(value.base, Prop):
        return value.base.name
    if isinstance(value, Fn) and value.name == "_rows" and isinstance(value.args[0], Prop):
        return value.args[0].name
    return None


def _root(d: Dom) -> str:
    while isinstance(d.base, Dom):
        d = d.base
    return d.name


def _core(v: S, name: str) -> Optional[Fn]:
    if isinstance(v, Fn) and v.name == name:
        return v
    if isinstance(v, Add):
        cores = [p for p in v.parts if isinstance(p, Fn) and p.name == name]
        if len(cores) == 1 and all(isinstance(p, Const) for p in v.parts if p is not cores[0]):
            return cores[0]
    return None


def index(base: S, sub: S, dims: int) -> Optional[S]:
    tok = token(sub)
    order = _core(base, "_order")
    if order is not None and _slice(sub) is not None:
        return Fn("_prefix", (order, _slice(sub)[1]))
    if is_map(base):
        if tok is not None:
            return at(base, tok)
        if is_map(sub):
            return replace(base, guard=and_(*[g for g in (base.guard, at(sub, element(base))) if g is not None]))
        if is_list(sub) or (isinstance(sub, Add) and is_list(_list_part(sub))):
            lst = sub if is_list(sub) else _list_part(sub)
            guard = subst(lst.guard, {element(lst): element(base)}) if lst.guard is not None else None
            key = subst(lst.key, {element(lst): element(base)}) if lst.key is not None else None
            return replace(base, guard=guard, cap=lst.cap, key=key)
        if isinstance(base.elt, Obj) and isinstance(sub, Const):
            return base.elt
        if _slice(sub) is not None or isinstance(sub, Const):
            return replace(base, elt=rows(base.elt, sub, dims))
        return None
    if is_list(base):
        if tok is not None:
            return tok
        if is_map(sub):
            mask = at(sub, element(base))
            return replace(base, guard=and_(*[g for g in (base.guard, mask) if g is not None]))
        prefix = _core(sub, "_prefix")
        if prefix is not None:
            order, cap = prefix.args
            ordered = order.args[0]
            key = subst(order.args[1], {element(ordered): element(base)})
            return replace(base, cap=cap, key=key)
        return None
    if isinstance(base, Fn) and base.name == "_zeros":
        if _slice(sub) is not None:
            return base
        if tok is not None:
            return ZERO
        return None
    if tok is not None:
        named = _accessor(base, tok) if isinstance(base, (Idx, Fn)) or (
            isinstance(base, Prop) and isinstance(base.base, Sym)) else None
        if named is not None:
            return named
        if collection_of(base) is not None:
            return tok
    if isinstance(base, Prop) and isinstance(base.base, Sym) and _slice(sub) is not None:
        lo, hi = _slice(sub)
        return Fn("_rows", (base, lo if lo is not None else ZERO, hi if hi is not None else Unk("end")))
    if isinstance(base, Prop) and _slice(sub) is not None:
        return rows(base, sub, dims)
    return None


def _list_part(v):
    return next((p for p in v.parts if is_list(p)), None) if isinstance(v, Add) else None


def _fn_part(v, name):
    return next((p for p in v.parts if isinstance(p, Fn) and p.name == name), None)


def store(cur: S, subs, value: S, dims: int) -> Optional[S]:
    if not (isinstance(cur, Fn) and cur.name == "_zeros") and not is_map(cur):
        return None
    *row_subs, col = subs
    tok = token(col)
    if tok is not None:
        new = value
        if isinstance(new, Prop) and new.role is None and isinstance(cur, Fn) \
                and isinstance(cur.args[0], Const) and cur.args[0].v > 1:
            new = Prop(new.base, new.name, "vector")
        if row_subs and (is_map(cur) or isinstance(cur, Fn)):
            old = at(cur, tok) if is_map(cur) else ZERO
            rng = _slice(row_subs[0])
            if rng is not None and is_zero(old) and (_number(rng[0]) or 0) == 0 \
                    and (rng[1] is None or _number(rng[1]) == dims):
                pass
            elif rng is not None and isinstance(old, Vec):
                lo = int(_number(rng[0]) or 0)
                hi = _number(rng[1]) if rng[1] is not None else None
                if lo == 0 and hi is None:
                    pass
                elif lo == 0 and hi == dims and not isinstance(new, Vec) \
                        and all(is_zero(x) for x in old.items[dims:]):
                    pass
                else:
                    items = list(old.items)
                    part = new.items if isinstance(new, Vec) else (new,)
                    items[lo:lo + len(part)] = part
                    new = Vec(tuple(items))
        if is_map(cur) and cur.name != tok.domain:
            return replace(cur, elt=new)
        rows_ = cur.args[0] if isinstance(cur, Fn) else None
        return Dom(tok.domain, var=tok.name, elt=new,
                   source=("rows:%d" % int(rows_.v)) if isinstance(rows_, Const) else cur.source
                   if is_map(cur) else None)
    if is_map(col) and is_map(cur):
        mask = at(col, element(cur))
        new = at(value, element(cur)) if is_map(value) else value
        return replace(cur, elt=cond(mask, new, cur.elt))
    return None


def binop(op, l: S, r: S, combine) -> Optional[S]:
    lm, rm = is_map(l), is_map(r)
    if not (lm or rm):
        return None
    if lm and rm:
        return replace(l, elt=combine(op, l.elt, at(r, element(l))))
    if lm:
        return replace(l, elt=combine(op, l.elt, r))
    return replace(r, elt=combine(op, l, r.elt))


def unary(f, v: S) -> Optional[S]:
    return replace(v, elt=f(v.elt)) if is_map(v) else None


def _norm_squared(x: S) -> Optional[S]:
    if isinstance(x, Pow) and x.exp == Const(2.0) and (is_vec(x.base) or isinstance(x.base, Vec)):
        return Pow(fn("norm", x.base), Const(2.0))
    return None


def row_sum(x: S) -> S:
    if isinstance(x, Vec):
        return add(*x.items)
    squared = _norm_squared(x)
    return squared if squared is not None else fn("sum", x)


def sqrt(x: S) -> S:
    if isinstance(x, Pow) and x.exp == Const(2.0) and isinstance(x.base, Fn) and x.base.name == "norm":
        return x.base
    return fn("sqrt", x)


def numpy_call(name: str, args, kwargs, counts) -> Optional[S]:
    a = args[0] if args else None
    axis = kwargs.get("axis")
    if name == "sqrt":
        return unary(sqrt, a) if is_map(a) else sqrt(a)
    if name in ("exp", "abs") and is_map(a):
        return unary(lambda x: fn(name, x), a)
    if name == "sum":
        if is_map(a):
            if axis == Const(0.0) or (axis is None and (is_vec(a.elt) or isinstance(a.elt, Vec)
                                                        or _norm_squared(a.elt) is not None)):
                return unary(row_sum, a)
            body = a.elt
            if isinstance(body, (Cmp, AndS, OrS, NotS)):
                return Count(a.var, Dom(a.name, var=a.var, guard=a.guard), and_(*[g for g in (a.guard, body) if g is not None]))
            return Reduce("sum", a.var, Dom(a.name, var=a.var, guard=a.guard, cap=a.cap, key=a.key), body)
        if axis is None and a is not None and _norm_squared(a) is not None:
            return _norm_squared(a)
        return None
    if name == "transpose" and (isinstance(a, Dom) or a is not None):
        return a
    if name == "flatnonzero" and is_map(a):
        return Dom(a.name, var=a.var, guard=and_(*[g for g in (a.guard, a.elt) if g is not None]),
                   cap=a.cap, key=a.key)
    if name == "argsort" and is_map(a):
        return Fn("_order", (Dom(a.name, var=a.var, guard=a.guard, cap=a.cap, key=a.key), a.elt))
    if name == "tile" and is_map(a) and not (is_vec(a.elt) or isinstance(a.elt, Vec)):
        return a
    if name == "zeros" and isinstance(a, Vec) and len(a.items) == 2 and not all(
            isinstance(x, Const) for x in a.items):
        n = a.items[1]
        if isinstance(n, Const) and not isinstance(a.items[0], Const):
            return None
        return Fn("_zeros", (a.items[0], n))
    if name == "size":
        if isinstance(a, Dom):
            return Count(a.var or "_e", a, a.guard)
        if isinstance(a, (Const, Sym)) and (not isinstance(a, Sym) or a.kind == "param"):
            return Const(1.0)
    return None


def length(a: S) -> Optional[S]:
    if isinstance(a, Dom):
        n = Count(a.var or "_e", Dom(a.name, var=a.var), a.guard)
        return fn("min", n, a.cap) if a.cap is not None else n
    return None


def index_list(start: S, stop: S, counts) -> Optional[S]:
    n = stop
    if isinstance(stop, Add):
        consts = [p for p in stop.parts if isinstance(p, Const)]
        rest = [p for p in stop.parts if not isinstance(p, Const)]
        if len(rest) == 1:
            n = rest[0]
    name = counts.get(n)
    return Dom(name, var="_" + name) if name else None


def _existence(test: S) -> Optional[Dom]:
    if isinstance(test, Cmp) and test.op in ("!=", ">") and test.right == ZERO:
        left = test.left
        if isinstance(left, Fn) and left.name == "min":
            left = left.args[0]
        if isinstance(left, Count):
            return left.domain
    return None


def _zero_map(like: Dom, zeros: Fn) -> Optional[Dom]:
    if isinstance(zeros, Fn) and zeros.name == "_zeros":
        return replace(like, elt=ZERO)
    return None


def merge(test: S, then: S, other: S) -> Optional[S]:
    if isinstance(test, NotS):
        return merge(test.x, other, then)
    if then == other:
        return then
    if isinstance(test, Cmp) and test.op == ">" and test.left == other and test.right == then:
        return fn("min", other, then)
    if is_list(then) and is_list(other) and isinstance(test, Cmp) and test.op == ">" \
            and then.cap is not None and replace(then, cap=None, key=None) == replace(other, cap=None, key=None):
        return then
    if is_map(then) and not is_map(other):
        other = _zero_map(then, other) or other
    if is_map(other) and not is_map(then):
        then = _zero_map(other, then) or then
    if is_map(then) and is_map(other) and then.name == other.name:
        if _existence(test) is not None:
            return then
        return replace(then, elt=cond(test, then.elt, at(other, element(then))))
    return None


def subscripted(body, var: str):
    found = []
    for stmt in ir.walk_stmts(body):
        for name in ("value", "target", "cond", "iter"):
            expr = getattr(stmt, name, None)
            if not isinstance(expr, ir.Expr):
                continue
            for node in ir.walk_exprs(expr):
                if not isinstance(node, ir.Index):
                    continue
                i = node.index
                if isinstance(i, ir.BinOp) and i.op in ("+", "-") and isinstance(i.right, ir.Num):
                    i = i.left
                if isinstance(i, ir.NameE) and i.id == var:
                    found.append(node.base)
    return found
