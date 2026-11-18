from __future__ import annotations

from typing import List, Optional

from .sexpr import (S, Add, AndS, Clamp, Cmp, Cond, Const, Count, Dom, Fn,
                    Idx, Mul, NotS, Obj, OrS, Pow, Prop, Reduce, Sym, Unk,
                    Vec, add, and_, children, cond, fn, idx, is_vec, is_zero,
                    memo_map, mul, neg, pow_, rebuild, subst, walk)

_AXF = Sym("_axf", "axis")


def rewrite(s: S, notes: Optional[List[str]] = None) -> S:
    notes = notes if notes is not None else []
    for _ in range(30):
        t = _rw(s, notes)
        if t is s or t == s:
            return s
        s = t
    notes.append("rewrite did not reach a fixed point")
    return s


def _rw(s: S, notes: List[str]) -> S:
    def step(node: S) -> Optional[S]:
        for rule in _RULES:
            r = rule(node, notes)
            if r is not None:
                return r
        return None
    return memo_map(s, step)


def axis_fold_vec(v: Vec) -> Optional[S]:
    if len(v.items) < 2:
        return None
    hit = [False]
    folded = _antiunify(list(v.items), hit)
    if folded is None or not hit[0]:
        return None
    return folded


def _antiunify(xs: List[S], hit: List[bool]) -> Optional[S]:
    first = xs[0]
    if all(x is first or x == first for x in xs[1:]):
        return first
    if all(isinstance(x, Idx) and isinstance(x.i, Const) and int(x.i.v) == i
           for i, x in enumerate(xs)):
        base = _antiunify([x.base for x in xs], hit)
        if base is None:
            return None
        hit[0] = True
        return base
    t = type(first)
    if not all(type(x) is t for x in xs):
        return None

    def au_opt(cols):
        if all(c is None for c in cols):
            return None, True
        if any(c is None for c in cols):
            return None, False
        r = _antiunify(list(cols), hit)
        return r, r is not None

    if isinstance(first, Reduce):
        if any(x.op != first.op or x.var != first.var for x in xs):
            return None
        d, ok1 = au_opt([x.domain for x in xs])
        b, ok2 = au_opt([x.body for x in xs])
        g, ok3 = au_opt([x.guard for x in xs])
        return Reduce(first.op, first.var, d, b, g) if ok1 and ok2 and ok3 else None
    if isinstance(first, Count):
        if any(x.var != first.var for x in xs):
            return None
        d, ok1 = au_opt([x.domain for x in xs])
        g, ok2 = au_opt([x.guard for x in xs])
        return Count(first.var, d, g) if ok1 and ok2 else None
    if isinstance(first, Clamp):
        if any(x.exact != first.exact for x in xs):
            return None
        v, ok1 = au_opt([x.x for x in xs])
        h, ok2 = au_opt([x.hi for x in xs])
        l, ok3 = au_opt([x.lo for x in xs])
        return Clamp(v, h, l, first.exact) if ok1 and ok2 and ok3 else None
    if isinstance(first, Dom):
        if any(x.name != first.name or x.var != first.var for x in xs):
            return None
        b, ok1 = au_opt([x.base for x in xs])
        g, ok2 = au_opt([x.guard for x in xs])
        e, ok3 = au_opt([x.elt for x in xs])
        return Dom(first.name, b, first.var, g, e, first.source) if ok1 and ok2 and ok3 else None

    if not _same_shape(xs):
        return None
    kid_rows = [children(x) for x in xs]
    if len({len(k) for k in kid_rows}) != 1 or not kid_rows[0]:
        return None
    new_kids = []
    for col in zip(*kid_rows):
        k = _antiunify(list(col), hit)
        if k is None:
            return None
        new_kids.append(k)
    return _reconstruct(first, new_kids)


def _same_shape(xs: List[S]) -> bool:
    a = xs[0]
    if isinstance(a, Fn):
        return all(x.name == a.name for x in xs)
    if isinstance(a, Cmp):
        return all(x.op == a.op for x in xs)
    if isinstance(a, Prop):
        return all(x.name == a.name for x in xs)
    if isinstance(a, (Add, Mul, AndS, OrS, Vec, Pow, Idx, NotS, Cond)):
        return True
    return False


def _reconstruct(s: S, kids: List[S]) -> Optional[S]:
    if isinstance(s, Add):
        return add(*kids)
    if isinstance(s, Mul):
        return mul(*kids)
    if isinstance(s, AndS):
        return and_(*kids)
    if isinstance(s, OrS):
        return OrS(tuple(kids))
    if isinstance(s, Vec):
        return Vec(tuple(kids))
    if isinstance(s, Pow):
        return pow_(kids[0], kids[1])
    if isinstance(s, Fn):
        return fn(s.name, *kids)
    if isinstance(s, Idx):
        return idx(kids[0], kids[1])
    if isinstance(s, Prop):
        return Prop(kids[0], s.name, s.role)
    if isinstance(s, Cmp):
        return Cmp(s.op, kids[0], kids[1])
    if isinstance(s, NotS):
        return NotS(kids[0])
    if isinstance(s, Cond):
        return cond(kids[0], kids[1], kids[2])
    return None


def axis_unfold(s: S, axis: S, notes: List[str]) -> S:
    def step(x: S) -> Optional[S]:
        if isinstance(x, Idx) and x.i == axis:
            return x.base
        return None
    out = memo_map(s, step)
    if any(x == axis for x in walk(out)):
        notes.append("axis loop not fully vectorisable; left symbolic axis")
    return out


def _r_axis_fold(s: S, notes) -> Optional[S]:
    if isinstance(s, Vec):
        return axis_fold_vec(s)
    return None


def _r_sqrt_norm(s: S, notes) -> Optional[S]:
    if not (isinstance(s, Fn) and s.name == "sqrt" and len(s.args) == 1):
        return None
    a = s.args[0]
    parts = a.parts if isinstance(a, Add) else (a,)
    comps = []
    for p in parts:
        if isinstance(p, Pow) and isinstance(p.exp, Const) and p.exp.v == 2.0:
            comps.append(p.base)
        else:
            return None
    if len(comps) < 2:
        return None
    return fn("norm", Vec(tuple(comps)))


def _r_sqrt_dot(s: S, notes) -> Optional[S]:
    if isinstance(s, Fn) and s.name == "sqrt" and len(s.args) == 1 and isinstance(s.args[0], Fn) \
            and s.args[0].name == "dot" and len(s.args[0].args) == 2 and s.args[0].args[0] == s.args[0].args[1]:
        return fn("norm", s.args[0].args[0])
    return None


def _r_norm_of_vec(s: S, notes) -> Optional[S]:
    if isinstance(s, Fn) and s.name == "norm" and len(s.args) == 1 \
            and isinstance(s.args[0], Vec):
        folded = axis_fold_vec(s.args[0])
        if folded is not None:
            return fn("norm", folded)
    return None


def _r_unit(s: S, notes) -> Optional[S]:
    if not isinstance(s, Mul):
        return None
    parts = list(s.parts)
    for i, p in enumerate(parts):
        if isinstance(p, Pow) and isinstance(p.exp, Const) and p.exp.v == -1.0 \
                and isinstance(p.base, Fn) and p.base.name == "norm":
            v = p.base.args[0]
            if v in parts:
                j = parts.index(v)
                rest = [q for k, q in enumerate(parts) if k not in (i, j)]
                return mul(*rest, fn("unit", v)) if rest else fn("unit", v)
            for j, q in enumerate(parts):
                if j != i and _negation(q, v):
                    rest = [r for k, r in enumerate(parts) if k not in (i, j)]
                    return mul(*rest, fn("unit", q)) if rest else fn("unit", q)
            for j, q in enumerate(parts):
                if j == i or isinstance(q, Const):
                    continue
                k = _prop_ratio(q, v)
                if k is not None:
                    rest = [r for m, r in enumerate(parts) if m not in (i, j)]
                    return mul(*rest, Const(k), fn("unit", v))
    return None


def _negation(a: S, b: S) -> bool:
    pa = list(a.parts) if isinstance(a, Add) else [a]
    pb = [neg(p) for p in (b.parts if isinstance(b, Add) else [b])]
    if len(pa) != len(pb):
        return False
    for p in pa:
        if p not in pb:
            return False
        pb.remove(p)
    return True


def _coefficient(part: S):
    if isinstance(part, Mul) and part.parts and isinstance(part.parts[0], Const):
        return part.parts[0].v, mul(*part.parts[1:])
    return 1.0, part


def _r_like_terms(s: S, notes) -> Optional[S]:
    if not isinstance(s, Add):
        return None
    atoms = [_coefficient(p)[1] for p in s.parts if not isinstance(p, Const)]
    if len(atoms) == len(set(atoms)):
        return None
    totals, order = {}, []
    for p in s.parts:
        c, atom = (p.v, Const(1.0)) if isinstance(p, Const) else _coefficient(p)
        if atom not in totals:
            order.append(atom)
            totals[atom] = 0.0
        totals[atom] += c
    return add(*(mul(Const(totals[atom]), atom) for atom in order if totals[atom] != 0.0))


def _prop_ratio(a: S, b: S):
    if a == b:
        return 1.0
    az, bz = is_zero(a), is_zero(b)
    if az or bz:
        return 1.0 if az and bz else None
    ka, ra = _split_mul_const(a)
    kb, rb = _split_mul_const(b)
    if kb == 0.0:
        return None
    if ra == rb:
        return ka / kb
    if type(ra) is not type(rb):
        return None
    if isinstance(ra, Add) and len(ra.parts) == len(rb.parts):
        k = None
        for pa, pb in zip(ra.parts, rb.parts):
            r = _prop_ratio(pa, pb)
            if r is None:
                return None
            if k is None:
                k = r
            elif abs(r - k) > 1e-9:
                return None
        return ka / kb * (1.0 if k is None else k)
    if isinstance(ra, Cond) and ra.test == rb.test:
        rt = _prop_ratio(ra.then, rb.then)
        ro = _prop_ratio(ra.orelse, rb.orelse)
        if rt is None or ro is None:
            return None
        if is_zero(ra.then) and is_zero(rb.then):
            k = ro
        elif is_zero(ra.orelse) and is_zero(rb.orelse):
            k = rt
        elif abs(rt - ro) <= 1e-9:
            k = rt
        else:
            return None
        return ka / kb * k
    return None


def _split_mul_const(x: S):
    if isinstance(x, Mul):
        c = 1.0
        rest = []
        for p in x.parts:
            if isinstance(p, Const):
                c *= p.v
            else:
                rest.append(p)
        return c, (mul(*rest) if rest else Const(1.0))
    if isinstance(x, Const):
        return x.v, Const(1.0)
    return 1.0, x


def _r_idx_distribute(s: S, notes) -> Optional[S]:
    if not isinstance(s, Idx):
        return None
    b = s.base
    if isinstance(b, Add) and is_vec(b):
        return add(*[idx(p, s.i) for p in b.parts])
    if isinstance(b, Mul) and is_vec(b):
        consts = [p for p in b.parts if isinstance(p, Const)]
        rest = [p for p in b.parts if not isinstance(p, Const)]
        if consts and len(rest) == 1:
            return mul(*consts, idx(rest[0], s.i))
    return None


def _const_scale_of_vec(v: S):
    if isinstance(v, Mul):
        cs = [p for p in v.parts if isinstance(p, Const)]
        if len(cs) == 1 and cs[0].v not in (0.0, 1.0):
            rest = [p for p in v.parts if not isinstance(p, Const)]
            if rest:
                return cs[0].v, mul(*rest)
    return None, None


def _r_norm_scale(s: S, notes) -> Optional[S]:
    if not (isinstance(s, Fn) and s.name in ("norm", "unit") and len(s.args) == 1):
        return None
    k, v = _const_scale_of_vec(s.args[0])
    if k is None:
        return None
    if s.name == "norm":
        return mul(Const(abs(k)), fn("norm", v))
    u = fn("unit", v)
    return u if k > 0 else mul(Const(-1.0), u)


def _r_pow_mulconst(s: S, notes) -> Optional[S]:
    if isinstance(s, Pow) and isinstance(s.base, Mul) and isinstance(s.exp, Const):
        cs = [p for p in s.base.parts if isinstance(p, Const)]
        if len(cs) == 1 and cs[0].v > 0 and cs[0].v != 1.0:
            rest = [p for p in s.base.parts if not isinstance(p, Const)]
            if rest:
                return mul(Const(cs[0].v ** s.exp.v), pow_(mul(*rest), s.exp))
    return None


def _r_vec_factor(s: S, notes) -> Optional[S]:
    if not (isinstance(s, Add) and is_vec(s) and len(s.parts) >= 2):
        return None
    ks: List[float] = []
    rests: List[List[S]] = []
    for p in s.parts:
        if not isinstance(p, Mul):
            return None
        cs = [q for q in p.parts if isinstance(q, Const)]
        if len(cs) != 1:
            return None
        ks.append(cs[0].v)
        rests.append([q for q in p.parts if not isinstance(q, Const)])
    k = abs(ks[0])
    if k in (0.0, 1.0) or any(abs(c) != k for c in ks):
        return None
    parts = [mul(Const(c / k), *r) for c, r in zip(ks, rests)]
    return mul(Const(k), add(*parts))


def _r_mul_distribute(s: S, notes) -> Optional[S]:
    if not isinstance(s, Mul):
        return None
    k = next((p for p in s.parts
              if isinstance(p, Const) and abs(p.v) not in (0.0, 1.0)), None)
    a = next((p for p in s.parts if isinstance(p, Add)), None)
    if k is None or a is None:
        return None
    def _cancellable(q: S) -> bool:
        if isinstance(q, Const):
            return q.v != 0.0
        if isinstance(q, Mul):
            return any(isinstance(r, Const) and abs(r.v) not in (0.0, 1.0)
                       for r in q.parts)
        return False
    if not any(_cancellable(q) for q in a.parts):
        return None
    rest = [p for p in s.parts if p is not k and p is not a]
    return mul(*rest, add(*[mul(k, q) for q in a.parts]))


def _r_cmp_shift(s: S, notes) -> Optional[S]:
    if not isinstance(s, Cmp) or s.op not in ("<", "<=", ">", ">=", "==", "!="):
        return None
    bl, cl = _split_add_const(s.left)
    br, cr = _split_add_const(s.right)
    if cl == 0.0 or cl != cr:
        return None
    return Cmp(s.op, bl, br)


def _split_add_const(x: S):
    if isinstance(x, Add):
        c = 0.0
        rest = []
        for p in x.parts:
            if isinstance(p, Const):
                c += p.v
            else:
                rest.append(p)
        return (add(*rest) if rest else Const(0.0)), c
    if isinstance(x, Const):
        return Const(0.0), x.v
    return x, 0.0


def _r_cond_zero_then(s: S, notes) -> Optional[S]:
    if isinstance(s, Cond) and isinstance(s.test, Cmp) and s.test.op in _NEG_CMP \
            and is_zero(s.then) and not is_zero(s.orelse):
        return cond(Cmp(_NEG_CMP[s.test.op], s.test.left, s.test.right),
                    s.orelse, s.then)
    return None


def _r_setmag_zero_guard(s: S, notes) -> Optional[S]:
    if not isinstance(s, Cond) or not isinstance(s.test, Cmp):
        return None
    for ren, other, ops in ((s.then, s.orelse, ("!=", ">")),
                            (s.orelse, s.then, ("==", "<", "<="))):
        u = None
        m: Optional[S] = None
        if isinstance(ren, Fn) and ren.name == "unit":
            u, m = ren, Const(1.0)
        elif isinstance(ren, Mul):
            u = next((p for p in ren.parts
                      if isinstance(p, Fn) and p.name == "unit"), None)
            if u is not None:
                rest = [p for p in ren.parts if p is not u]
                m = mul(*rest) if rest else Const(1.0)
        if u is None:
            continue
        v = u.args[0]
        op, rhs = _norm_cmp_oriented(s.test, v)
        if op not in ops or not (isinstance(rhs, Const) and rhs.v == 0.0):
            continue
        if _prop_ratio(other, v) is None:
            continue
        if isinstance(m, Mul):
            neg_c = next((p for p in m.parts
                          if isinstance(p, Const) and p.v < 0), None)
            if neg_c is not None:
                rest = [p for p in m.parts if p is not neg_c]
                m = mul(Const(-neg_c.v), *rest)
                v = mul(Const(-1.0), v)
        elif isinstance(m, Const) and m.v < 0:
            m, v = Const(-m.v), mul(Const(-1.0), v)
        notes.append("recognised zero-guarded renormalisation "
                     "(constant-magnitude command)")
        return Clamp(v, hi=m, lo=m, exact=True)
    return None


def _same_domain(a: S, b: S) -> bool:
    na = a.name if isinstance(a, Dom) else repr(a)
    nb = b.name if isinstance(b, Dom) else repr(b)
    return na == nb


def _r_mean(s: S, notes) -> Optional[S]:
    if not isinstance(s, Mul):
        return None
    parts = list(s.parts)
    red_i = next((i for i, p in enumerate(parts)
                  if isinstance(p, Reduce) and p.op == "sum"), None)
    cnt_i = next((i for i, p in enumerate(parts)
                  if isinstance(p, Pow) and isinstance(p.base, Count)
                  and isinstance(p.exp, Const) and p.exp.v == -1.0), None)
    if red_i is None or cnt_i is None:
        return None
    red, cnt = parts[red_i], parts[cnt_i].base
    if not _same_domain(red.domain, cnt.domain):
        return None
    rest = [q for k, q in enumerate(parts) if k not in (red_i, cnt_i)]
    m = Reduce("mean", red.var, red.domain, red.body, red.guard)
    return mul(*rest, m) if rest else m


def _r_reduce_gate(s: S, notes) -> Optional[S]:
    if not isinstance(s, Reduce):
        return None
    b = s.body
    if isinstance(b, Cond) and is_zero(b.orelse):
        g = b.test if s.guard is None else and_(s.guard, b.test)
        return Reduce(s.op, s.var, s.domain, b.then, g)
    if isinstance(b, Cond) and is_zero(b.then):
        g = NotS(b.test) if s.guard is None else and_(s.guard, NotS(b.test))
        return Reduce(s.op, s.var, s.domain, b.orelse, g)
    return None


def _r_reduce_split(s: S, notes) -> Optional[S]:
    if not (isinstance(s, Reduce) and s.op == "sum" and isinstance(s.body, Cond)):
        return None
    b = s.body
    if is_zero(b.then) or is_zero(b.orelse):
        return None
    yes = b.test if s.guard is None else and_(s.guard, b.test)
    no = NotS(b.test) if s.guard is None else and_(s.guard, NotS(b.test))
    return add(Reduce(s.op, s.var, s.domain, b.then, yes),
               Reduce(s.op, s.var, s.domain, b.orelse, no))


def _r_gate_lift(s: S, notes) -> Optional[S]:
    if not isinstance(s, Mul):
        return None
    for i, p in enumerate(s.parts):
        if isinstance(p, Cond):
            rest = [q for k, q in enumerate(s.parts) if k != i]
            return cond(p.test, mul(*rest, p.then), mul(*rest, p.orelse))
    return None


def _r_cond_not(s: S, notes) -> Optional[S]:
    if isinstance(s, Cond) and isinstance(s.test, NotS):
        return cond(s.test.x, s.orelse, s.then)
    return None


_NEG_CMP = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}


def _r_not_norm(s: S, notes) -> Optional[S]:
    if isinstance(s, NotS) and isinstance(s.x, Cmp) and s.x.op in _NEG_CMP:
        return Cmp(_NEG_CMP[s.x.op], s.x.left, s.x.right)
    if isinstance(s, NotS) and isinstance(s.x, NotS):
        return s.x.x
    return None


def _r_and_absorb(s: S, notes) -> Optional[S]:
    if not isinstance(s, AndS):
        return None
    for i, p in enumerate(s.parts):
        if isinstance(p, NotS) and isinstance(p.x, AndS):
            inner = [q for q in p.x.parts if q not in s.parts]
            if len(inner) < len(p.x.parts):
                rest = [q for j, q in enumerate(s.parts) if j != i]
                if not inner:
                    return and_(*rest)
                repl = NotS(inner[0]) if len(inner) == 1 \
                    else NotS(AndS(tuple(inner)))
                return and_(*rest, repl)
    return None


def _is_count_expr(x: S) -> bool:
    return isinstance(x, Count) or (isinstance(x, Fn) and x.name == "len")


def _r_emptiness(s: S, notes) -> Optional[S]:
    if not isinstance(s, Cond):
        return None
    t = s.test
    if isinstance(t, Cmp) and t.op in (">", ">=", "!="):
        if _is_count_expr(t.left) and isinstance(t.right, Const) and is_zero(s.orelse):
            notes.append("dropped emptiness guard (empty set contributes 0)")
            return s.then
    if isinstance(t, Cmp) and t.op in ("<", "<=", "=="):
        if _is_count_expr(t.left) and isinstance(t.right, Const) and is_zero(s.then):
            notes.append("dropped emptiness guard (empty set contributes 0)")
            return s.orelse
    if isinstance(t, (Dom, Prop)) and is_zero(s.orelse):
        notes.append("dropped emptiness guard (empty set contributes 0)")
        return s.then
    if isinstance(t, NotS) and isinstance(t.x, (Dom, Prop)) and is_zero(s.then):
        notes.append("dropped emptiness guard (empty set contributes 0)")
        return s.orelse
    return None


def _norm_of(x: S) -> Optional[S]:
    if isinstance(x, Fn) and x.name == "norm" and len(x.args) == 1:
        return x.args[0]
    return None


def _r_clamp(s: S, notes) -> Optional[S]:
    if not isinstance(s, Cond):
        return None
    for pass_leaf, mag_leaf in ((s.orelse, s.then), (s.then, s.orelse)):
        if isinstance(mag_leaf, Clamp) and mag_leaf.exact \
                and mag_leaf.x == pass_leaf and mag_leaf.hi is not None:
            v, m = mag_leaf.x, mag_leaf.hi
            for t in _conj(s.test):
                cc = t.x if isinstance(t, NotS) else t
                if isinstance(cc, Cmp):
                    op, other = _norm_cmp_oriented(cc, v)
                    if op is not None and other == m:
                        notes.append("recognised conditional renormalise as clamp")
                        return Clamp(v, hi=m)
    pairs = _cond_leaf_paths(s)
    leaves = [l for l, _ in pairs]
    bases = []
    for l in leaves:
        if not is_zero(l) and not _is_scaled_copy_of_any(l, leaves) and l not in bases:
            bases.append(l)
    if len(bases) != 1:
        return None
    base = bases[0]
    norm_targets = [base]
    if isinstance(base, Mul):
        norm_targets += [p for p in base.parts
                         if isinstance(p, (Reduce, Add, Vec)) or is_vec(p)]
    hi = lo = None
    matched_tgt = None
    for l, path in pairs:
        if l == base:
            continue
        if is_zero(l):
            m = Const(0.0)
        elif isinstance(l, Clamp) and l.exact and l.x == base and l.hi is not None:
            m = l.hi
        else:
            m = _scale_limit(l, base)
            if m is None:
                return None
        for c in _normalize_path(path):
            neg = isinstance(c, NotS)
            cc = c.x if neg else c
            if not isinstance(cc, Cmp):
                continue
            for tgt in norm_targets:
                op, other = _norm_cmp_oriented(cc, tgt)
                if op is None or other != m:
                    continue
                if neg:
                    op = _NEG_CMP.get(op, op)
                if op in (">", ">="):
                    hi = m
                else:
                    lo = m
                matched_tgt = matched_tgt or tgt
                break
    if hi is None and lo is None:
        return None
    if lo is not None and is_zero(lo):
        lo = None
    if hi is None and lo is None:
        return None
    notes.append("recognised conditional magnitude clamp")
    if matched_tgt is not None and matched_tgt != base and isinstance(base, Mul):
        rest = list(base.parts)
        rest.remove(matched_tgt)
        return mul(*rest, Clamp(matched_tgt, hi=hi, lo=lo))
    return Clamp(base, hi=hi, lo=lo)


def _conj(t: S):
    return list(t.parts) if isinstance(t, AndS) else [t]


def _normalize_path(path):
    pos = [c for c in path if not isinstance(c, NotS)]
    out = list(pos)
    for c in path:
        if not isinstance(c, NotS):
            continue
        if isinstance(c.x, AndS):
            rem = [q for q in c.x.parts if q not in pos]
            if len(rem) == 1:
                q = rem[0]
                out.append(Cmp(_NEG_CMP[q.op], q.left, q.right)
                           if isinstance(q, Cmp) and q.op in _NEG_CMP else NotS(q))
                continue
        out.append(c)
    return out


def _cond_leaf_paths(s: S, path=None, out=None):
    path = [] if path is None else path
    out = [] if out is None else out
    if isinstance(s, Cond):
        _cond_leaf_paths(s.then, path + _conj(s.test), out)
        _cond_leaf_paths(s.orelse, path + [NotS(s.test)], out)
    else:
        out.append((s, path))
    return out


def _is_scaled_copy_of_any(l: S, leaves) -> bool:
    if isinstance(l, Clamp) and l.exact and any(o == l.x for o in leaves):
        return True
    return any(o != l and _scale_limit(l, o) is not None for o in leaves)


def _scale_limit(scaled: S, base: S) -> Optional[S]:
    if isinstance(scaled, Fn) and scaled.name == "unit" and scaled.args[0] == base:
        return Const(1.0)
    if not isinstance(scaled, Mul):
        return None
    parts = list(scaled.parts)
    u = next((p for p in parts if isinstance(p, Fn) and p.name == "unit"
              and p.args[0] == base), None)
    if u is not None:
        parts.remove(u)
        return mul(*parts) if parts else None
    if base in parts:
        parts.remove(base)
        vec = base
    elif isinstance(base, Mul):
        missing = []
        for p in base.parts:
            if p in parts:
                parts.remove(p)
            else:
                missing.append(p)
        if len(missing) == 1:
            v = missing[0]
            u = next((q for q in parts if isinstance(q, Fn) and q.name == "unit"
                      and q.args[0] == v), None)
            if u is None:
                return None
            parts.remove(u)
            return mul(*parts) if parts else None
        if missing:
            return None
        vec = base
    else:
        return None
    inv = next((p for p in parts
                if isinstance(p, Pow) and isinstance(p.exp, Const) and p.exp.v == -1.0
                and _norm_of(p.base) == vec), None)
    if inv is None:
        return None
    parts.remove(inv)
    if len(parts) == 1:
        return parts[0]
    if len(parts) > 1:
        return mul(*parts)
    return None


def _cmp_says_greater(t: S, a: S, b: S) -> bool:
    if not isinstance(t, Cmp):
        return False
    if t.op in (">", ">=") and t.left == a and t.right == b:
        return True
    if t.op in ("<", "<=") and t.left == b and t.right == a:
        return True
    return False


def _norm_cmp_oriented(c: Cmp, base: S):
    if _norm_of(c.left) == base:
        return c.op, c.right
    if _norm_of(c.right) == base:
        flip = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
        return flip.get(c.op, c.op), c.left
    return None, None


def _r_idx_of_cond(s: S, notes) -> Optional[S]:
    if isinstance(s, Idx) and isinstance(s.base, Cond):
        return cond(s.base.test, idx(s.base.then, s.i), idx(s.base.orelse, s.i))
    if isinstance(s, Idx) and isinstance(s.base, Clamp):
        return None
    return None


def _r_reduce_of_vec(s: S, notes) -> Optional[S]:
    if isinstance(s, Reduce) and isinstance(s.body, Vec):
        folded = axis_fold_vec(s.body)
        if folded is not None:
            return Reduce(s.op, s.var, s.domain, folded, s.guard)
    return None


_RULES = [
    _r_like_terms,
    _r_cond_not,
    _r_not_norm,
    _r_and_absorb,
    _r_idx_distribute,
    _r_axis_fold,
    _r_sqrt_norm,
    _r_sqrt_dot,
    _r_norm_of_vec,
    _r_norm_scale,
    _r_pow_mulconst,
    _r_vec_factor,
    _r_mul_distribute,
    _r_unit,
    _r_mean,
    _r_reduce_gate,
    _r_reduce_split,
    _r_gate_lift,
    _r_emptiness,
    _r_cond_zero_then,
    _r_cmp_shift,
    _r_setmag_zero_guard,
    _r_clamp,
    _r_idx_of_cond,
    _r_reduce_of_vec,
]
