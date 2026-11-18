from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from . import arrays, fields, ir
from .program import Program, norm_attr as _norm_attr, norm_param as _norm_param
from .sexpr import (S, Add, AndS, Clamp, Cmp, Cond, Const, Count, Dom, Fn,
                    Idx, Mul, NotS, Obj, OrS, Pow, Prop, Reduce, Sym, Unk,
                    Vec, ZERO, add, and_, cmp, cond, div, fn, idx, is_vec,
                    is_zero, mul, neg, pow_, source_of, sub, subst, walk)

MATH_CONSTS = {"pi": math.pi, "e": math.e, "inf": math.inf}
MATH_FNS = {"sqrt": "sqrt", "exp": "exp", "cos": "cos", "sin": "sin",
            "tan": "tan", "acos": "acos", "asin": "asin", "atan2": "atan2",
            "degrees": "degrees", "radians": "radians", "fabs": "abs",
            "abs": "abs", "floor": "floor", "ceil": "ceil", "log": "log",
            "log2": "log", "hypot": "hypot"}

_AXIS = Sym("_axis", "axis")


@dataclass
class Env:
    vars: Dict[str, S] = field(default_factory=dict)
    heap: Dict[S, S] = field(default_factory=dict)

    def copy(self) -> "Env":
        return Env(dict(self.vars), dict(self.heap))


@dataclass
class Path:
    guard: S
    env: Env
    kind: str = "fall"
    value: Optional[S] = None


@dataclass
class Result:
    ret: Optional[S]
    env: Env
    params: Dict[str, float]
    actuations: List[Tuple[str, Tuple[S, ...]]]
    notes: List[str]
    trace: "Trace" = None
    entry_args: List[Dict[str, S]] = field(default_factory=list)


@dataclass
class Trace:
    samples: list = field(default_factory=list)
    potentials: list = field(default_factory=list)
    occupancies: list = field(default_factory=list)
    gradients: list = field(default_factory=list)
    gradient_ids: set = field(default_factory=set)


class Exec:
    def __init__(self, program: Program, use_shortcuts: bool = True,
                 max_depth: int = 12, dims: int = 2, while_iters: int = 2):
        self.prog = program
        self.use_shortcuts = use_shortcuts
        self.max_depth = max_depth
        self.dims = dims
        self.vectors = program.vector_fields()
        self.trace = Trace()
        self.entry: Optional[ir.FuncDef] = None
        self.entry_args: List[Dict[str, S]] = []
        self.counts: Dict[S, str] = {}
        self.classes: Dict[str, str] = program.element_classes()
        self.while_iters = max(1, while_iters)
        self.params: Dict[str, float] = dict(program.params)
        self.actuations: List[Tuple[str, Tuple[S, ...]]] = []
        self.notes: List[str] = []
        self.genv: Dict[str, S] = {}
        self.alias: Dict[S, Tuple[S, str]] = {}
        self._suffix_n = 0
        self._ctor_n = 0

    def run(self, qualname: str, tick=None, init: bool = True) -> Result:
        f, mod = self.prog.functions[qualname]
        self.entry = f
        env = Env()
        if tick is not None:
            args = {p: Sym(p, "free") for p in list(tick.params) + list(tick.free)}
            genv = Env(args, env.heap)
            for stmts in (tick.setup, tick.body):
                paths = self._exec_block(stmts, Path(Const(1.0), genv), tick.module, 0)
                genv = _merge_paths_env(paths, genv)
            env = genv
            ret = None
        else:
            selfv = Obj("self")
            args = {}
            if f.cls is not None:
                if init:
                    ini = self.prog.functions.get("%s.%s.__init__" % (mod.name, f.cls))
                    if ini:
                        self._harvest_init(ini[0], ini[1], selfv, env)
                args[f.params[0]] = selfv
                rest = f.params[1:]
            else:
                rest = f.params
            for p in rest:
                args[p] = Sym(p, "free")
            ret, env = self._call_function(f, mod, args, env, depth=0)
        return Result(ret, env, self.params, self.actuations, self.notes, self.trace,
                      self.entry_args)

    def _harvest_init(self, f: ir.FuncDef, mod: ir.Module, selfv: S, env: Env):
        for s in ir.walk_stmts(f.body):
            if isinstance(s, ir.Assign) and isinstance(s.target, ir.Attr) \
                    and isinstance(s.target.base, ir.NameE) \
                    and s.target.base.id == f.params[0]:
                name = _norm_attr(s.target.name)
                v = ir._const_value(s.value)
                if v is not None:
                    self.params[name] = v
                    env.heap[self._prop(selfv, name)] = Sym(name, "param")

    def _prop(self, base: S, name: str) -> Prop:
        name = _norm_attr(name)
        return Prop(base, name, "vector" if name in self.vectors else None)

    def _domain(self, value: S) -> Dom:
        if isinstance(value, Dom):
            return value
        return Dom(_dom_name(value), base=value, source=source_of(value))

    def _call_function(self, f: ir.FuncDef, mod: ir.Module,
                       args: Dict[str, S], env: Env, depth: int
                       ) -> Tuple[Optional[S], Env]:
        if depth > self.max_depth:
            return Unk("depth:" + f.name), env
        if f is self.entry:
            self.entry_args.append(dict(args))
        genv = Env(dict(args), env.heap)
        for p, d in f.defaults.items():
            if p not in genv.vars:
                v = self._eval(d, genv, mod, depth)
                if isinstance(v, Const) and not float(v.v).is_integer():
                    name = _norm_param(p)
                    self.params.setdefault(name, v.v)
                    v = Sym(name, "param")
                genv.vars[p] = v
        gen = _generator_shape(f)
        if gen is not None:
            var, domx, prefix, conds = gen
            domv = self._domain(self._eval(domx, genv, mod, depth))
            elem = Obj(var, domain=_dom_name(domv), source=source_of(domv))
            genv.vars[var] = elem
            for st in prefix:
                for _ in self._exec_stmt(st, Path(Const(1.0), genv), mod, depth):
                    pass
            guard = and_(*[self._eval(c, genv, mod, depth) for c in conds]) \
                if conds else None
            return Dom(_dom_name(domv), base=domv, var=var, guard=guard,
                       source=source_of(domv)), env

        paths = self._exec_block(f.body, Path(Const(1.0), genv), mod, depth)
        ret = None
        heap = None
        for p in paths:
            v = p.value if p.kind == "return" and p.value is not None else ZERO
            ret = v if ret is None else cond(p.guard, v, ret)
            heap = p.env.heap if heap is None else _merge_heap(p.guard, p.env.heap, heap)
        out_env = env.copy()
        if heap is not None:
            out_env.heap = heap
        return ret, out_env

    def _exec_block(self, stmts: List[ir.Stmt], p: Path, mod: ir.Module,
                    depth: int) -> List[Path]:
        paths = [p]
        for i, s in enumerate(stmts):
            new_paths: List[Path] = []
            for q in paths:
                if q.kind != "fall":
                    new_paths.append(q)
                    continue
                new_paths.extend(self._exec_stmt(s, q, mod, depth))
            paths = new_paths
            if len(paths) > 64:
                self.notes.append("path explosion at %s; merging coarsely" % s.loc)
                paths = [_coarse_merge(paths)]
        return paths

    def _exec_stmt(self, s: ir.Stmt, p: Path, mod: ir.Module,
                   depth: int) -> List[Path]:
        env = p.env
        if isinstance(s, ir.Assign):
            v = self._eval(s.value, env, mod, depth)
            self._store(s.target, v, env, mod, depth, decl=s.decl)
            return [p]
        if isinstance(s, ir.AugAssign):
            cur = self._eval(s.target, env, mod, depth)
            v = self._eval(s.value, env, mod, depth)
            nv = {"+": add, "-": sub, "*": mul, "/": div}.get(s.op, lambda a, b: Unk("aug" + s.op, (a, b)))(cur, v)
            self._store(s.target, nv, env, mod, depth)
            return [p]
        if isinstance(s, ir.ExprStmt):
            self._eval(s.value, env, mod, depth)
            return [p]
        if isinstance(s, ir.Return):
            v = self._eval(s.value, env, mod, depth) if s.value is not None else None
            return [Path(p.guard, env, "return", v)]
        if isinstance(s, ir.If):
            test = self._decide(self._eval(s.cond, env, mod, depth))
            if isinstance(test, Const):
                branch = s.body if test.v else s.orelse
                return self._exec_block(branch, p, mod, depth)
            tp = self._exec_block(s.body, Path(and_(p.guard, test), env.copy(), "fall"), mod, depth)
            ep = self._exec_block(s.orelse, Path(and_(p.guard, NotS(test)), env.copy(), "fall"), mod, depth)
            falls = [q for q in tp + ep if q.kind == "fall"]
            others = [q for q in tp + ep if q.kind != "fall"]
            if falls:
                merged = falls[0]
                for q in falls[1:]:
                    merged = Path(OrS((merged.guard, q.guard)) if merged.guard != q.guard else merged.guard,
                                  _merge_env(_relative(q.guard, p.guard), q.env, merged.env), "fall")
                merged = Path(p.guard, merged.env, "fall")
                return [merged] + others
            return others
        if isinstance(s, ir.For):
            return [self._exec_for(s, p, mod, depth)]
        if isinstance(s, ir.While):
            wenv = env
            for _ in range(self.while_iters):
                body_paths = self._exec_block(
                    s.body, Path(p.guard, wenv.copy(), "fall"), mod, depth)
                wenv = _merge_paths_env(body_paths, wenv)
            self._mirror_clones(wenv)
            return [Path(p.guard, wenv, "fall")]
        if isinstance(s, (ir.Break, ir.Continue)):
            return [Path(p.guard, env, "break" if isinstance(s, ir.Break) else "continue")]
        return [p]

    def _decide(self, test: S) -> S:
        if isinstance(test, Const):
            return test
        values = {}
        for node in walk(test):
            if isinstance(node, (Prop, Obj, Dom, Unk, Count, Reduce, Fn, Vec, Idx)):
                return test
            if isinstance(node, Sym):
                if node.kind != "param" or node.name not in self.params:
                    return test
                values[node.name] = self.params[node.name]

        def ev(x):
            if isinstance(x, Const):
                return x.v
            if isinstance(x, Sym):
                return values[x.name]
            if isinstance(x, Add):
                return sum(ev(p) for p in x.parts)
            if isinstance(x, Mul):
                out = 1.0
                for p in x.parts:
                    out *= ev(p)
                return out
            if isinstance(x, Pow):
                return ev(x.base) ** ev(x.exp)
            if isinstance(x, Cmp):
                a, b = ev(x.left), ev(x.right)
                return float({"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b,
                              "==": a == b, "!=": a != b}[x.op])
            if isinstance(x, AndS):
                return float(all(ev(p) for p in x.parts))
            if isinstance(x, OrS):
                return float(any(ev(p) for p in x.parts))
            if isinstance(x, NotS):
                return float(not ev(x.x))
            raise ValueError(x)
        try:
            return Const(ev(test))
        except (ValueError, KeyError, ZeroDivisionError, OverflowError):
            return test

    def _adopt(self, s: ir.For, it: S, env: Env, mod: ir.Module, depth: int) -> Optional[Dom]:
        if not (isinstance(it, Fn) and it.name in ("range", "range_incl")
                and isinstance(s.target, ir.NameE)):
            return None
        if any(isinstance(a, Count) and isinstance(a.domain, Dom) for a in walk(it)):
            return None
        names = set()
        for base in arrays.subscripted(s.body, s.target.id):
            try:
                value = _deref(self._eval(base, env, mod, depth), env)
            except Exception:
                continue
            name = arrays.collection_of(value)
            if name is not None:
                names.add(name)
        if len(names) != 1:
            return None
        name = names.pop()
        stop = it.args[-1] if len(it.args) <= 2 else it.args[1]
        if isinstance(stop, Add):
            rest = [q for q in stop.parts if not isinstance(q, Const)]
            stop = rest[0] if len(rest) == 1 else stop
        self.counts[stop] = name
        return Dom(name)

    def _exec_for(self, s: ir.For, p: Path, mod: ir.Module, depth: int) -> Path:
        env = p.env
        it = self._eval(s.iter, env, mod, depth)
        early = self._adopt(s, it, env, mod, depth)

        if isinstance(it, Vec):
            for item in it.items:
                self._bind_target(s.target, item, env)
                paths = self._exec_block(s.body, Path(p.guard, env, "fall"), mod, depth)
                env = _merge_paths_env(paths, env)
            return Path(p.guard, env, "fall")

        adopted = early
        if adopted is None and isinstance(it, Fn) and it.name in ("range", "range_incl"):
            if any(isinstance(a, Fn) and a.name == "len" for a in walk(it)):
                self._bind_target(s.target, _AXIS, env)
                paths = self._exec_block(s.body, Path(p.guard, env, "fall"), mod, depth)
                env = _merge_paths_env(paths, env)
                _fold_axis_env(env, self.notes)
                return Path(p.guard, env, "fall")
            cargs = [a for a in it.args if isinstance(a, Const)]
            step = it.args[2].v if len(it.args) > 2 and isinstance(it.args[2], Const) else 1.0
            if len(cargs) == len(it.args) and it.args and step == 1.0 \
                    and all(float(a.v).is_integer() for a in cargs[:2]):
                lo, hi = (0, int(cargs[0].v)) if len(cargs) == 1 \
                    else (int(cargs[0].v), int(cargs[1].v))
                if it.name == "range_incl":
                    hi += 1
                if hi - lo <= 8:
                    for k in range(lo, hi):
                        self._bind_target(s.target, Const(float(k)), env)
                        paths = self._exec_block(s.body, Path(p.guard, env, "fall"), mod, depth)
                        env = _merge_paths_env(paths, env)
                    return Path(p.guard, env, "fall")
            cnt = next((a for a in walk(it) if isinstance(a, Count)), None)
            if cnt is not None and isinstance(cnt.domain, Dom):
                adopted = cnt.domain
            elif step != 1.0 and len(it.args) >= 2:
                var0 = s.target.id if isinstance(s.target, ir.NameE) else "_e"
                elem0 = Obj(var0, domain=var0 + "_samples")
                lo_g = cmp("<=", it.args[0], elem0)
                hi_g = cmp("<=" if it.name == "range_incl" else "<",
                           elem0, it.args[1])
                adopted = Dom(var0 + "_samples", base=it, var=var0,
                              guard=and_(lo_g, hi_g))
                self._note_once(
                    "numeric range sampled symbolically: %s in [%r, %r] "
                    "step %r" % (var0, it.args[0], it.args[1], it.args[2]))

        index_var = None
        if isinstance(it, Fn) and it.name == "enumerate" and it.args:
            if isinstance(s.target, ir.TupleE) and len(s.target.items) == 2:
                iv, ev = s.target.items
                index_var = iv.id if isinstance(iv, ir.NameE) else None
                s = ir.For(ev, s.iter, s.body, loc=s.loc)
            it = it.args[0]

        if adopted is not None:
            domv = adopted
        else:
            domv = self._domain(it)
        var = s.target.id if isinstance(s.target, ir.NameE) else "_e"
        before = env.copy()
        if adopted is not None:
            elem = Obj(var, domain=domv.name, source=source_of(domv))
            env.vars[var] = elem
        elif domv.elt is not None and isinstance(domv.elt, Obj):
            elem = self._clone_obj(domv.elt)
            env.vars[var] = elem
        elif domv.elt is not None:
            elem = Obj(var, domain=domv.name, source=source_of(domv))
            env.vars[var] = subst(domv.elt,
                                  {Obj(domv.var or var, domain=domv.name): elem})
        else:
            elem = Obj(var, domain=domv.name, source=source_of(domv))
            env.vars[var] = elem
        bound_elem = env.vars[var]
        if index_var:
            env.vars[index_var] = Sym("_" + index_var, "free")
        paths = self._exec_block(s.body, Path(p.guard, env, "fall"), mod, depth)
        env = _merge_paths_env(paths, env)

        out = before.copy()
        changed = [(("var", k), before.vars.get(k), env.vars[k])
                   for k in env.vars if env.vars.get(k) != before.vars.get(k) and k != var] + \
                  [(("heap", k), before.heap.get(k), env.heap[k])
                   for k in env.heap if env.heap.get(k) != before.heap.get(k)]
        for key, old, new in changed:
            base = old if old is not None else (key[1] if key[0] == "heap" else Sym(str(key[1]), "free"))
            if isinstance(new, Fn) and new.name == "_rstore":
                k, off = fields.affine_of(new.args[1])
                val = fields.occupancy(domv, k, off)
                self._note_once("rasterisation loop over '%s' recognised as an "
                                "occupancy grid (scale=%r, offset=%r)"
                                % (domv.name, k, off))
                if key[0] == "var":
                    out.vars[key[1]] = val
                else:
                    out.heap[key[1]] = val
                continue
            if arrays.is_map(new) and new.name == domv.name and new.var == var \
                    and not arrays.is_map(old):
                val = new
                if key[0] == "var":
                    out.vars[key[1]] = val
                else:
                    out.heap[key[1]] = val
                continue
            if arrays.is_map(new) and isinstance(old, Fn) and old.name == "_zeros":
                old = Dom(new.name, var=new.var, source=new.source, elt=ZERO)
            if arrays.is_map(new) and arrays.is_map(old) and new.var == old.var \
                    and new.name == old.name and new.elt != old.elt:
                d = _delta_of(new.elt, old.elt)
                if d is not None and not is_zero(d):
                    red = Reduce("sum", var, domv, d, guard=domv.guard)
                    val = replace(old, elt=add(old.elt, red))
                    if key[0] == "var":
                        out.vars[key[1]] = val
                    else:
                        out.heap[key[1]] = val
                    continue
            appended = _peel_append(new, base)
            if appended is not None:
                elt, guard = appended
                dname = key[1] if key[0] == "var" else key[1].name
                val = Dom(str(dname), base=domv, var=var,
                          guard=and_(*[g for g in (domv.guard, guard)
                                       if g is not None]) if (domv.guard or guard)
                          else None, elt=elt, source=source_of(domv))
                self._note_once("loop over '%s' builds '%s' element-wise "
                                "(filtered map)" % (domv.name, dname))
            elif isinstance(new, Dom) and _is_empty_container(base):
                val = new
            elif isinstance(new, Dom) and isinstance(old, Dom) \
                    and new.name == old.name:
                val = new
            elif (isinstance(new, Cond) and bound_elem in (new.then, new.orelse)) \
                    or new == bound_elem:
                val = bound_elem
                self._note_once("selection loop over '%s': kept the matching "
                                "element as the representative" % domv.name)
            elif key[0] == "heap" and old is None:
                val = new
            else:
                delta = _delta_of(new, base)
                if delta is not None and not is_zero(delta):
                    red = Reduce("sum", var, domv, delta, guard=domv.guard)
                    val = red if is_zero(base) else add(base, red)
                elif old == new:
                    continue
                else:
                    self.notes.append("loop over %s: %s overwritten (no accumulator)"
                                      % (domv.name, key[1]))
                    val = Unk("loop", (new,))
            if key[0] == "var":
                out.vars[key[1]] = val
            else:
                out.heap[key[1]] = val
        return Path(p.guard, out, "fall")

    def _store(self, target: ir.Expr, v: S, env: Env, mod: ir.Module,
               depth: int, decl: bool = False):
        if isinstance(target, ir.NameE):
            if not decl and target.id not in env.vars \
                    and (target.id in self.prog.globals
                         or target.id in self.genv):
                self.genv[target.id] = v
                return
            env.vars[target.id] = v
        elif isinstance(target, ir.Attr):
            base = self._eval(target.base, env, mod, depth)
            env.heap[self._prop(base, target.name)] = v
        elif isinstance(target, ir.Index):
            chain: List[S] = []
            t = target
            while isinstance(t, ir.Index):
                chain.insert(0, self._eval(t.index, env, mod, depth))
                t = t.base
            rootv = self._eval(t, env, mod, depth)
            stored = arrays.store(rootv, chain, v, self.dims)
            if stored is not None:
                self._store(t, stored, env, mod, depth)
                return
            if any(isinstance(i, Fn) and i.name == "_slice" for i in chain) \
                    and isinstance(rootv, Fn) and rootv.name in ("_grid0", "_rstore"):
                bounds: List[S] = []
                for i in chain:
                    bounds.extend(i.args if isinstance(i, Fn) and i.name == "_slice"
                                  else (i, i))
                self._store(t, Fn("_rstore", (rootv,) + tuple(bounds) + (v,)),
                            env, mod, depth)
                return
            if len(chain) == 1 and isinstance(chain[0], (Cmp, AndS, OrS, NotS)):
                self._store(t, cond(chain[0], v, rootv), env, mod, depth)
                return
            i = self._eval(target.index, env, mod, depth)
            cur = self._eval(target.base, env, mod, depth)
            self._store(target.base, _component_store(cur, i, v), env, mod, depth)
        elif isinstance(target, (ir.TupleE, ir.ListE)):
            items = v.items if isinstance(v, Vec) else \
                [idx(v, Const(float(k))) for k in range(len(target.items))]
            for t, item in zip(target.items, items):
                self._store(t, item, env, mod, depth)

    def _bind_target(self, target: ir.Expr, v: S, env: Env):
        if isinstance(target, ir.NameE):
            env.vars[target.id] = v
        elif isinstance(target, (ir.TupleE, ir.ListE)):
            items = v.items if isinstance(v, Vec) else \
                [idx(v, Const(float(k))) for k in range(len(target.items))]
            for t, item in zip(target.items, items):
                self._bind_target(t, item, env)

    def _eval(self, e: ir.Expr, env: Env, mod: ir.Module, depth: int) -> S:
        if isinstance(e, ir.Num):
            return Const(e.value)
        if isinstance(e, ir.Str):
            return Unk("str")
        if isinstance(e, ir.NameE):
            return self._eval_name(e.id, env, mod)
        if isinstance(e, ir.Attr):
            return self._eval_attr(e, env, mod, depth)
        if isinstance(e, ir.Index):
            base = _deref(self._eval(e.base, env, mod, depth), env)
            iv = self._eval(e.index, env, mod, depth)
            fsym = fields.next_field_sym(base) if not isinstance(
                base, (Vec, Prop, Obj, Dom)) else None
            if fsym is not None:
                self._note_once("grid field sampled at the indexing point "
                                "(window means -> point evaluation)")
                sample = subst(base, {fsym: fields.slice_center(iv)})
                if any(id(x) in self.trace.gradient_ids for x in walk(base)):
                    self.trace.samples.append((iv, sample))
                    self.trace.gradient_ids.add(id(sample))
                return sample
            got = arrays.index(base, iv, self.dims)
            if got is not None:
                return got
            cv = _container_value(base)
            if cv is not None:
                base = cv
            if isinstance(base, Dom):
                if isinstance(iv, Fn) and iv.name in ("argmax", "argmin"):
                    return Fn("select_" + iv.name, (base, iv.args[0]))
                if base.elt is not None:
                    return base.elt
                return Obj(base.var or "_e", domain=base.name, source=source_of(base))
            return idx(base, iv)
        if isinstance(e, ir.SliceE):
            lo = self._eval(e.lo, env, mod, depth) if e.lo is not None else Unk("nobound")
            hi = self._eval(e.hi, env, mod, depth) if e.hi is not None else Unk("nobound")
            return Fn("_slice", (lo, hi))
        if isinstance(e, ir.Call):
            return self._eval_call(e, env, mod, depth)
        if isinstance(e, ir.BinOp):
            l = self._eval(e.left, env, mod, depth)
            r = self._eval(e.right, env, mod, depth)
            return self._binop(e.op, l, r)
        if isinstance(e, ir.UnOp):
            v = self._eval(e.operand, env, mod, depth)
            if e.op == "-":
                return arrays.unary(neg, v) or neg(v)
            if e.op == "not":
                return arrays.unary(NotS, v) or NotS(v)
            if e.op == "*":
                return Fn("_splat", (v,))
            return v
        if isinstance(e, ir.CmpOp):
            op = _norm_cmp(e.op)
            left, right = self._eval(e.left, env, mod, depth), self._eval(e.right, env, mod, depth)
            return arrays.binop(op, left, right, lambda o, a, b: cmp(o, a, b)) or cmp(op, left, right)
        if isinstance(e, ir.BoolOp):
            parts = [self._eval(x, env, mod, depth) for x in e.parts]
            return and_(*parts) if e.op == "and" else OrS(tuple(parts))
        if isinstance(e, (ir.TupleE, ir.ListE)):
            return Vec(tuple(self._eval(x, env, mod, depth) for x in e.items))
        if isinstance(e, ir.IfExp):
            return cond(self._eval(e.cond, env, mod, depth),
                        self._eval(e.then, env, mod, depth),
                        self._eval(e.orelse, env, mod, depth))
        if isinstance(e, ir.Comp):
            return self._eval_comp(e, env, mod, depth)
        if isinstance(e, ir.Opaque):
            return Unk(e.tag)
        return Unk("expr")

    def _eval_name(self, name: str, env: Env, mod: ir.Module) -> S:
        if name in env.vars:
            return env.vars[name]
        if name == "None":
            return Unk("none")
        if name in self.genv:
            return self.genv[name]
        consts = self.prog.consts.get(mod.name, {})
        if name in consts:
            return Sym(_norm_param(name), "param")
        if name in mod.imports:
            return Unk("module:" + mod.imports[name])
        if name in mod.from_imports:
            m, remote = mod.from_imports[name]
            if m in ("math", "numpy", "numpy.linalg") and remote in MATH_CONSTS:
                return Const(MATH_CONSTS[remote])
            return Unk("import:%s.%s" % (m, remote))
        if name in self.prog.globals:
            v = self._eval_global(name, env, mod)
            self.genv[name] = v
            return v
        return Sym(name, "free")

    def _eval_global(self, name: str, env: Env, mod: ir.Module) -> S:
        init = self.prog.globals.get(name)
        if isinstance(init, ir.Opaque):
            return Sym(name, "free")
        if isinstance(init, ir.Call):
            args = ", ".join(
                "%g" % a.value for a in init.args if isinstance(a, ir.Num))
            self._note_once("module global '%s' kept as a named world anchor "
                            "(initialised %s(%s))"
                            % (name, getattr(init.fn, "id", "?"), args))
            return Prop(Obj(name, domain="anchor"), "position",
                        "vector" if len(init.args) >= 2 else None)
        if init is not None:
            return self._eval(init, env, mod, 0)
        return Sym(name, "free")

    def _eval_attr(self, e: ir.Attr, env: Env, mod: ir.Module, depth: int) -> S:
        base = self._eval(e.base, env, mod, depth)
        if isinstance(base, Unk) and base.tag.startswith("module:"):
            m = base.tag[7:]
            if m in ("math", "numpy") and e.name in MATH_CONSTS:
                return Const(MATH_CONSTS[e.name])
            return Unk("module:%s.%s" % (m, e.name))
        prop = self._prop(base, e.name)
        if prop in env.heap:
            return env.heap[prop]
        name = _norm_attr(e.name)
        fwd = self._forward_prop(prop, env)
        if fwd is not None:
            return fwd
        if name in ("x", "y", "z") and (isinstance(base, Vec) or is_vec(base)):
            return idx(base, Const(float("xyz".index(name))))
        if not (isinstance(base, Obj) and base.domain) \
                and name in self.prog.class_consts \
                and name not in self.vectors and name not in self.prog.state_attrs:
            self.params.setdefault(name, self.prog.class_consts[name])
            return Sym(name, "param")
        return prop

    def _forward_prop(self, prop: Prop, env: Env) -> Optional[S]:
        info = self.alias.get(prop.base)
        if info is None:
            return None
        orig, suffix = info
        src = Prop(orig, prop.name, prop.role)
        v = env.heap.get(src)
        if v is None:
            v = self._forward_prop(src, env)
        if v is None:
            return None
        return self._rename(v, suffix)

    def _rename(self, v: S, suffix: str) -> S:
        tag = "@" + suffix
        mapping: Dict[S, S] = {}
        for w in walk(v):
            if isinstance(w, Obj) and w.name != "self" \
                    and w.domain != "anchor" and not w.name.endswith(tag):
                nw = Obj(w.name + tag, w.domain, w.source)
                mapping[w] = nw
                self.alias.setdefault(nw, (w, suffix))
        return subst(v, mapping) if mapping else v

    def _clone_obj(self, orig: S) -> S:
        self._suffix_n += 1
        suffix = "e%d" % self._suffix_n
        clone = Obj("%s@%s" % (orig.name, suffix), orig.domain, orig.source)
        self.alias[clone] = (orig, suffix)
        return clone

    def _mirror_clones(self, env: Env):
        for key in list(env.heap):
            if isinstance(key, Prop) and key.base in self.alias:
                orig, _ = self.alias[key.base]
                env.heap[Prop(orig, key.name, key.role)] = env.heap[key]

    def _eval_comp(self, e: ir.Comp, env: Env, mod: ir.Module, depth: int) -> S:
        it = self._eval(e.iter, env, mod, depth)
        if isinstance(it, Fn) and it.name == "zip" and len(it.args) == 2 \
                and isinstance(e.var, ir.TupleE) and isinstance(e.elt, ir.BinOp) \
                and e.elt.op == "*":
            return Fn("_dotgen", it.args)
        if isinstance(it, Vec) or is_vec(it):
            env2 = Env(dict(env.vars), env.heap)
            self._bind_target(e.var, it, env2)
            if e.conds:
                return Unk("filtered_map")
            return self._eval(e.elt, env2, mod, depth)
        domv = self._domain(it)
        var = e.var.id if isinstance(e.var, ir.NameE) else "_e"
        env2 = Env(dict(env.vars), env.heap)
        elem = Obj(var, domain=domv.name, source=source_of(domv))
        env2.vars[var] = elem
        guard = and_(*[self._eval(c, env2, mod, depth) for c in e.conds]) if e.conds else None
        if domv.guard is not None:
            guard = and_(domv.guard, guard) if guard is not None else domv.guard
        eltv = self._eval(e.elt, env2, mod, depth)
        return Dom(domv.name, base=domv.base or it, var=var, guard=guard,
                   elt=None if eltv == elem else eltv, source=source_of(domv))

    def _eval_call(self, e: ir.Call, env: Env, mod: ir.Module, depth: int) -> S:
        args: List[S] = []
        for a in e.args:
            v = self._eval(a, env, mod, depth)
            if isinstance(v, Fn) and v.name == "_splat":
                inner = v.args[0]
                if isinstance(inner, Vec):
                    args.extend(inner.items)
                elif is_vec(inner):
                    args.extend(idx(inner, Const(float(k)))
                                for k in range(self.dims))
                else:
                    args.append(inner)
            else:
                args.append(v)

        fnexpr = e.fn
        if isinstance(fnexpr, ir.NameE) and fnexpr.id == "hasattr" and len(e.args) == 2 \
                and isinstance(e.args[1], ir.Str):
            return Const(1.0 if _norm_attr(e.args[1].value) in self.prog.assigned_attrs else 0.0)
        if isinstance(fnexpr, ir.Attr):
            basev = self._eval(fnexpr.base, env, mod, depth)
            name = fnexpr.name
            if isinstance(basev, Unk) and basev.tag.startswith("module:"):
                m = basev.tag[7:]
                if m.startswith("numpy"):
                    kwargs = {k: self._eval(v, env, mod, depth) for k, v in e.kwargs.items()}
                    got = arrays.numpy_call(name, args, kwargs, self.counts)
                    if got is not None:
                        return got
                if m in self.prog.modules:
                    got = self.prog.functions.get("%s.%s" % (m, name))
                    if got:
                        return self._inline(got[0], got[1], args, env, depth, name)
                return self._external(m, name, args)
            return self._method(basev, name, args, env, mod, depth, e)
        if isinstance(fnexpr, ir.NameE):
            name = fnexpr.id
            got = self.prog.module_fn(mod, name)
            if got:
                if self.use_shortcuts:
                    sc = self._shortcut(name, args)
                    if sc is not None:
                        return sc
                return self._inline(got[0], got[1], args, env, depth, name)
            made = self._construct(name, args, env, mod, depth)
            if made is not None:
                return made
            b = self._builtin(name, args)
            if b is not None:
                return b
            if name in mod.from_imports:
                m, remote = mod.from_imports[name]
                return self._external(m, remote, args)
            return Unk("call:" + name, tuple(args))
        return Unk("call", tuple(args))

    def _construct(self, name: str, args: List[S], env: Env, mod: ir.Module,
                   depth: int) -> Optional[S]:
        got = self.prog.find_ctor(mod, name)
        if got is None:
            if name in ("Vector2d", "Vector3d", "Vector4d", "Vector2f",
                        "Vector3f", "Point") and args:
                return Vec(tuple(args))
            return None
        f, fmod = got
        self._ctor_n += 1
        obj = Obj("%s%d" % (name.lower(), self._ctor_n))
        call_args = {f.params[0]: obj} if f.params else {}
        for pname, v in zip(f.params[1:], args):
            call_args[pname] = v
        _, env2 = self._call_function(f, fmod, call_args, env, depth + 1)
        env.heap = env2.heap
        return obj

    def _method(self, obj: S, name: str, args: List[S], env: Env,
                mod: ir.Module, depth: int, e: ir.Call) -> S:
        cands = self.prog.methods.get(name)
        if cands and len(cands) > 1 and isinstance(obj, Obj) and obj.domain:
            cls = self.classes.get(obj.domain)
            cands = [c for c in cands if c[0].cls == cls] or cands
        if cands:
            f, fmod = cands[0]
            if self.use_shortcuts:
                sc = self._shortcut(name, args)
                if sc is not None:
                    return sc
            call_args = {f.params[0]: obj}
            for p, v in zip(f.params[1:], args):
                call_args[p] = v
            ret, env2 = self._call_function(f, fmod, call_args, env, depth + 1)
            env.heap = env2.heap
            if arrays.is_map(ret) and any(isinstance(x, Unk) for x in walk(ret.elt)):
                vector = isinstance(ret.elt, Vec) or is_vec(ret.elt) or (ret.source or "").startswith("rows:")
                ret = replace(ret, elt=Prop(arrays.element(ret), _norm_attr(name),
                                            "vector" if vector else None))
            return ret if ret is not None else Unk("void:" + name)
        cpp = self._cpp_method(obj, name, args, env, mod, depth, e)
        if cpp is not None:
            return cpp
        if args:
            self.actuations.append((name, tuple(args)))
        return Unk("call:" + name, tuple(args))

    def _cpp_method(self, obj: S, name: str, args: List[S], env: Env,
                    mod: ir.Module, depth: int, e: ir.Call) -> Optional[S]:
        if name in ("push_back", "emplace_back", "push_front", "append"):
            if isinstance(e.fn, ir.Attr):
                v = args[0] if args else Unk("elt")
                self._store(e.fn.base, Fn("_append", (obj, v)), env, mod, depth)
            return Unk("void:" + name)
        if name == "clear":
            if isinstance(e.fn, ir.Attr):
                self._store(e.fn.base, Vec(()), env, mod, depth)
            return Unk("void:clear")
        if name in ("pop_front", "pop_back", "erase", "reserve", "resize",
                    "shrink_to_fit", "sleep"):
            return Unk("void:" + name)
        if name == "size":
            return self._builtin("len", [obj]) or fn("len", obj)
        if name == "empty":
            n_ = self._builtin("len", [obj]) or fn("len", obj)
            return cmp("==", n_, ZERO)
        if name in ("begin", "end", "cbegin", "cend", "front", "back", "at"):
            return obj
        if name == "norm":
            return fn("norm", obj)
        if name == "squaredNorm":
            return pow_(fn("norm", obj), Const(2.0))
        if name in ("normalized", "normalize"):
            return fn("unit", obj)
        if name == "dot" and args:
            return fn("dot", obj, args[0])
        if name == "cwiseProduct" and args:
            return mul(obj, args[0])
        if name == "getParam" and len(e.args) == 2 \
                and isinstance(e.args[0], ir.Str):
            pname = e.args[0].value.strip("/").strip('"').replace("/", "_")
            self._store(e.args[1], Sym(pname, "param"), env, mod, depth)
            self._note_once("ROS parameter '%s' bound as a named param" % pname)
            return Unk("void:getParam")
        if name == "publish" and args:
            v = args[0]
            if not (is_vec(v) or isinstance(v, Vec)):
                comps = [env.heap.get(Prop(v, c)) for c in ("x", "y", "z")]
                if comps[0] is not None and comps[1] is not None:
                    v = Vec(tuple(c if c is not None else ZERO for c in comps))
            if is_vec(v) or isinstance(v, (Vec, Idx, Clamp)):
                self.actuations.append(("publish", (v,)))
            return Unk("void:publish")
        return None

    def _inline(self, f: ir.FuncDef, fmod: ir.Module, args: List[S],
                env: Env, depth: int, name: str) -> S:
        if self.use_shortcuts:
            sc = self._shortcut(name, args)
            if sc is not None:
                return sc
        call_args = dict(zip(f.params, args))
        ret, env2 = self._call_function(f, fmod, call_args, env, depth + 1)
        env.heap = env2.heap
        return ret if ret is not None else Unk("void:" + name)

    def _shortcut(self, name: str, args: List[S]) -> Optional[S]:
        if name in ("magnitude", "get_magnitude") and args:
            v = Vec(tuple(args)) if len(args) > 1 else args[0]
            return fn("norm", v)
        if name == "get_distance" and len(args) == 2:
            return fn("norm", sub(args[0], args[1]))
        if name in ("unit_vector", "normalize") and args:
            return fn("unit", args[0])
        if name == "angle_between" and len(args) == 2:
            return fn("angle", args[0], args[1])
        if name == "limit_magnitude" and len(args) >= 2:
            lo = args[2] if len(args) > 2 else None
            return Clamp(args[0], hi=args[1], lo=lo)
        if name == "set_magnitude" and len(args) == 2:
            return Clamp(args[0], hi=args[1], lo=args[1], exact=True)
        if name == "dot" and len(args) == 2:
            return fn("dot", args[0], args[1])
        return None

    def _external(self, m: str, name: str, args: List[S]) -> S:
        if name in ("distance_transform_edt", "bwdist") and args:
            self.trace.occupancies.append(args[0])
            return fields.bwdist(args[0], self.notes)
        if m in ("math", "numpy", "numpy.linalg"):
            if name in MATH_FNS:
                return fn(MATH_FNS[name], *args)
            if name == "pow" and len(args) == 2:
                return self._binop("**", args[0], args[1])
            if name in ("norm",):
                return fn("norm", *args)
            if name == "array" and args:
                return args[0]
            if name == "zeros" and args and isinstance(args[0], Const):
                return Vec(tuple([ZERO] * int(args[0].v)))
            if name == "zeros" and args and isinstance(args[0], Vec) \
                    and len(args[0].items) == 2:
                return Fn("_grid0", ())
            if name == "meshgrid" and len(args) == 2:
                return Vec((fields.GC, fields.GR))
            if name == "arange" and args:
                return Fn("arange", tuple(args))
            if name == "gradient" and args:
                self.trace.potentials.append(args[0])
                gradient = fields.gradient(args[0], self.notes)
                self.trace.gradients.append(gradient)
                self.trace.gradient_ids.update(id(x) for x in gradient.items
                                               if isinstance(x, (Add, Mul, Pow, Fn)))
                return gradient
            if name == "dot" and len(args) == 2:
                return fn("dot", args[0], args[1])
            if name in ("sum", "mean") and args and isinstance(args[0], Dom):
                d = args[0]
                return Reduce("sum" if name == "sum" else "mean",
                              d.var or "_e", d, d.elt or Obj(d.var or "_e", d.name, source_of(d)),
                              guard=d.guard)
            if name == "mean" and args and isinstance(args[0], Vec):
                items = args[0].items
                return mul(add(*items), Const(1.0 / len(items)))
            if name == "mean" and args and not isinstance(
                    args[0], (Dom, Prop, Obj, Unk)):
                return args[0]
            if name in ("round", "asarray", "float64", "abs"):
                return args[0] if args else Unk(name)
            if name == "isscalar" and args:
                return Const(0.0 if is_vec(args[0]) else 1.0)
        sc = self._shortcut(name, args) if self.use_shortcuts else None
        if sc is not None:
            return sc
        return Unk("call:%s.%s" % (m, name), tuple(args))

    def _builtin(self, name: str, args: List[S]) -> Optional[S]:
        if name in MATH_FNS and args:
            return fn(MATH_FNS[name], *args)
        if name in ("atoi", "atof", "atol", "stod", "stoi", "double") and args:
            return args[0]
        if name in ("max_element", "min_element") and args:
            return Fn("_arg%s_it" % name[:3], (args[0],))
        if name == "distance" and len(args) == 2 and isinstance(args[1], Fn) \
                and args[1].name in ("_argmax_it", "_argmin_it"):
            return Fn("argmax" if args[1].name == "_argmax_it" else "argmin",
                      (args[1].args[0],))
        if name == "len" and args and arrays.length(args[0]) is not None:
            return arrays.length(args[0])
        if name == "len" and args and is_vec(args[0]) and not isinstance(args[0], Vec):
            return Const(float(self.dims))
        if name == "range" and len(args) >= 2:
            listed = arrays.index_list(args[0], args[1], self.counts)
            if listed is not None:
                return listed
        if name == "len" and args:
            a = args[0]
            if isinstance(a, Vec):
                return Const(float(len(a.items)))
            if isinstance(a, Dom):
                return Count(a.var or "_e", a, a.guard)
            if isinstance(a, (Prop, Obj, Unk)):
                return Count("_e", Dom(_dom_name(a), base=a))
            return fn("len", a)
        if name == "sum" and args:
            a = args[0]
            if isinstance(a, Fn) and a.name == "_dotgen":
                return fn("dot", *a.args)
            if isinstance(a, Dom):
                return Reduce("sum", a.var or "_e", a,
                              a.elt or Obj(a.var or "_e", a.name, source_of(a)), guard=a.guard)
            return fn("sum", a)
        if name in ("list", "tuple", "float", "int", "round"):
            return args[0] if args else Unk(name)
        if name == "abs":
            return fn("abs", *args)
        if name in ("min", "max"):
            return fn(name, *args)
        if name == "pow" and len(args) == 2:
            return self._binop("**", args[0], args[1])
        if name in ("range", "zip", "enumerate"):
            return Fn(name, tuple(args))
        if name in ("print", "isinstance", "getattr", "str", "repr", "format"):
            return Unk(name)
        return None

    def _note_once(self, text: str):
        if text not in self.notes:
            self.notes.append(text)

    def _binop(self, op: str, l: S, r: S) -> S:
        mapped = arrays.binop(op, l, r, self._binop)
        if mapped is not None:
            return mapped
        if op == "&":
            return and_(l, r)
        if op == "|":
            return OrS((l, r))
        if op == "+":
            return add(l, r)
        if op == "-":
            return sub(l, r)
        if op == "*":
            return mul(l, r)
        if op == "/":
            return div(l, r)
        if op == "**":
            if isinstance(l, Const) and abs(l.v - math.e) < 1e-12:
                return fn("exp", r)
            return pow_(l, r)
        if op == "//":
            return div(l, r)
        if op == "%":
            return fn("mod", l, r)
        if op == "@":
            return fn("dot", l, r)
        return Unk("op" + op, (l, r))


def _deref(base: S, env: Env) -> S:
    if isinstance(base, Prop):
        return env.heap.get(base, base)
    return base


def _norm_cmp(op: str) -> str:
    return {"is": "==", "is_not": "!=", "in": "in", "not_in": "not_in"}.get(op, op)


def _dom_name(v: S) -> str:
    if isinstance(v, Dom):
        return v.name
    if isinstance(v, Prop):
        return v.name
    if isinstance(v, Sym):
        return v.name
    if isinstance(v, Obj):
        return v.name
    if isinstance(v, Unk) and v.tag.startswith("call:"):
        return v.tag[5:]
    if isinstance(v, Fn) and v.name in ("range", "range_incl"):
        return "range"
    if isinstance(v, Add) and all(isinstance(p, (Sym, Dom, Prop)) for p in v.parts):
        return "+".join(_dom_name(p) for p in v.parts)
    return "set"


def _generator_shape(f: ir.FuncDef) -> Optional[
        Tuple[str, ir.Expr, List[ir.Stmt], List[ir.Expr]]]:
    body = [s for s in f.body if not isinstance(s, (ir.Pass, ir.Return))]
    if len(body) != 1 or not isinstance(body[0], ir.For):
        return None
    loop = body[0]
    if not isinstance(loop.target, ir.NameE):
        return None
    var = loop.target.id
    prefix: List[ir.Stmt] = []
    conds: List[ir.Expr] = []
    inner = list(loop.body)
    while inner:
        head = inner[0]
        if isinstance(head, (ir.Assign, ir.AugAssign)):
            prefix.append(head)
            inner = inner[1:]
        elif isinstance(head, ir.If) and not head.orelse and len(inner) == 1:
            conds.append(head.cond)
            inner = list(head.body)
        else:
            break
    has_yield = any(isinstance(s, ir.ExprStmt) and isinstance(s.value, ir.Opaque)
                    and "yield" in s.value.tag for s in inner)
    if has_yield:
        return var, loop.iter, prefix, conds
    return None


def _delta_of(new: S, old: S) -> Optional[S]:
    if is_zero(old):
        return new
    if new == old:
        return ZERO
    if isinstance(new, Add) and old in new.parts:
        rest = list(new.parts)
        rest.remove(old)
        return add(*rest)
    if isinstance(new, Vec) and isinstance(old, Vec) and len(new.items) == len(old.items):
        ds = [_delta_of(n, o) for n, o in zip(new.items, old.items)]
        if all(d is not None for d in ds):
            return Vec(tuple(ds))
    if isinstance(new, Cond):
        dt = _delta_of(new.then, old)
        de = _delta_of(new.orelse, old)
        if dt is not None and de is not None:
            return cond(new.test, dt, de)
    return None


def _peel_append(new: S, base: S) -> Optional[Tuple[S, Optional[S]]]:
    guard = None
    x = new
    if isinstance(x, Cond):
        if isinstance(x.then, Fn) and x.then.name == "_append" \
                and x.orelse == base:
            guard = x.test
            x = x.then
        elif isinstance(x.orelse, Fn) and x.orelse.name == "_append" \
                and x.then == base:
            guard = NotS(x.test)
            x = x.orelse
    if isinstance(x, Fn) and x.name == "_append":
        root, elt = x.args
        if root == base or _is_empty_container(root):
            return elt, guard
    return None


def _is_empty_container(v: S) -> bool:
    return isinstance(v, Vec) and not v.items


def _container_value(v: S) -> Optional[S]:
    if not (isinstance(v, Fn) and v.name == "_append"):
        return None
    items: List[S] = []
    x = v
    while isinstance(x, Fn) and x.name == "_append":
        items.append(x.args[1])
        x = x.args[0]
    items.reverse()
    if _is_empty_container(x):
        return Vec(tuple(items))
    if isinstance(x, Dom):
        return x
    return None


def _component_store(cur: S, i: S, v: S) -> S:
    if isinstance(cur, Dom) and cur.elt is not None and not isinstance(i, Const):
        return Dom(cur.name, cur.base, cur.var, cur.guard, v, cur.source)
    if isinstance(cur, Vec) and isinstance(i, Const):
        items = list(cur.items)
        k = int(i.v)
        if 0 <= k < len(items):
            items[k] = v
            return Vec(tuple(items))
    if isinstance(i, Const):
        k = int(i.v)
        n = max(k + 1, 2)
        items = [idx(cur, Const(float(j))) for j in range(n)]
        items[k] = v
        return Vec(tuple(items))
    if i == _AXIS:
        return v
    return Unk("store", (cur, i, v))


def _conjuncts(g: S):
    return g.parts if isinstance(g, AndS) else (g,)


def _relative(guard: S, outer: S) -> S:
    if outer == Const(1.0) or outer is None:
        return guard
    drop = set(_conjuncts(outer))
    rest = [c for c in _conjuncts(guard) if c not in drop]
    return and_(*rest) if rest else Const(1.0)


def _merge(guard: S, a: S, b: S) -> S:
    merged = arrays.merge(guard, a, b)
    return merged if merged is not None else cond(guard, a, b)


def _merge_env(guard: S, a: Env, b: Env) -> Env:
    out = Env(dict(b.vars), dict(b.heap))
    for k, v in a.vars.items():
        if k not in out.vars:
            out.vars[k] = v
        elif out.vars[k] != v:
            out.vars[k] = _merge(guard, v, out.vars[k])
    for k, v in a.heap.items():
        if k not in out.heap:
            out.heap[k] = v
        elif out.heap[k] != v:
            out.heap[k] = _merge(guard, v, out.heap[k])
    return out


def _merge_heap(guard: S, a: Dict[S, S], b: Dict[S, S]) -> Dict[S, S]:
    out = dict(b)
    for k, v in a.items():
        if k not in out:
            out[k] = v
        elif out[k] != v:
            out[k] = cond(guard, v, out[k])
    return out


def _merge_paths_env(paths: List[Path], fallback: Env) -> Env:
    falls = [q for q in paths if q.kind in ("fall", "break", "continue")]
    if not falls:
        return fallback
    env = falls[0].env
    for q in falls[1:]:
        env = _merge_env(q.guard, q.env, env)
    return env


def _coarse_merge(paths: List[Path]) -> Path:
    env = paths[0].env
    for q in paths[1:]:
        env = _merge_env(q.guard, q.env, env)
    return Path(paths[0].guard, env, paths[0].kind, paths[0].value)


def _fold_axis_env(env: Env, notes: List[str]):
    from .canonical import axis_unfold
    for k in list(env.vars):
        env.vars[k] = axis_unfold(env.vars[k], _AXIS, notes)
    for k in list(env.heap):
        env.heap[k] = axis_unfold(env.heap[k], _AXIS, notes)
