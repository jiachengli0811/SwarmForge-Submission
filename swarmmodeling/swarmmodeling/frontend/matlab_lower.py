from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

from .. import ir

NUMPY = "numpy"


def _np(*path, loc=None):
    expr = ir.NameE(NUMPY, loc=loc)
    for name in path:
        expr = ir.Attr(expr, name, loc=loc)
    return expr


def _num(v, loc=None):
    return ir.Num(float(v), loc=loc)


def _minus_one(e: ir.Expr) -> ir.Expr:
    if isinstance(e, ir.Num):
        return _num(e.value - 1, e.loc)
    if isinstance(e, ir.BinOp) and e.op == "+" and isinstance(e.right, ir.Num) and e.right.value == 1:
        return e.left
    return ir.BinOp("-", e, _num(1, getattr(e, "loc", None)), loc=getattr(e, "loc", None))


def _plus_one(e: ir.Expr) -> ir.Expr:
    if isinstance(e, ir.Num):
        return _num(e.value + 1, e.loc)
    return ir.BinOp("+", e, _num(1, getattr(e, "loc", None)), loc=getattr(e, "loc", None))


class _Symbols:
    def __init__(self, modules: List[ir.Module]):
        self.functions: Set[str] = set()
        self.classes: Set[str] = set()
        self.methods: Set[str] = set()
        self.properties: Set[str] = set()
        for mod in modules:
            for s in mod.body:
                if isinstance(s, ir.FuncDef):
                    (self.methods if s.cls else self.functions).add(s.name)
                elif isinstance(s, ir.ClassDef):
                    self.classes.add(s.name)
                    for m in s.body:
                        if isinstance(m, ir.FuncDef):
                            self.methods.add(m.name)
                        elif isinstance(m, ir.Assign) and isinstance(m.target, ir.NameE):
                            self.properties.add(m.target.id)
            if not any(isinstance(s, (ir.FuncDef, ir.ClassDef)) for s in mod.body):
                self.functions.add(mod.name)


def _logical(e) -> bool:
    return isinstance(e, (ir.CmpOp, ir.BoolOp)) or (isinstance(e, ir.UnOp) and e.op == "~") \
        or (isinstance(e, ir.BinOp) and e.op in ("&", "|"))


class _Lower:
    def __init__(self, symbols: _Symbols, locals_: Set[str], masks: Set[str] = frozenset()):
        self.sym, self.locals, self.masks = symbols, locals_, masks

    def expr(self, e: ir.Expr, nargout: int = 1) -> ir.Expr:
        L = getattr(e, "loc", None)
        if isinstance(e, ir.NameE):
            if e.id in ("true", "false") and e.id not in self.locals:
                return _num(1 if e.id == "true" else 0, L)
            if e.id == "pi" and e.id not in self.locals:
                return ir.Attr(ir.NameE("math", loc=L), "pi", loc=L)
            if e.id in self.sym.functions and e.id not in self.locals:
                return ir.Call(e, [], loc=L)
            return e
        if isinstance(e, (ir.Num, ir.Str, ir.Opaque, ir.EndE, ir.AllE)):
            return e
        if isinstance(e, ir.Attr):
            base = self.expr(e.base)
            if e.name in self.sym.methods and e.name not in self.sym.properties:
                return ir.Call(ir.Attr(base, e.name, loc=L), [], loc=L)
            return ir.Attr(base, e.name, loc=L)
        if isinstance(e, ir.Apply):
            return self.apply(e, nargout)
        if isinstance(e, ir.BinOp):
            op = {".*": "*", "./": "/", ".^": "**", "^": "**", ".\\": "\\"}.get(e.op, e.op)
            return ir.BinOp(op, self.expr(e.left), self.expr(e.right), loc=L)
        if isinstance(e, ir.CmpOp):
            return ir.CmpOp(e.op, self.expr(e.left), self.expr(e.right), loc=L)
        if isinstance(e, ir.BoolOp):
            return ir.BoolOp(e.op, [self.expr(p) for p in e.parts], loc=L)
        if isinstance(e, ir.UnOp):
            operand = self.expr(e.operand)
            if e.op == "~":
                return ir.UnOp("not", operand, loc=L)
            if e.op in ("'", ".'"):
                return ir.Call(_np("transpose", loc=L), [operand], loc=L)
            return ir.UnOp(e.op, operand, loc=L)
        if isinstance(e, ir.MatrixE):
            rows = [[self.expr(x) for x in row] for row in e.rows]
            if not rows:
                return ir.ListE([], loc=L)
            if len(rows) == 1:
                return ir.ListE(rows[0], loc=L)
            if all(len(r) == 1 for r in rows):
                return ir.ListE([r[0] for r in rows], loc=L)
            return ir.ListE([ir.ListE(r, loc=L) for r in rows], loc=L)
        if isinstance(e, ir.RangeE):
            args = [self.expr(e.start), _plus_one(self.expr(e.stop))]
            if e.step is not None:
                args.append(self.expr(e.step))
            return ir.Call(ir.NameE("range", loc=L), args, loc=L)
        if isinstance(e, ir.TupleE):
            return ir.TupleE([self.expr(x) for x in e.items], loc=L)
        return e

    def apply(self, e: ir.Apply, nargout: int) -> ir.Expr:
        L = e.loc
        base = e.base
        if isinstance(base, ir.NameE) and base.id not in self.locals:
            return self.call(base.id, e.args, nargout, L)
        if isinstance(base, ir.Attr) and base.name in self.sym.methods \
                and base.name not in self.sym.properties:
            return ir.Call(ir.Attr(self.expr(base.base), base.name, loc=L),
                           [self.expr(a) for a in e.args], loc=L)
        return self.index(self.expr(base), e.args, L)

    def index(self, base: ir.Expr, args, L) -> ir.Expr:
        out = base
        for arg in args:
            out = ir.Index(out, self.subscript(arg, base, L), loc=L)
        return out

    def subscript(self, arg: ir.Expr, base: ir.Expr, L) -> ir.Expr:
        if isinstance(arg, ir.AllE):
            return ir.SliceE(None, None, loc=L)
        if isinstance(arg, ir.RangeE) and arg.step is None:
            return ir.SliceE(_minus_one(self.with_end(arg.start, base)),
                             self.with_end(arg.stop, base), loc=L)
        if _logical(arg) or (isinstance(arg, ir.NameE) and arg.id in self.masks):
            return self.expr(arg)
        return _minus_one(self.with_end(arg, base))

    def with_end(self, e: ir.Expr, base: ir.Expr) -> ir.Expr:
        if isinstance(e, ir.EndE):
            return ir.Call(ir.NameE("len"), [base], loc=e.loc)
        e = self.expr(e)
        if isinstance(e, ir.BinOp):
            return ir.BinOp(e.op, self.with_end(e.left, base), self.with_end(e.right, base), loc=e.loc)
        return e

    def call(self, name: str, args, nargout: int, L) -> ir.Expr:
        a = [self.expr(x) for x in args]

        def np_call(*path, kwargs=None, argv=None):
            return ir.Call(_np(*path, loc=L), a if argv is None else argv, kwargs or {}, loc=L)

        if name in ("zeros", "ones"):
            shape = a if len(a) > 1 else (a * 2 if a else [_num(1, L)])
            return np_call(name, argv=[ir.TupleE(shape, loc=L)])
        if name == "eye":
            return np_call("eye")
        if name in ("sqrt", "exp", "abs", "sin", "cos", "tan", "floor", "ceil", "round",
                    "atan2", "mod", "sign", "log"):
            return np_call(name)
        if name == "sum":
            return np_call("sum", kwargs={"axis": _minus_one(a[1])}, argv=a[:1]) if len(a) > 1 \
                else np_call("sum", argv=a[:1])
        if name in ("min", "max"):
            return np_call("minimum" if name == "min" else "maximum") if len(a) == 2 \
                else np_call("min" if name == "min" else "max", argv=a[:1])
        if name == "norm":
            return ir.Call(_np("linalg", "norm", loc=L), a[:1], loc=L)
        if name == "dot":
            return np_call("dot")
        if name in ("length", "numel"):
            return ir.Call(ir.NameE("len", loc=L), a[:1], loc=L)
        if name == "size":
            shape = np_call("shape", argv=a[:1])
            return ir.Index(shape, _minus_one(a[1]), loc=L) if len(a) > 1 else shape
        if name == "isfield":
            return ir.Call(ir.NameE("hasattr", loc=L), a, loc=L)
        if name == "isempty":
            return ir.CmpOp("==", np_call("size", argv=a[:1]), _num(0, L), loc=L)
        if name == "sort":
            order = ir.BinOp("+", np_call("argsort", kwargs={"kind": ir.Str("stable")}, argv=a[:1]),
                             _num(1, L), loc=L)
            return ir.TupleE([np_call("sort", argv=a[:1]), order], loc=L) if nargout == 2 \
                else np_call("sort", argv=a[:1])
        if name == "repmat":
            return np_call("tile", argv=[a[0], ir.TupleE(a[1:], loc=L)])
        if name == "find":
            return ir.BinOp("+", np_call("flatnonzero", argv=a[:1]), _num(1, L), loc=L)
        if name in ("rand", "randn"):
            return ir.Call(_np("random", name, loc=L), a, loc=L)
        if name in self.sym.classes or name in self.sym.functions:
            return ir.Call(ir.NameE(name, loc=L), a, loc=L)
        return ir.Call(ir.NameE(name, loc=L), a, loc=L)

    def target(self, t: ir.Expr) -> ir.Expr:
        if isinstance(t, ir.Apply):
            return self.index(self.target(t.base) if not isinstance(t.base, ir.NameE)
                              else t.base, t.args, t.loc)
        if isinstance(t, ir.Attr):
            return ir.Attr(self.target(t.base), t.name, loc=t.loc)
        if isinstance(t, ir.TupleE):
            return ir.TupleE([self.target(x) for x in t.items], loc=t.loc)
        return t

    def block(self, body: List[ir.Stmt]) -> List[ir.Stmt]:
        out = []
        for s in body:
            out.extend(self.stmt(s))
        return out

    def stmt(self, s: ir.Stmt) -> List[ir.Stmt]:
        L = s.loc
        if isinstance(s, ir.Assign):
            nargout = len(s.target.items) if isinstance(s.target, ir.TupleE) else 1
            value = self.expr(s.value, nargout)
            target = self.target(s.target)
            if isinstance(target, ir.TupleE) and isinstance(value, ir.TupleE):
                return [ir.Assign(t, v, loc=L) for t, v in zip(target.items, value.items)
                        if not (isinstance(t, ir.NameE) and t.id == "~")]
            return [ir.Assign(target, value, loc=L)]
        if isinstance(s, ir.If):
            return [ir.If(self.expr(s.cond), self.block(s.body), self.block(s.orelse), loc=L)]
        if isinstance(s, ir.For):
            iterable = s.iter
            if isinstance(iterable, ir.UnOp) and iterable.op in ("'", ".'"):
                iterable = iterable.operand
            return [ir.For(s.target, self.expr(iterable), self.block(s.body), loc=L)]
        if isinstance(s, ir.While):
            return [ir.While(self.expr(s.cond), self.block(s.body), loc=L)]
        if isinstance(s, ir.Return):
            return [ir.Return(self.expr(s.value) if s.value is not None else None, loc=L)]
        if isinstance(s, ir.ExprStmt):
            return [ir.ExprStmt(self.expr(s.value), loc=L)]
        return [s]


def _locals(f: ir.FuncDef) -> Set[str]:
    names = set(f.params)
    for s in ir.walk_stmts(f.body):
        targets = []
        if isinstance(s, ir.Assign):
            targets = s.target.items if isinstance(s.target, ir.TupleE) else [s.target]
        elif isinstance(s, ir.For):
            targets = [s.target]
        elif isinstance(s, ir.Return) and s.value is not None:
            targets = s.value.items if isinstance(s.value, ir.TupleE) else [s.value]
        for t in targets:
            while isinstance(t, (ir.Apply, ir.Attr, ir.Index)):
                t = t.base
            if isinstance(t, ir.NameE):
                names.add(t.id)
    return names


def _masks(body) -> Set[str]:
    return {s.target.id for s in ir.walk_stmts(body)
            if isinstance(s, ir.Assign) and isinstance(s.target, ir.NameE) and _logical(s.value)}


def _script_locals(body) -> Set[str]:
    return _locals(ir.FuncDef("<script>", [], body=body))


def _function(f: ir.FuncDef, symbols: _Symbols, cls=None) -> ir.FuncDef:
    lower = _Lower(symbols, _locals(f), _masks(f.body))
    out = ir.FuncDef(f.name, list(f.params), body=lower.block(f.body), loc=f.loc)
    out.cls = cls if cls is not None else f.cls
    if out.cls and f.name == out.cls:
        output = f.body[-1].value if f.body and isinstance(f.body[-1], ir.Return) else None
        if isinstance(output, ir.NameE):
            out.name = "__init__"
            out.params = [output.id] + list(f.params)
            out.body = [s for s in out.body if not isinstance(s, ir.Return)]
    return out


def lower_program(modules: List[ir.Module]) -> List[ir.Module]:
    for mod in modules:
        folder = Path(mod.path).parent.name
        if folder.startswith("@"):
            for s in mod.body:
                if isinstance(s, ir.FuncDef):
                    s.cls = folder[1:]
                    break
    symbols = _Symbols(modules)
    out = []
    for mod in modules:
        body = []
        script = [s for s in mod.body if not isinstance(s, (ir.FuncDef, ir.ClassDef))]
        lower = _Lower(symbols, _script_locals(script), _masks(script))
        for s in mod.body:
            if isinstance(s, ir.FuncDef):
                body.append(_function(s, symbols))
            elif isinstance(s, ir.ClassDef):
                members = [_function(m, symbols, s.name) if isinstance(m, ir.FuncDef) else m
                           for m in s.body]
                body.append(ir.ClassDef(s.name, members, loc=s.loc))
            else:
                body.extend(lower.stmt(s))
        lowered = ir.Module(mod.name, mod.path, body, loc=mod.loc, language="matlab")
        lowered.imports = {NUMPY: NUMPY, "math": "math"}
        lowered.star_imports = [other.name for other in modules if other is not mod]
        out.append(lowered)
    return out
