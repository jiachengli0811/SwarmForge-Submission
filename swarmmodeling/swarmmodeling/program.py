from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from . import ir


def _counted(body: List[ir.Stmt]) -> List[ir.Stmt]:
    out: List[ir.Stmt] = []
    k = 0
    while k < len(body):
        s, nxt = body[k], body[k + 1] if k + 1 < len(body) else None
        if isinstance(s, ir.Assign) and isinstance(s.target, ir.NameE) and isinstance(nxt, ir.While) \
                and isinstance(nxt.cond, ir.CmpOp) and nxt.cond.op in ("<", "<=") \
                and isinstance(nxt.cond.left, ir.NameE) and nxt.cond.left.id == s.target.id and nxt.body:
            var, last = s.target.id, nxt.body[-1]
            steps = isinstance(last, ir.Assign) and isinstance(last.target, ir.NameE) \
                and last.target.id == var and isinstance(last.value, ir.BinOp) and last.value.op == "+" \
                and isinstance(last.value.left, ir.NameE) and last.value.left.id == var \
                and isinstance(last.value.right, ir.Num) and last.value.right.value == 1
            steps = steps or (isinstance(last, ir.AugAssign) and last.op == "+" and isinstance(last.target, ir.NameE)
                              and last.target.id == var and isinstance(last.value, ir.Num) and last.value.value == 1)
            others = [x for x in ir.walk_stmts(nxt.body[:-1]) if isinstance(x, (ir.Assign, ir.AugAssign, ir.For))
                      and isinstance(getattr(x, "target", None), ir.NameE) and x.target.id == var]
            if steps and not others:
                stop = nxt.cond.right if nxt.cond.op == "<" else \
                    ir.BinOp("+", nxt.cond.right, ir.Num(1.0), loc=nxt.loc)
                out.append(s)
                out.append(ir.For(ir.NameE(var, loc=nxt.loc),
                                  ir.Call(ir.NameE("range", loc=nxt.loc), [s.value, stop], loc=nxt.loc),
                                  _counted(nxt.body[:-1]), loc=nxt.loc))
                k += 2
                continue
        out.append(s)
        k += 1
    return out


def norm_attr(name: str) -> str:
    return name.lstrip("_")


def norm_param(name: str) -> str:
    return name.lstrip("_").lower()


def qualified(expr: ir.Expr, mod: ir.Module) -> str:
    if isinstance(expr, ir.NameE):
        if expr.id in mod.from_imports:
            module, name = mod.from_imports[expr.id]
            return module + "." + name
        return mod.imports.get(expr.id, "")
    if isinstance(expr, ir.Attr):
        base = qualified(expr.base, mod)
        return base + "." + expr.name if base else ""
    return ""


ARRAY_BUILDERS = {"numpy.array", "numpy.asarray", "numpy.zeros", "numpy.ones", "numpy.empty",
                  "numpy.full", "numpy.zeros_like", "numpy.ones_like", "numpy.copy"}
VECTOR_CONSUMERS = {"numpy.linalg.norm", "scipy.linalg.norm", "numpy.dot", "numpy.cross"}


class Program:
    def __init__(self):
        self.modules: Dict[str, ir.Module] = {}
        self.functions: Dict[str, Tuple[ir.FuncDef, ir.Module]] = {}
        self.methods: Dict[str, List[Tuple[ir.FuncDef, ir.Module]]] = {}
        self.params: Dict[str, float] = {}
        self.consts: Dict[str, Dict[str, float]] = {}
        self.class_consts: Dict[str, float] = {}
        self.state_attrs: set = set()
        self.globals: Dict[str, Optional[ir.Expr]] = {}
        self._vectors: Optional[Set[str]] = None
        self.assigned_attrs: Set[str] = set()
        self._top_known: Dict[object, float] = {}
        self._top_assigned: Set[tuple] = set()
        self._matlab: List[ir.Module] = []

    def load(self, path: str, name: str = None) -> ir.Module:
        from .frontend import parse_file
        mod = parse_file(path, name)
        if mod.language == "matlab":
            self._matlab = [m for m in self._matlab if m.name != mod.name] + [mod]
        else:
            self.add(mod)
        return mod

    def finish(self) -> "Program":
        if self._matlab:
            from .frontend.matlab_lower import lower_program
            for mod in lower_program(self._matlab):
                self.add(mod)
            self._matlab = []
        for mod in self.modules.values():
            mod.body = _counted(mod.body)
            for s in ir.walk_stmts(mod.body):
                if isinstance(s, (ir.FuncDef, ir.For, ir.While, ir.If)):
                    s.body = _counted(s.body)
                if isinstance(s, ir.If):
                    s.orelse = _counted(s.orelse)
        self._vectors = None
        return self

    def element_classes(self) -> Dict[str, str]:
        classes = {m.name.split(".")[-1] for m in self.modules.values()}
        classes = {s.name for m in self.modules.values() for s in ir.walk_stmts(m.body)
                   if isinstance(s, ir.ClassDef)}
        found: Dict[str, Set[str]] = {}

        def built(e):
            return e.fn.id if isinstance(e, ir.Call) and isinstance(e.fn, ir.NameE) \
                and e.fn.id in classes else None

        for body, _, _ in self.bodies():
            for s in ir.walk_stmts(body):
                if isinstance(s, ir.Assign) and isinstance(s.target, ir.NameE) \
                        and isinstance(s.value, (ir.ListE, ir.TupleE)):
                    for item in s.value.items:
                        if built(item):
                            found.setdefault(s.target.id, set()).add(built(item))
                if isinstance(s, ir.ExprStmt) and isinstance(s.value, ir.Call) \
                        and isinstance(s.value.fn, ir.Attr) and s.value.fn.name == "append" \
                        and isinstance(s.value.fn.base, ir.NameE) and s.value.args and built(s.value.args[0]):
                    found.setdefault(s.value.fn.base.id, set()).add(built(s.value.args[0]))
        return {name: cls.pop() for name, cls in found.items() if len(cls) == 1}

    def add(self, mod: ir.Module) -> ir.Module:
        self.modules[mod.name] = mod
        self._vectors = None
        for qual, f in ir.functions(mod).items():
            self.functions["%s.%s" % (mod.name, qual)] = (f, mod)
            if f.name != "__init__":
                for s in ir.walk_stmts(f.body):
                    if isinstance(s, (ir.Assign, ir.AugAssign)) \
                            and isinstance(s.target, ir.Attr):
                        self.state_attrs.add(norm_attr(s.target.name))
            if f.cls is not None:
                self.methods.setdefault(f.name, []).append((f, mod))
            if f.name == "__init__" and f.cls is not None and f.params:
                for s in ir.walk_stmts(f.body):
                    if isinstance(s, ir.Assign) and isinstance(s.target, ir.Attr) \
                            and isinstance(s.target.base, ir.NameE) \
                            and s.target.base.id == f.params[0]:
                        v = ir._const_value(s.value)
                        if v is not None:
                            self.class_consts.setdefault(norm_attr(s.target.name), v)
        self.consts[mod.name] = ir.module_constants(mod)
        for cname, v in self.consts[mod.name].items():
            self.params[norm_param(cname)] = v
        for s in mod.body:
            if isinstance(s, ir.Assign) and isinstance(s.target, ir.NameE) \
                    and s.target.id not in self.consts[mod.name]:
                self.globals.setdefault(s.target.id, s.value)
        for body, _, _ in [(f.body, None, None) for f in ir.functions(mod).values()] + \
                [(mod.body, None, None)]:
            for s in ir.walk_stmts(body):
                if isinstance(s, (ir.Assign, ir.AugAssign)) and isinstance(s.target, ir.Attr):
                    self.assigned_attrs.add(norm_attr(s.target.name))
        self._top_level_constants(mod)
        return mod

    def _top_level_constants(self, mod: ir.Module):
        known = self._top_known

        def value(e):
            c = ir._const_value(e)
            if c is not None:
                return c
            if isinstance(e, ir.NameE):
                return known.get(e.id)
            if isinstance(e, ir.Attr) and isinstance(e.base, ir.NameE):
                return known.get((e.base.id, e.name))
            if isinstance(e, ir.UnOp) and e.op == "-":
                v = value(e.operand)
                return -v if v is not None else None
            if isinstance(e, ir.BinOp) and e.op in ("+", "-", "*", "/"):
                a, b = value(e.left), value(e.right)
                if a is None or b is None or (e.op == "/" and b == 0):
                    return None
                return {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[e.op]
            return None

        def decide(c):
            if isinstance(c, ir.UnOp) and c.op == "not":
                d = decide(c.operand)
                return None if d is None else not d
            if isinstance(c, ir.Call) and isinstance(c.fn, ir.NameE) and c.fn.id == "hasattr" \
                    and len(c.args) == 2 and isinstance(c.args[0], ir.NameE) \
                    and isinstance(c.args[1], ir.Str):
                return (c.args[0].id, c.args[1].value) in assigned
            v = value(c)
            return None if v is None else bool(v)

        assigned = self._top_assigned

        def run(stmts):
            for s in stmts:
                if isinstance(s, ir.Assign) and isinstance(s.target, ir.Attr) \
                        and isinstance(s.target.base, ir.NameE):
                    key = (s.target.base.id, s.target.name)
                    assigned.add(key)
                    v = value(s.value)
                    if v is None:
                        known.pop(key, None)
                    else:
                        known[key] = v
                elif isinstance(s, ir.Assign) and isinstance(s.target, ir.NameE):
                    v = value(s.value)
                    if v is not None:
                        known[s.target.id] = v
                elif isinstance(s, ir.If):
                    d = decide(s.cond)
                    if d is not None:
                        run(s.body if d else s.orelse)

        run([s for s in mod.body if not isinstance(s, (ir.FuncDef, ir.ClassDef))])
        for key, v in known.items():
            if isinstance(key, tuple):
                self.class_consts[norm_attr(key[1])] = v

    def module_fn(self, mod: ir.Module, name: str) -> Optional[Tuple[ir.FuncDef, ir.Module]]:
        got = self.functions.get("%s.%s" % (mod.name, name))
        if got:
            return got
        if name in mod.from_imports:
            m, remote = mod.from_imports[name]
            return self.functions.get("%s.%s" % (m.split(".")[-1], remote))
        for m in getattr(mod, "star_imports", []):
            got = self.functions.get("%s.%s" % (m.split(".")[-1], name))
            if got:
                return got
        return None

    def find_ctor(self, mod: ir.Module, name: str) -> Optional[Tuple[ir.FuncDef, ir.Module]]:
        mods = [mod.name] + [m.split(".")[-1] for m in
                             getattr(mod, "star_imports", [])] + list(self.modules)
        for m in mods:
            got = self.functions.get("%s.%s.__init__" % (m, name))
            if got:
                return got
        return None

    def static_value(self, expr: ir.Expr) -> Optional[float]:
        c = ir._const_value(expr)
        if c is not None:
            return c
        if isinstance(expr, ir.NameE):
            return self._top_known.get(expr.id)
        if isinstance(expr, ir.Attr) and isinstance(expr.base, ir.NameE):
            return self._top_known.get((expr.base.id, expr.name))
        return None

    def argument_value(self, entry: str, param: str, depth: int = 3) -> Optional[float]:
        f, _ = self.functions[entry]
        if param not in f.params:
            return None
        position = f.params.index(param) - (1 if f.cls else 0)
        values = set()
        for qual, (caller, _) in list(self.functions.items()) + [
                (m.name, (ir.FuncDef(m.name, [], body=[s for s in m.body if not isinstance(
                    s, (ir.FuncDef, ir.ClassDef))]), m)) for m in self.modules.values()]:
            for stmt in ir.walk_stmts(caller.body):
                for name in ("value", "target", "cond", "iter"):
                    expr = getattr(stmt, name, None)
                    if not isinstance(expr, ir.Expr):
                        continue
                    for call in ir.walk_exprs(expr):
                        if not (isinstance(call, ir.Call) and isinstance(call.fn, (ir.Attr, ir.NameE))
                                and (call.fn.name if isinstance(call.fn, ir.Attr) else call.fn.id) == f.name
                                and position < len(call.args)):
                            continue
                        arg = call.args[position]
                        v = self.static_value(arg)
                        if v is None and isinstance(arg, ir.NameE) and arg.id in caller.params \
                                and depth > 0 and qual in self.functions:
                            v = self.argument_value(qual, arg.id, depth - 1)
                        values.add(v)
        return values.pop() if len(values) == 1 and None not in values else None

    def bodies(self):
        for f, mod in self.functions.values():
            yield f.body, f.params, mod
        for mod in self.modules.values():
            yield [s for s in mod.body if not isinstance(s, (ir.FuncDef, ir.ClassDef))], [], mod

    def vector_fields(self) -> Set[str]:
        if self._vectors is not None:
            return self._vectors
        vectors: Set[str] = set()
        containers: Set[str] = set()
        copies: Set[Tuple[str, str]] = set()
        indexed_params: Dict[int, Set[int]] = {}
        components: Dict[str, Set[str]] = {}

        def field(e):
            return norm_attr(e.name) if isinstance(e, ir.Attr) else None

        def is_array(value, mod, target=None):
            if isinstance(value, (ir.ListE, ir.TupleE)) and len(value.items) >= 2:
                return not any(field(item) == target for item in value.items)
            if isinstance(value, ir.MatrixE):
                return True
            if isinstance(value, ir.Call) and qualified(value.fn, mod) in ARRAY_BUILDERS:
                return True
            if isinstance(value, ir.BinOp) and value.op in ("+", "-", "*", "/"):
                return is_array(value.left, mod) or is_array(value.right, mod)
            return False

        for body, params, mod in self.bodies():
            defined: Dict[str, Set[str]] = {}
            for stmt in ir.walk_stmts(body):
                if isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameE):
                    made_of, pending = set(), [stmt.value]
                    while pending:
                        e = pending.pop()
                        if isinstance(e, ir.BinOp) and e.op in ("+", "-"):
                            pending.extend((e.left, e.right))
                        elif isinstance(e, ir.UnOp) and e.op == "-":
                            pending.append(e.operand)
                        elif field(e):
                            made_of.add(field(e))
                        elif isinstance(e, ir.NameE):
                            made_of |= defined.get(e.id, set())
                    defined[stmt.target.id] = made_of
                exprs = [c for c in (getattr(stmt, "value", None), getattr(stmt, "target", None),
                                     getattr(stmt, "cond", None), getattr(stmt, "iter", None))
                         if isinstance(c, ir.Expr)]
                if isinstance(stmt, (ir.Assign, ir.AugAssign)):
                    target = field(stmt.target)
                    if target:
                        if is_array(stmt.value, mod, target):
                            vectors.add(target)
                        source = field(stmt.value)
                        if source:
                            copies.add((source, target))
                for expr in exprs:
                    for node in ir.walk_exprs(expr):
                        if isinstance(node, (ir.Attr, ir.Call)):
                            inner = node.base if isinstance(node, ir.Attr) else (
                                node.fn.base if isinstance(node.fn, ir.Attr) else None)
                            if isinstance(inner, ir.Index) and field(inner.base):
                                containers.add(field(inner.base))
                        if isinstance(node, ir.Attr) and node.name in ("x", "y", "z") \
                                and field(node.base):
                            components.setdefault(field(node.base), set()).add(node.name)
                        if isinstance(node, ir.Index) and isinstance(node.index, (ir.Num, ir.NameE)):
                            name = field(node.base)
                            if name:
                                vectors.add(name)
                        if isinstance(node, ir.Call) and qualified(node.fn, mod) in VECTOR_CONSUMERS:
                            for arg in node.args:
                                for inner in ir.walk_exprs(arg):
                                    name = field(inner)
                                    if name:
                                        vectors.add(name)
                                    if isinstance(inner, ir.NameE):
                                        vectors.update(defined.get(inner.id, set()))
        vectors |= {f for f, named in components.items() if len(named) >= 2}
        grown = True
        while grown:
            grown = False
            for source, target in copies:
                if source in vectors and target not in vectors:
                    vectors.add(target)
                    grown = True
        self._vectors = vectors - containers
        return self._vectors
