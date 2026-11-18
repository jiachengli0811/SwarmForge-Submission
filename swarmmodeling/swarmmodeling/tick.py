from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional

from . import ir
from .program import Program

CONTROLLED = "__controlled__"


@dataclass
class Tick:
    module: ir.Module
    params: List[str]
    body: List[ir.Stmt]
    setup: List[ir.Stmt] = field(default_factory=list)
    where: str = ""
    collection: Optional[str] = None
    free: List[str] = field(default_factory=list)


def _calls(expr):
    return [n for n in ir.walk_exprs(expr) if isinstance(n, ir.Call)]


def _targets(prog: Program, entry: str):
    f, mod = prog.functions[entry]
    method = f.name if f.cls is not None and len(prog.methods.get(f.name, [])) == 1 else None
    function = f.name if f.cls is None else None
    return f, mod, method, function


def _reaches(call: ir.Call, method, function) -> bool:
    if method and isinstance(call.fn, ir.Attr) and call.fn.name == method:
        return True
    return bool(function) and isinstance(call.fn, ir.NameE) and call.fn.id == function


def _element(loop: ir.Stmt, call: ir.Call) -> bool:
    if not (isinstance(loop, ir.For) and isinstance(loop.target, ir.NameE)):
        return False
    var = loop.target.id
    operands = [call.fn.base] if isinstance(call.fn, ir.Attr) else []
    operands += list(call.args)
    return any(isinstance(o, ir.NameE) and o.id == var for o in operands)


def _sites(body, method, function, stack=()):
    for stmt in body:
        here = stack + (stmt,)
        for name in ("value", "target", "cond", "iter"):
            expr = getattr(stmt, name, None)
            if isinstance(expr, ir.Expr):
                for call in _calls(expr):
                    if _reaches(call, method, function):
                        yield here, call
        if isinstance(stmt, ir.If):
            yield from _sites(stmt.body, method, function, here)
            yield from _sites(stmt.orelse, method, function, here)
        elif isinstance(stmt, (ir.For, ir.While)):
            yield from _sites(stmt.body, method, function, here)


def _drivers(prog: Program):
    for qual, (f, mod) in prog.functions.items():
        yield f.body, f.params, mod, qual
        for nested in ir.walk_stmts(f.body):
            if isinstance(nested, ir.FuncDef):
                yield nested.body, nested.params, mod, "%s.%s" % (qual, nested.name)
    for mod in prog.modules.values():
        yield [s for s in mod.body if not isinstance(s, (ir.FuncDef, ir.ClassDef))], [], mod, mod.name


def _iterated(loop: ir.Stmt) -> Optional[str]:
    if not (isinstance(loop, ir.For) and isinstance(loop.target, ir.NameE)):
        return None
    if isinstance(loop.iter, ir.NameE):
        return loop.iter.id
    from .arrays import subscripted
    bases = {b.id for b in subscripted(loop.body, loop.target.id) if isinstance(b, ir.NameE)}
    return bases.pop() if len(bases) == 1 else None


def _bind_controlled(body, collection):
    body = copy.deepcopy(body)

    def rename(expr, old):
        for node in ir.walk_exprs(expr):
            if isinstance(node, ir.NameE) and node.id == old:
                node.id = CONTROLLED

    for stmt in ir.walk_stmts(body):
        if isinstance(stmt, ir.For) and isinstance(stmt.target, ir.NameE) \
                and _iterated(stmt) == collection:
            old = stmt.target.id
            stmt.target.id = CONTROLLED
            for inner in ir.walk_stmts(stmt.body):
                for name in ("value", "target", "cond", "iter"):
                    expr = getattr(inner, name, None)
                    if isinstance(expr, ir.Expr):
                        rename(expr, old)
    return body


def find_tick(prog: Program, entry: str) -> Optional[Tick]:
    f, mod, method, function = _targets(prog, entry)
    if not (method or function):
        return None
    for body, params, dmod, label in _drivers(prog):
        if body is f.body:
            continue
        for stack, call in _sites(body, method, function):
            loops = [s for s in stack if isinstance(s, (ir.For, ir.While))]
            inner = next((i for i in range(len(loops) - 1, -1, -1) if _element(loops[i], call)), None)
            if inner is None:
                continue
            collection = _iterated(loops[inner])
            if collection is None:
                continue
            if inner > 0:
                time = loops[inner - 1]
                tick_body = _bind_controlled(time.body, collection)
                where = "%s: body of the loop at %s" % (label, time.loc)
            else:
                tick_body = _bind_controlled(body, collection)
                where = "%s: the function body, called once per tick" % label
            free = []
            if label == dmod.name:
                consts = prog.consts.get(dmod.name, {})
                free = sorted({s.target.id for s in ir.walk_stmts(body)
                               if isinstance(s, ir.Assign) and isinstance(s.target, ir.NameE)
                               and s.target.id not in consts and s.target.id != CONTROLLED})
            return Tick(dmod, list(params), tick_body, where=where, collection=collection,
                        free=free)
    return None
