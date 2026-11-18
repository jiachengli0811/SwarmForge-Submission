from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Loc:
    file: str
    line: int

    def __str__(self):
        return "%s:%d" % (self.file, self.line)


class Expr:
    loc: Optional[Loc] = None


@dataclass
class Num(Expr):
    value: float
    loc: Optional[Loc] = None


@dataclass
class Str(Expr):
    value: str
    loc: Optional[Loc] = None


@dataclass
class NameE(Expr):
    id: str
    loc: Optional[Loc] = None


@dataclass
class Attr(Expr):
    base: Expr
    name: str
    loc: Optional[Loc] = None


@dataclass
class Index(Expr):
    base: Expr
    index: Expr
    loc: Optional[Loc] = None


@dataclass
class Call(Expr):
    fn: Expr
    args: List[Expr] = field(default_factory=list)
    kwargs: Dict[str, Expr] = field(default_factory=dict)
    loc: Optional[Loc] = None


@dataclass
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr
    loc: Optional[Loc] = None


@dataclass
class UnOp(Expr):
    op: str
    operand: Expr
    loc: Optional[Loc] = None


@dataclass
class CmpOp(Expr):
    op: str
    left: Expr
    right: Expr
    loc: Optional[Loc] = None


@dataclass
class BoolOp(Expr):
    op: str
    parts: List[Expr] = field(default_factory=list)
    loc: Optional[Loc] = None


@dataclass
class TupleE(Expr):
    items: List[Expr] = field(default_factory=list)
    loc: Optional[Loc] = None


@dataclass
class ListE(Expr):
    items: List[Expr] = field(default_factory=list)
    loc: Optional[Loc] = None


@dataclass
class IfExp(Expr):
    cond: Expr
    then: Expr
    orelse: Expr
    loc: Optional[Loc] = None


@dataclass
class Comp(Expr):
    elt: Expr
    var: Expr
    iter: Expr
    conds: List[Expr] = field(default_factory=list)
    kind: str = "gen"
    loc: Optional[Loc] = None


@dataclass
class SliceE(Expr):
    lo: Optional[Expr] = None
    hi: Optional[Expr] = None
    loc: Optional[Loc] = None


@dataclass
class Opaque(Expr):
    tag: str
    text: str = ""
    loc: Optional[Loc] = None


@dataclass
class MatrixE(Expr):
    rows: List[List[Expr]] = field(default_factory=list)
    loc: Optional[Loc] = None


@dataclass
class Apply(Expr):
    base: Expr
    args: List[Expr] = field(default_factory=list)
    loc: Optional[Loc] = None


@dataclass
class RangeE(Expr):
    start: Expr
    stop: Expr
    step: Optional[Expr] = None
    loc: Optional[Loc] = None


@dataclass
class AllE(Expr):
    loc: Optional[Loc] = None


@dataclass
class EndE(Expr):
    loc: Optional[Loc] = None


class Stmt:
    loc: Optional[Loc] = None


@dataclass
class Assign(Stmt):
    target: Expr
    value: Expr
    loc: Optional[Loc] = None
    decl: bool = False


@dataclass
class AugAssign(Stmt):
    target: Expr
    op: str
    value: Expr
    loc: Optional[Loc] = None


@dataclass
class If(Stmt):
    cond: Expr
    body: List[Stmt] = field(default_factory=list)
    orelse: List[Stmt] = field(default_factory=list)
    loc: Optional[Loc] = None


@dataclass
class For(Stmt):
    target: Expr
    iter: Expr
    body: List[Stmt] = field(default_factory=list)
    loc: Optional[Loc] = None


@dataclass
class While(Stmt):
    cond: Expr
    body: List[Stmt] = field(default_factory=list)
    loc: Optional[Loc] = None


@dataclass
class Return(Stmt):
    value: Optional[Expr] = None
    loc: Optional[Loc] = None


@dataclass
class ExprStmt(Stmt):
    value: Expr
    loc: Optional[Loc] = None


@dataclass
class Break(Stmt):
    loc: Optional[Loc] = None


@dataclass
class Continue(Stmt):
    loc: Optional[Loc] = None


@dataclass
class Pass(Stmt):
    loc: Optional[Loc] = None


@dataclass
class FuncDef(Stmt):
    name: str
    params: List[str] = field(default_factory=list)
    defaults: Dict[str, Expr] = field(default_factory=dict)
    body: List[Stmt] = field(default_factory=list)
    cls: Optional[str] = None
    loc: Optional[Loc] = None


@dataclass
class ClassDef(Stmt):
    name: str
    body: List[Stmt] = field(default_factory=list)
    loc: Optional[Loc] = None


@dataclass
class Module(Stmt):
    name: str
    path: str
    body: List[Stmt] = field(default_factory=list)
    imports: Dict[str, str] = field(default_factory=dict)
    from_imports: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    star_imports: List[str] = field(default_factory=list)
    loc: Optional[Loc] = None
    language: str = "generic"


def walk_stmts(body: List[Stmt]):
    for s in body:
        yield s
        for sub in _children(s):
            yield from walk_stmts(sub)


def walk_exprs(expr):
    seen, pending = set(), [expr]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        for spec in fields(node) if is_dataclass(node) else ():
            value = getattr(node, spec.name)
            items = value.values() if isinstance(value, dict) else \
                value if isinstance(value, list) else [value]
            pending.extend(i for i in items if isinstance(i, Expr))


def _children(s: Stmt) -> List[List[Stmt]]:
    if isinstance(s, If):
        return [s.body, s.orelse]
    if isinstance(s, (For, While, FuncDef, ClassDef, Module)):
        return [s.body]
    return []


def functions(mod: Module) -> Dict[str, FuncDef]:
    out: Dict[str, FuncDef] = {}
    for s in mod.body:
        if isinstance(s, FuncDef):
            out[s.name if s.cls is None else "%s.%s" % (s.cls, s.name)] = s
        elif isinstance(s, ClassDef):
            for m in s.body:
                if isinstance(m, FuncDef):
                    m.cls = s.name
                    out["%s.%s" % (s.name, m.name)] = m
    return out


def module_constants(mod: Module) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s in mod.body:
        if isinstance(s, Assign) and isinstance(s.target, NameE):
            v = _const_value(s.value)
            if v is not None:
                out[s.target.id] = v
    return out


def _const_value(e: Expr) -> Optional[float]:
    if isinstance(e, Num):
        return float(e.value)
    if isinstance(e, UnOp) and e.op == "-":
        v = _const_value(e.operand)
        return -v if v is not None else None
    if isinstance(e, BinOp) and e.op in ("+", "-", "*", "/"):
        l, r = _const_value(e.left), _const_value(e.right)
        if l is None or r is None or (e.op == "/" and r == 0):
            return None
        return {"+": l + r, "-": l - r, "*": l * r, "/": l / r}[e.op]
    return None
