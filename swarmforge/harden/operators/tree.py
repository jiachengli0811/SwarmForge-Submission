from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import sympy as sp
from sympy.core.relational import Relational
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.logic.boolalg import BooleanFalse, BooleanTrue

from ...solve.record import INPUTS
from ..document import gate_of, read_site, sites
from ..grammar import is_tame

T1, T2, T3, T4, T5 = "T1", "T2", "T3", "T4", "T5"
TARGETS: Mapping[str, str] = {T1: "variable", T2: "constant",
                              T3: "arithmetic", T4: "function",
                              T5: "predicate"}

FUNCTIONS: Mapping[Any, str] = {sp.exp: "exp", sp.log: "log",
                                sp.tanh: "tanh", sp.atan: "atan",
                                sp.sinh: "sinh", sp.cos: "cos"}
_RELATIONS = (sp.StrictLessThan, sp.LessThan, sp.StrictGreaterThan,
              sp.GreaterThan, sp.Equality, sp.Unequality)
_CONNECTIVES = (sp.And, sp.Or, sp.Not)

GATE = "gate"


@dataclass(frozen=True)
class Node:
    law: str
    slot: str
    path: Tuple[int, ...]
    expr: Any
    target: str
    kind: str
    role: str = "root"
    parent: str = ""
    dynamic: bool = False

    @property
    def site(self) -> str:
        return "%s/%s" % (self.law, self.slot)

    @property
    def where(self) -> str:
        return path_text(self.path)


def path_text(path: Tuple[int, ...]) -> str:
    return ".".join(str(i) for i in path)


def path_of(text: str) -> Tuple[int, ...]:
    return tuple(int(i) for i in str(text).split(".") if i != "")


def is_input(symbol: Any) -> bool:
    return isinstance(symbol, sp.Symbol) and symbol.name in INPUTS


def reads_input(expr: Any) -> bool:
    return any(is_input(s) for s in getattr(expr, "free_symbols", ()))


def _is_sign(node: Any, parent: Any) -> bool:
    return (isinstance(parent, sp.Mul) and node == sp.Integer(-1))


def classify(node: Any, parent: Any = None,
             index: Optional[int] = None) -> Tuple[str, str]:
    if isinstance(parent, sp.Pow) and index == 1 and node.is_number:
        return T2, "exponent"
    if isinstance(node, (BooleanTrue, BooleanFalse)):
        return "", "true"
    if isinstance(node, ExprCondPair):
        return "", "pair"
    if isinstance(node, _RELATIONS):
        return T5, "relation"
    if isinstance(node, _CONNECTIVES):
        return T5, "connective"
    if isinstance(node, sp.ITE):
        return T5, "ite"
    if isinstance(node, sp.Piecewise):
        return T5, "piecewise"
    if isinstance(node, sp.Symbol):
        return (T1, "variable") if is_input(node) else (T2, "parameter")
    if getattr(node, "is_Number", False):
        return ("", "sign") if _is_sign(node, parent) else (T2, "number")
    if isinstance(node, sp.Add):
        return T3, "add"
    if isinstance(node, sp.Mul):
        if len(node.args) == 2 and node.args[0] == sp.Integer(-1):
            return T3, "neg"
        return T3, "mul"
    if isinstance(node, sp.Pow):
        k = node.args[1]
        if not k.is_number:
            return T3, "power"
        if k == sp.Rational(1, 2):
            return T3, "sqrt"
        if k == -1:
            return T3, "reciprocal"
        if k == 2:
            return T3, "square"
        return T3, "power"
    if isinstance(node, sp.Abs):
        return T3, "abs"
    if isinstance(node, sp.Min):
        return T4, "min"
    if isinstance(node, sp.Max):
        return T4, "max"
    if getattr(node, "func", None) in FUNCTIONS:
        return T4, FUNCTIONS[node.func]
    return "", type(node).__name__


def _role(parent: Any, index: int) -> str:
    if isinstance(parent, sp.Pow):
        return "exponent" if index == 1 else "base"
    if isinstance(parent, ExprCondPair):
        return "branch" if index == 0 else "condition"
    if isinstance(parent, _RELATIONS):
        return "side"
    if isinstance(parent, (sp.Add, sp.Mul)):
        return "operand"
    return "argument"


def walk(expr: Any, path: Tuple[int, ...] = (), parent: Any = None,
         index: Optional[int] = None
         ) -> Iterator[Tuple[Tuple[int, ...], Any, Any, Optional[int]]]:
    yield path, expr, parent, index
    for i, arg in enumerate(getattr(expr, "args", ())):
        yield from walk(arg, path + (i,), expr, i)


def at(expr: Any, path: Tuple[int, ...]) -> Any:
    for i in path:
        expr = expr.args[i]
    return expr


def replace_at(expr: Any, path: Tuple[int, ...], new: Any) -> Any:
    if not path:
        return new
    args = list(expr.args)
    args[path[0]] = replace_at(args[path[0]], path[1:], new)
    return expr.func(*args)


def parent_of(expr: Any, path: Tuple[int, ...]) -> Tuple[Any, int]:
    return at(expr, path[:-1]), path[-1]


def nodes_of(expr: Any, law: str, slot: str) -> List[Node]:
    out = []
    for path, node, parent, index in walk(expr):
        target, kind = classify(node, parent, index)
        out.append(Node(law, slot, path, node, target, kind,
                        "root" if parent is None else _role(parent, index),
                        "" if parent is None else classify(
                            parent, *(_grand(expr, path)))[1],
                        reads_input(node)))
    return out


def _grand(expr: Any, path: Tuple[int, ...]) -> Tuple[Any, Optional[int]]:
    if len(path) < 2:
        return None, None
    return at(expr, path[:-2]), path[-2]


def law_nodes(document: Mapping[str, Any], law: str) -> List[Node]:
    out: List[Node] = []
    for site in sites(document):
        if site.law != law:
            continue
        out.extend(nodes_of(read_site(document, site).expr, law, site.slot))
    gate = gate_of(document, law)
    out.append(Node(law, GATE, (), None, T5, "gate:%s" % gate.kind))
    if gate.gated:
        out.append(Node(law, GATE, ("feature",), None, T1, "feature",
                        role="feature", parent="gate"))
    return out


def laws_of(document: Mapping[str, Any]) -> List[str]:
    return [str(t.get("name")) for t in document.get("terms") or ()]


def all_nodes(document: Mapping[str, Any]) -> List[Node]:
    return [n for law in laws_of(document) for n in law_nodes(document, law)]


DIGITS = 6


def round_constant(value: float) -> float:
    if value == 0.0 or not math.isfinite(value):
        return value
    return round(value, DIGITS - 1 - int(math.floor(math.log10(abs(value)))))


def literal(value: float) -> sp.Expr:
    value = round_constant(float(value))
    return (sp.Integer(int(value)) if float(value).is_integer()
            and abs(value) < 1e15 else sp.Float(value))


def clean(expr: Any) -> Any:
    if isinstance(expr, sp.Pow):
        base, k = expr.args
        if getattr(k, "is_Float", False):
            value = float(k)
            for exact in (round(value), round(2 * value) / 2):
                if abs(value - exact) < 1e-9:
                    k = sp.Rational(int(2 * exact), 2)
                    break
        return sp.Pow(clean(base), k if k.is_number else clean(k))
    if getattr(expr, "is_Float", False):
        rounded = round_constant(float(expr))
        return expr if rounded == float(expr) else sp.Float(rounded)
    if not getattr(expr, "args", ()) or isinstance(expr, sp.Symbol):
        return expr
    args = tuple(clean(arg) for arg in expr.args)
    return expr if args == expr.args else expr.func(*args)


_FOREIGN = (sp.re, sp.im, sp.arg, sp.conjugate, sp.sign)


def broken(expr: Any, values: Mapping[str, Optional[float]]) -> str:
    if expr is None:
        return "nothing was built"
    bad = (sp.zoo, sp.nan, sp.oo, -sp.oo)
    if getattr(expr, "has", None) is None:
        return ""
    if expr.has(*bad):
        return "the tree holds an infinity or a NaN"
    if expr.has(*_FOREIGN):
        return "the tree holds re/im/sign, which no evaluator reads"
    if any(not is_tame(n) for n in expr.atoms(sp.Number)):
        return "a literal is too large to print or compile"
    bound = {sp.Symbol(k): v for k, v in values.items() if v is not None}
    try:
        if bound and expr.subs(bound).has(*bad):
            return "with its parameters bound, the tree divides by zero"
    except (TypeError, ValueError, ZeroDivisionError):
        return "the tree does not bind its parameters"
    return ""


__all__ = ["FUNCTIONS", "GATE", "Node", "T1", "T2", "T3", "T4", "T5",
           "TARGETS", "all_nodes", "at", "broken", "classify", "clean",
           "is_input", "law_nodes", "laws_of", "literal", "nodes_of",
           "parent_of", "path_of", "path_text", "reads_input", "replace_at",
           "round_constant", "walk"]
