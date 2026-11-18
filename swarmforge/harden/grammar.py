from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Tuple

import sympy as sp
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.logic.boolalg import BooleanFalse, BooleanTrue

from ..solve.record import INPUTS

INPUT_NAMES: Tuple[str, ...] = tuple(INPUTS)

UNARY = (sp.exp, sp.log, sp.tanh, sp.atan, sp.sinh, sp.Abs, sp.cos)
NARY = (sp.Min, sp.Max)

_RELATIONS = (sp.StrictLessThan, sp.LessThan, sp.StrictGreaterThan,
              sp.GreaterThan, sp.Equality, sp.Unequality)
_CONNECTIVES = (sp.And, sp.Or, sp.Not, sp.ITE)

MAX_NODES = 600
MAX_DEPTH = 48

NESTED = (sp.tanh, sp.atan, sp.sinh)
NESTING_LIMIT = 3

MAX_NUMBER_BITS = 256
MAX_FLOAT_BINARY_EXP = 512


class Kind(Enum):
    INPUT = "input"
    CONST = "const"
    EXPONENT = "exponent"
    OP = "op"
    FUNC = "func"
    PRED = "pred"
    BRANCH = "branch"


class Type(Enum):
    NUM = "num"
    BOOL = "bool"


def is_tame(value: Any) -> bool:
    if value in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        return False
    if isinstance(value, sp.Float):
        _sign, man, exp, bc = value._mpf_
        if man == 0:
            return True
        return abs(exp) <= MAX_FLOAT_BINARY_EXP and bc <= MAX_NUMBER_BITS
    if isinstance(value, sp.Integer):
        return abs(int(value)).bit_length() <= MAX_NUMBER_BITS
    if isinstance(value, sp.Rational):
        return (abs(int(value.p)).bit_length() <= MAX_NUMBER_BITS
                and abs(int(value.q)).bit_length() <= MAX_NUMBER_BITS)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return number == number and abs(number) != float("inf")


def classify(node: Any, parent: Any = None,
             index: Optional[int] = None) -> Tuple[Kind, Type]:
    if isinstance(parent, sp.Pow) and index == 1:
        return Kind.EXPONENT, Type.NUM
    if isinstance(node, (BooleanTrue, BooleanFalse)):
        return Kind.PRED, Type.BOOL
    if isinstance(node, _RELATIONS) or isinstance(node, _CONNECTIVES):
        return Kind.PRED, Type.BOOL
    if isinstance(node, (sp.Piecewise, ExprCondPair)):
        return Kind.BRANCH, Type.NUM
    if isinstance(node, sp.Symbol):
        return (Kind.INPUT if node.name in INPUT_NAMES else Kind.CONST), Type.NUM
    if getattr(node, "is_Number", False):
        return Kind.CONST, Type.NUM
    if isinstance(node, (sp.Add, sp.Mul, sp.Pow)):
        return Kind.OP, Type.NUM
    if getattr(node, "func", None) in UNARY + NARY:
        return Kind.FUNC, Type.NUM
    raise ValueError("node %r (%s) is outside the grammar"
                     % (node, type(node).__name__))


def size(expr: Any) -> Tuple[int, int]:
    nodes, deepest = 0, 0
    stack = [(expr, 1)]
    while stack:
        node, depth = stack.pop()
        nodes += 1
        deepest = max(deepest, depth)
        stack.extend((child, depth + 1) for child in node.args)
    return nodes, deepest


def nesting(expr: Any) -> int:
    deepest = 0
    stack = [(expr, 0)]
    while stack:
        node, count = stack.pop()
        if getattr(node, "func", None) in NESTED:
            count += 1
        deepest = max(deepest, count)
        stack.extend((child, count) for child in node.args)
    return deepest


def typecheck(expr: Any) -> Optional[str]:
    if expr is None:
        return None
    nodes, depth = size(expr)
    if nodes > MAX_NODES or depth > MAX_DEPTH:
        return ("the tree has %d nodes and is %d deep, over the limits of %d "
                "and %d" % (nodes, depth, MAX_NODES, MAX_DEPTH))
    nested = nesting(expr)
    if nested > NESTING_LIMIT:
        return ("%d nested hyperbolic or inverse-trigonometric functions on "
                "one path, over the limit of %d" % (nested, NESTING_LIMIT))
    return _check(expr, None, None)


def _check(expr: Any, parent: Any, index: Optional[int]) -> Optional[str]:
    try:
        kind, _type = classify(expr, parent, index)
    except ValueError as error:
        return str(error)

    if kind is Kind.EXPONENT:
        if getattr(expr, "is_Number", False):
            return None if is_tame(expr) else (
                "exponent %s is too large to print or compile" % expr)
        return _expect(expr, Type.NUM, parent, index, "an exponent",
                       already=True)
    if kind is Kind.CONST and getattr(expr, "is_Number", False):
        return None if is_tame(expr) else (
            "constant %s is too large to print or compile" % expr)
    if kind in (Kind.INPUT, Kind.CONST):
        return None

    if isinstance(expr, sp.Piecewise):
        if not expr.args:
            return "an empty Piecewise"
        for pair in expr.args:
            if not isinstance(pair, ExprCondPair):
                return ("Piecewise branch %r is not an (expression, "
                        "condition) pair" % (pair,))
        last = expr.args[-1].cond
        if not isinstance(last, BooleanTrue):
            return ("Piecewise is not total: its last condition is %s rather "
                    "than True, so some state may reach no branch" % last)
        for pair in expr.args:
            problem = _check(pair, expr, None)
            if problem is not None:
                return problem
        return None
    if isinstance(expr, ExprCondPair):
        value, condition = expr.args
        problem = _expect(value, Type.NUM, expr, 0, "a Piecewise branch")
        if problem is not None:
            return problem
        return _expect(condition, Type.BOOL, expr, 1, "a Piecewise condition")

    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        problem = _expect(base, Type.NUM, expr, 0, "a power's base")
        if problem is not None:
            return problem
        return _check(exponent, expr, 1)
    if isinstance(expr, (sp.Add, sp.Mul)) or getattr(expr, "func",
                                                     None) in NARY:
        for position, argument in enumerate(expr.args):
            problem = _expect(argument, Type.NUM, expr, position,
                              "an operand of %s" % expr.func.__name__)
            if problem is not None:
                return problem
        return None
    if getattr(expr, "func", None) in UNARY:
        if len(expr.args) != 1:
            return ("%s takes one argument, got %d"
                    % (expr.func.__name__, len(expr.args)))
        return _expect(expr.args[0], Type.NUM, expr, 0,
                       "the argument of %s" % expr.func.__name__)

    if isinstance(expr, (BooleanTrue, BooleanFalse)):
        return None
    if isinstance(expr, _RELATIONS):
        for position, side in enumerate(expr.args):
            problem = _expect(side, Type.NUM, expr, position,
                              "a side of a relation")
            if problem is not None:
                return problem
        return None
    if isinstance(expr, _CONNECTIVES):
        arity = {sp.Not: 1, sp.ITE: 3}.get(expr.func)
        if arity is not None and len(expr.args) != arity:
            return ("%s takes %d arguments, got %d"
                    % (expr.func.__name__, arity, len(expr.args)))
        if isinstance(expr, sp.ITE):
            for inner in sp.preorder_traversal(expr):
                if inner is not expr and isinstance(inner, (sp.And, sp.Or)) \
                        and inner.has(sp.ITE):
                    return ("an ITE above %s(...) that contains an ITE "
                            "cannot be compiled" % inner.func.__name__)
        for position, argument in enumerate(expr.args):
            problem = _expect(argument, Type.BOOL, expr, position,
                              "an operand of %s" % expr.func.__name__)
            if problem is not None:
                return problem
        return None
    return "node %r (%s) is outside the grammar" % (expr, type(expr).__name__)


def _expect(node: Any, wanted: Type, parent: Any, index: int, where: str,
            already: bool = False) -> Optional[str]:
    if already:
        try:
            _kind, actual = classify(node)
        except ValueError as error:
            return str(error)
        if actual is not wanted:
            return "%s must be %s, but %s is %s" % (where, wanted.value,
                                                    node, actual.value)
        return _check(node, None, None)
    try:
        _kind, actual = classify(node, parent, index)
    except ValueError as error:
        return str(error)
    if actual is not wanted:
        return ("%s must be %s, but %s is %s"
                % (where, wanted.value, node, actual.value))
    return _check(node, parent, index)


__all__ = ["INPUT_NAMES", "MAX_DEPTH", "MAX_NODES", "NESTING_LIMIT", "Kind",
           "Type", "classify", "is_tame", "nesting",
           "size", "typecheck"]
