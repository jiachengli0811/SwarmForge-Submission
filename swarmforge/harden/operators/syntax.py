from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanAtom, BooleanFunction

from ...solve.record import _threshold_feature
from ..document import (Document, Site, gate_of, read_site, set_gate,
                        write_site)
from .base import SYNTAX, Mutation, Operator, OperatorContext
from .calibrate import Calibration
from .tree import (GATE, T1, T2, T3, T4, T5, Node, at, broken, clean,
                   law_nodes, laws_of, literal, path_of, path_text,
                   reads_input, replace_at, walk)

Path = Tuple[int, ...]
Edit = Tuple[Path, Any]

DISTANCES = ("dist", "clearance")
SPEEDS = ("rel_speed", "shill_speed")

SCALES = (0.5, 0.8, 1.25, 2.0, -1.0)
OFFSETS = (-0.5, -0.1, 0.1, 0.5)
GAINS = (0.5, 2.0)
UNARY = ("neg", "abs", "sqrt", "square", "reciprocal")
BINARY = ("add", "sub", "mul", "div")
PARTNERS = 2
PARTNER_NODES = 12
FUNCTIONS = ("exp", "log", "tanh", "atan")
LEVELS = (0.25, 0.5, 0.75)
CUT_FACTORS = (0, 0.5, 2)
RELATIONS = {"<": sp.StrictLessThan, "<=": sp.LessThan,
             ">": sp.StrictGreaterThan, ">=": sp.GreaterThan}


def numeric(node: Node) -> bool:
    if node.expr is None or node.role == "exponent" or node.target == "":
        return False
    return isinstance(node.expr, sp.Expr) and not isinstance(
        node.expr, (Relational, BooleanFunction, BooleanAtom))


def zero(node: Node) -> bool:
    return numeric(node) and node.expr.is_number and node.expr == 0


def dynamic(node: Node) -> bool:
    return numeric(node) and node.dynamic and node.slot != "weight"


def vocabulary(document: Document, law: str) -> Tuple[str, ...]:
    term = next(t for t in document["terms"] if t["name"] == law)
    if term.get("direction") == "shill":
        return ("dist", "rel_speed", "shill_speed")
    return ("dist", "rel_speed")


def units(name: str) -> str:
    return "length" if name in DISTANCES else "speed"


def _size(cal: Calibration, node: Node, expr: Any = None) -> float:
    found = cal.size(node.law, node.slot, node.expr if expr is None else expr)
    return found if found is not None else 1.0


def _unary(kind: str, q: Any) -> Any:
    return {"neg": lambda: -q, "abs": lambda: sp.Abs(q),
            "sqrt": lambda: sp.sqrt(q), "square": lambda: q ** 2,
            "reciprocal": lambda: 1 / q}[kind]()


def _function(name: str, q: Any, scale: float) -> Any:
    s = literal(scale)
    if name == "log":
        return s * sp.log(1 + sp.Abs(q) / s)
    return s * {"exp": sp.exp, "tanh": sp.tanh, "atan": sp.atan}[name](q / s)


def _operand_of(node: Node) -> Any:
    expr = node.expr
    if node.kind == "neg":
        return expr.args[1]
    return expr.args[0]


class SyntaxOperator(Operator):
    family = SYNTAX

    def eligible(self, node: Node) -> bool:
        raise NotImplementedError

    def variations(self, node: Node, document: Document
                   ) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError

    def rewrite(self, root: Any, node: Node, args: Dict[str, Any],
                cal: Calibration) -> Optional[Edit]:
        raise NotImplementedError

    def gate_edit(self, document: Document, law: str, args: Dict[str, Any],
                  cal: Calibration) -> Optional[Document]:
        return None


    def recipes(self, document: Document,
                context: OperatorContext) -> List[Mutation]:
        out = []
        for law in laws_of(document):
            for node in law_nodes(document, law):
                if not self.eligible(node):
                    continue
                for args in self.variations(node, document):
                    out.append(Mutation(self.name, site=node.site,
                                        args=dict(args, path=node.where)))
        return out

    def apply(self, document: Document, mutation: Mutation,
              context: OperatorContext) -> Optional[Document]:
        law, slot = mutation.site.split("/")
        cal = Calibration.of(document, context)
        args = dict(mutation.args)
        if slot == GATE:
            return self.gate_edit(document, law, args, cal)
        site = Site(law, slot)
        root = read_site(document, site).expr
        path = path_of(args.pop("path", ""))
        node = next((n for n in law_nodes(document, law)
                     if n.slot == slot and n.path == path), None)
        if node is None or not self.eligible(node):
            return None
        edit = self.rewrite(root, node, args, cal)
        if edit is None:
            return None
        where, new = edit
        if new is None:
            return None
        tree = clean(replace_at(root, where, sp.sympify(new)))
        if tree == root:
            return None
        if slot == "weight" and reads_input(tree):
            return None
        problem = broken(tree, cal.namespace)
        if problem:
            raise ValueError(problem)
        return write_site(document, site, tree)


class VariableInsert(SyntaxOperator):
    name, target, action = "variable.insert", T1, "insert"
    description = "the node starts to read an input: N*u/r_u or N + s*u/r_u"

    def eligible(self, node):
        return numeric(node) and node.slot in ("magnitude", "threshold")

    def variations(self, node, document):
        forms = ("plus",) if zero(node) else ("times", "plus")
        for name in vocabulary(document, node.law):
            for form in forms:
                yield {"input": name, "form": form}

    def rewrite(self, root, node, args, cal):
        u = sp.Symbol(args["input"])
        unit = u / literal(cal.reference(node.law, args["input"], node.slot))
        if args["form"] == "times":
            return node.path, node.expr * unit
        return node.path, node.expr + literal(_size(cal, node)) * unit


class VariableDelete(SyntaxOperator):
    name, target, action = "variable.delete", T1, "delete"
    description = "the dependence on an input removed: v -> its reference"

    def eligible(self, node):
        return node.kind == "variable"

    def variations(self, node, document):
        yield {}

    def rewrite(self, root, node, args, cal):
        return node.path, literal(cal.reference(node.law, node.expr.name,
                                                node.slot))


class VariableReplace(SyntaxOperator):
    name, target, action = "variable.replace", T1, "replace"
    description = ("one input for another -- in a tree or as the gate's "
                   "input -- rescaled across units")

    def eligible(self, node):
        return node.kind in ("variable", "feature")

    def variations(self, node, document):
        was = (_feature(document, node.law) if node.kind == "feature"
               else node.expr.name)
        for name in vocabulary(document, node.law) + ("clearance",):
            if name != was:
                yield {"to": name}

    def rewrite(self, root, node, args, cal):
        was, now = node.expr.name, args["to"]
        new = sp.Symbol(now)
        if units(was) != units(now):
            new = new * literal(cal.reference(node.law, was, node.slot)
                                / cal.reference(node.law, now, node.slot))
        return node.path, new

    def gate_edit(self, document, law, args, cal):
        gate = gate_of(document, law)
        if not gate.gated:
            return None
        was, now = _feature(document, law), args["to"]
        threshold = _compared(document, law)
        if units(was) != units(now):
            threshold = threshold * literal(
                cal.reference(law, now, GATE) / cal.reference(law, was, GATE))
        return set_gate(document, law, gate.kind, now, clean(threshold),
                        gate.params)


def _feature(document: Document, law: str) -> Optional[str]:
    gate = gate_of(document, law)
    if not gate.gated:
        return None
    return gate.feature or _threshold_feature(gate.threshold)[1]


def _compared(document: Document, law: str) -> Any:
    gate = gate_of(document, law)
    if gate.feature:
        return gate.threshold
    return _threshold_feature(gate.threshold)[0]


class ConstantInsert(SyntaxOperator):
    name, target, action = "constant.insert", T2, "insert"
    description = "an offset N + c*s, or a gain k*N, inserted at a node"

    def eligible(self, node):
        return numeric(node)

    def variations(self, node, document):
        for c in OFFSETS:
            yield {"form": "plus", "c": c}
        if zero(node):
            return
        for k in GAINS:
            yield {"form": "times", "k": k}

    def rewrite(self, root, node, args, cal):
        if args["form"] == "times":
            return node.path, literal(args["k"]) * node.expr
        return node.path, node.expr + literal(args["c"] * _size(cal, node))


def _drop(root: Any, path: Path) -> Optional[Edit]:
    if not path:
        return None
    parent_path, index = path[:-1], path[-1]
    parent = at(root, parent_path)
    if isinstance(parent, sp.Mul) and len(parent.args) == 2 \
            and parent.args[0] == -1 and parent_path:
        return _drop(root, parent_path)
    if not isinstance(parent, (sp.Add, sp.Mul)) or len(parent.args) < 2:
        return None
    rest = [a for i, a in enumerate(parent.args) if i != index]
    return parent_path, parent.func(*rest)


class ConstantDelete(SyntaxOperator):
    name, target, action = "constant.delete", T2, "delete"
    description = "a constant operand of a sum or a product dropped"

    def eligible(self, node):
        return node.kind in ("number", "parameter") and node.role == "operand"

    def variations(self, node, document):
        yield {}

    def rewrite(self, root, node, args, cal):
        return _drop(root, node.path)


class ConstantReplace(SyntaxOperator):
    name, target, action = "constant.replace", T2, "replace"
    description = ("a constant scaled (its sign flipped among the factors); "
                   "an exponent stepped")

    def eligible(self, node):
        return node.kind in ("number", "parameter", "exponent")

    def variations(self, node, document):
        if node.kind == "exponent":
            k = sp.nsimplify(node.expr)
            steps = []
            for new in (k - 1, k + 1, 2 * k, k / 2):
                if new != 0 and new != k and abs(new) <= 8 \
                        and new not in steps:
                    steps.append(new)
                    yield {"to": str(new)}
            return
        if node.kind == "number" and node.expr == 0:
            return
        for k in SCALES:
            yield {"k": k}

    def rewrite(self, root, node, args, cal):
        if node.kind == "exponent":
            return node.path, sp.Rational(args["to"])
        if node.kind == "number":
            return node.path, literal(args["k"] * float(node.expr))
        return node.path, literal(args["k"]) * node.expr


def _partners(node: Node, nodes: List[Node]) -> List[Node]:
    out = []
    for other in nodes:
        if other.slot != node.slot or other.path == node.path:
            continue
        if not dynamic(other) or other.expr == node.expr:
            continue
        if sum(1 for _ in walk(other.expr)) > PARTNER_NODES:
            continue
        out.append(other)
    out.sort(key=lambda n: (sum(1 for _ in walk(n.expr)), n.path))
    return out[:PARTNERS]


class ArithmeticInsert(SyntaxOperator):
    name, target, action = "arithmetic.insert", T3, "insert"
    description = ("a unary operator wrapped round a node; the node combined "
                   "with another subtree of its site by + - * /")

    def eligible(self, node):
        return dynamic(node)

    def recipes(self, document, context):
        out = []
        for law in laws_of(document):
            nodes = law_nodes(document, law)
            for node in nodes:
                if not self.eligible(node):
                    continue
                for kind in UNARY:
                    out.append(Mutation(self.name, site=node.site, args={
                        "path": node.where, "unary": kind}))
                for other in _partners(node, nodes):
                    for op in BINARY:
                        out.append(Mutation(self.name, site=node.site, args={
                            "path": node.where, "binary": op,
                            "with": other.where}))
        return out

    def rewrite(self, root, node, args, cal):
        n = node.expr
        if "unary" in args:
            new = _unary(args["unary"], n)
            if args["unary"] in ("neg", "abs"):
                return node.path, new
            return node.path, cal.match(node.law, node.slot, n, new)
        m = at(root, path_of(args["with"]))
        new = {"add": n + m, "sub": n - m, "mul": n * m,
               "div": n / m}[args["binary"]]
        return node.path, cal.match(node.law, node.slot, n, new)


class ArithmeticDelete(SyntaxOperator):
    name, target, action = "arithmetic.delete", T3, "delete"
    description = ("an operand of a sum or a product dropped; a unary "
                   "operator unwrapped")

    def eligible(self, node):
        return node.target == T3

    def variations(self, node, document):
        if node.kind in ("add", "mul"):
            for j in range(len(node.expr.args)):
                yield {"drop": j}
        else:
            yield {"unwrap": True}

    def rewrite(self, root, node, args, cal):
        n = node.expr
        if "drop" in args:
            rest = [a for i, a in enumerate(n.args) if i != args["drop"]]
            if not rest:
                return None
            new = n.func(*rest)
            return node.path, cal.match(node.law, node.slot, n, new)
        inner = _operand_of(node)
        if node.kind in ("neg", "abs"):
            return node.path, inner
        return node.path, cal.match(node.law, node.slot, n, inner)


class ArithmeticReplace(SyntaxOperator):
    name, target, action = "arithmetic.replace", T3, "replace"
    description = ("how one operand joins its sum or product changed among "
                   "+ - * /; one unary operator for another")

    def eligible(self, node):
        return node.target == T3

    def variations(self, node, document):
        n = node.expr
        if node.kind == "add":
            for j in range(len(n.args)):
                for op in ("sub", "mul", "div"):
                    yield {"operand": j, "op": op}
        elif node.kind == "mul":
            for j, arg in enumerate(n.args):
                if arg == -1:
                    continue
                for op in ("div", "add", "sub"):
                    yield {"operand": j, "op": op}
        elif node.kind == "power" and not n.args[1].is_number:
            for op in ("mul", "div"):
                yield {"op": op}
        else:
            current = node.kind if node.kind in UNARY else ""
            for kind in UNARY:
                if kind != current:
                    yield {"unary": kind}

    def rewrite(self, root, node, args, cal):
        n = node.expr
        if "unary" in args:
            new = _unary(args["unary"], _operand_of(node))
            if args["unary"] == "neg":
                return node.path, new
            return node.path, cal.match(node.law, node.slot, n, new)
        if "operand" not in args:
            base, exponent = n.args
            new = base * exponent if args["op"] == "mul" else base / exponent
            return node.path, cal.match(node.law, node.slot, n, new)
        j, op = args["operand"], args["op"]
        operand = n.args[j]
        rest = n.func(*[a for i, a in enumerate(n.args) if i != j])
        if node.kind == "add":
            if op == "sub":
                return node.path, rest - operand
            new = rest * operand if op == "mul" else rest / operand
        else:
            if op == "div":
                return node.path, rest / operand
            new = rest + operand if op == "add" else rest - operand
        return node.path, cal.match(node.law, node.slot, n, new)


class FunctionInsert(SyntaxOperator):
    name, target, action = "function.insert", T4, "insert"
    description = ("s*f(N/s) for exp, tanh, atan, s*log(1 + |N|/s); Min or "
                   "Max of N and a level it passes")

    def eligible(self, node):
        return dynamic(node)

    def variations(self, node, document):
        for name in FUNCTIONS:
            yield {"function": name}
        for name in ("min", "max"):
            for q in LEVELS:
                yield {"function": name, "q": q}

    def rewrite(self, root, node, args, cal):
        n, name = node.expr, args["function"]
        if name in ("min", "max"):
            level = cal.quantile(node.law, node.slot, n, args["q"])
            if level is None:
                return None
            return node.path, (sp.Min if name == "min" else sp.Max)(
                n, literal(level))
        new = _function(name, n, _size(cal, node))
        return node.path, cal.match(node.law, node.slot, n, new)


class FunctionDelete(SyntaxOperator):
    name, target, action = "function.delete", T4, "delete"
    description = "a function unwrapped; a Min or Max reduced to one argument"

    def eligible(self, node):
        return node.target == T4

    def variations(self, node, document):
        if node.kind in ("min", "max"):
            for j in range(len(node.expr.args)):
                yield {"keep": j}
        else:
            yield {"unwrap": True}

    def rewrite(self, root, node, args, cal):
        n = node.expr
        if "keep" in args:
            return node.path, n.args[args["keep"]]
        return node.path, cal.match(node.law, node.slot, n, n.args[0])


class FunctionReplace(SyntaxOperator):
    name, target, action = "function.replace", T4, "replace"
    description = "one function for another over the same argument; Min/Max"

    def eligible(self, node):
        return node.target == T4

    def variations(self, node, document):
        if node.kind in ("min", "max"):
            yield {"function": "max" if node.kind == "min" else "min"}
            return
        for name in FUNCTIONS:
            if name != node.kind:
                yield {"function": name}

    def rewrite(self, root, node, args, cal):
        n, name = node.expr, args["function"]
        if name in ("min", "max"):
            return node.path, (sp.Min if name == "min" else sp.Max)(*n.args)
        inner = n.args[0]
        new = _function(name, inner, _size(cal, node, inner))
        return node.path, cal.match(node.law, node.slot, n, new)


def _flip(kind: str) -> str:
    return "above" if kind == "below" else "below"


class PredicateInsert(SyntaxOperator):
    name, target, action = "predicate.insert", T5, "insert"
    description = ("a gate 1{d <= t} or 1{d > t} on an ungated law; a cut "
                   "Piecewise((N, u <= t), (k*N, True)) in a magnitude")

    def eligible(self, node):
        return node.kind == "gate:always" or (
            node.slot == "magnitude" and node.path == () and numeric(node))

    def variations(self, node, document):
        if node.slot == GATE:
            for kind in ("below", "above"):
                for q in LEVELS:
                    yield {"gate": kind, "q": q}
            return
        for name in vocabulary(document, node.law):
            for relation in ("<=", ">"):
                for q in LEVELS:
                    for k in CUT_FACTORS:
                        yield {"input": name, "relation": relation, "q": q,
                               "factor": k}

    def rewrite(self, root, node, args, cal):
        name = args["input"]
        level = literal(cal.level(node.law, name, args["q"], node.slot))
        condition = RELATIONS[args["relation"]](sp.Symbol(name), level)
        return node.path, sp.Piecewise(
            (node.expr, condition),
            (sp.nsimplify(args["factor"]) * node.expr, True))

    def gate_edit(self, document, law, args, cal):
        if gate_of(document, law).gated:
            return None
        feature = cal.semantic(law, GATE, "dist")
        level = literal(cal.level(law, feature, args["q"], GATE,
                                  extent=True))
        return set_gate(document, law, args["gate"], feature, level)


class PredicateDelete(SyntaxOperator):
    name, target, action = "predicate.delete", T5, "delete"
    description = ("the gate removed; a Piecewise reduced to one branch; an "
                   "operand of And/Or dropped; a Not or an ITE unwrapped")

    def eligible(self, node):
        return (node.target == T5 and node.kind != "relation"
                and node.kind != "gate:always")

    def variations(self, node, document):
        n = node.expr
        if node.slot == GATE:
            yield {"gate": "remove"}
        elif node.kind == "piecewise":
            for j in range(len(n.args)):
                yield {"keep": j}
        elif node.kind == "ite":
            for j in (1, 2):
                yield {"keep": j}
        elif isinstance(n, sp.Not):
            yield {"unwrap": True}
        elif len(n.args) >= 2:
            for j in range(len(n.args)):
                yield {"drop": j}

    def rewrite(self, root, node, args, cal):
        n = node.expr
        if node.kind == "piecewise":
            return node.path, n.args[args["keep"]].expr
        if node.kind == "ite":
            return node.path, n.args[args["keep"]]
        if "unwrap" in args:
            return node.path, n.args[0]
        rest = [a for i, a in enumerate(n.args) if i != args["drop"]]
        return node.path, n.func(*rest)

    def gate_edit(self, document, law, args, cal):
        if not gate_of(document, law).gated:
            return None
        return set_gate(document, law, "always")


class PredicateReplace(SyntaxOperator):
    name, target, action = "predicate.replace", T5, "replace"
    description = ("the gate's direction flipped; a relation's operator "
                   "changed; And for Or; two Piecewise branches swapped")

    def eligible(self, node):
        return node.target == T5 and node.kind != "gate:always"

    def variations(self, node, document):
        n = node.expr
        if node.slot == GATE:
            yield {"gate": "flip"}
        elif node.kind == "relation":
            current = next((k for k, v in RELATIONS.items()
                            if isinstance(n, v)), None)
            for name in RELATIONS:
                if name != current:
                    yield {"relation": name}
        elif node.kind == "connective" and not isinstance(n, sp.Not):
            yield {"connective": "Or" if isinstance(n, sp.And) else "And"}
        elif node.kind == "piecewise":
            for j in range(len(n.args) - 1):
                yield {"swap": j}
        elif node.kind == "ite":
            yield {"swap": 1}

    def rewrite(self, root, node, args, cal):
        n = node.expr
        if "relation" in args:
            return node.path, RELATIONS[args["relation"]](*n.args)
        if "connective" in args:
            return node.path, (sp.Or if args["connective"] == "Or"
                               else sp.And)(*n.args)
        if node.kind == "ite":
            return node.path, sp.ITE(n.args[0], n.args[2], n.args[1])
        j = args["swap"]
        exprs = [pair.expr for pair in n.args]
        exprs[j], exprs[j + 1] = exprs[j + 1], exprs[j]
        return node.path, sp.Piecewise(*[(e, pair.cond) for e, pair
                                         in zip(exprs, n.args)])

    def gate_edit(self, document, law, args, cal):
        gate = gate_of(document, law)
        if not gate.gated:
            return None
        return set_gate(document, law, _flip(gate.kind), gate.feature,
                        gate.threshold, gate.params)


SYNTAX_OPERATORS: Tuple[SyntaxOperator, ...] = (
    VariableInsert(), VariableDelete(), VariableReplace(),
    ConstantInsert(), ConstantDelete(), ConstantReplace(),
    ArithmeticInsert(), ArithmeticDelete(), ArithmeticReplace(),
    FunctionInsert(), FunctionDelete(), FunctionReplace(),
    PredicateInsert(), PredicateDelete(), PredicateReplace(),
)


__all__ = ["SYNTAX_OPERATORS", "SyntaxOperator", "dynamic", "numeric",
           "vocabulary", "zero"]
