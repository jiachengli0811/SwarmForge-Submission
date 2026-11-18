from typing import Dict, List

from .base import (OPERATORS, PATTERN, SYNTAX, Intent, Mutation, Operator,
                   OperatorContext, apply, counts,
                   mutation_laws, operator, recipes, register, unregister)
from .calibrate import Calibration
from .patterns import PATTERNS, Pattern
from .syntax import SYNTAX_OPERATORS, SyntaxOperator
from .tree import TARGETS, Node, all_nodes, law_nodes

for _op in SYNTAX_OPERATORS + PATTERNS:
    register(_op, replace=True)


def names(family: str = "") -> List[str]:
    return [name for name, op in sorted(OPERATORS.items())
            if not family or op.family == family]


def catalogue() -> List[Dict[str, object]]:
    return [op.describe() for _name, op in sorted(OPERATORS.items())]


__all__ = ["Calibration", "Intent", "Mutation", "Node", "OPERATORS",
           "Operator", "OperatorContext", "PATTERN", "PATTERNS", "Pattern",
           "SYNTAX", "SYNTAX_OPERATORS", "SyntaxOperator", "TARGETS",
           "all_nodes", "apply", "catalogue", "counts",
           "law_nodes", "mutation_laws", "names", "operator", "recipes",
           "register", "unregister"]
