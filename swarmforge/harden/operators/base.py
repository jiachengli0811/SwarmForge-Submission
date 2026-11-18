from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

from sympy.polys.polyerrors import BasePolynomialError

from ...solve.config import FamilyConfig
from ..deadline import APPLY_SECONDS, Overtime, deadline
from ..document import Document, interface_change
from ..domain import Domain

SYNTAX, PATTERN = "syntax", "pattern"


@dataclass(frozen=True)
class Mutation:
    operator: str
    site: Optional[str] = None
    param: Optional[str] = None
    args: Mapping[str, Any] = field(default_factory=dict)

    @property
    def target(self) -> str:
        return self.site or self.param or "-"

    @property
    def law(self) -> Optional[str]:
        return None if not self.site else self.site.split("/")[0]

    def key(self) -> Tuple[Any, ...]:
        return (self.operator, self.site, self.param,
                tuple(sorted((k, _hashable(v)) for k, v in self.args.items())))

    def label(self) -> str:
        detail = ", ".join("%s=%s" % (k, _render(v))
                           for k, v in sorted(self.args.items()))
        return "%s(%s)%s" % (self.operator, self.target,
                             " [%s]" % detail if detail else "")

    def to_dict(self) -> Dict[str, Any]:
        return {"operator": self.operator, "site": self.site,
                "param": self.param, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Mutation":
        return cls(str(raw["operator"]), raw.get("site"), raw.get("param"),
                   dict(raw.get("args") or {}))


def _hashable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value


def _render(value: Any) -> str:
    return "%g" % value if isinstance(value, float) else str(value)


@dataclass(frozen=True)
class OperatorContext:
    config: FamilyConfig
    domains: Mapping[str, Domain]
    original: Document
    reference: Any = None
    scale: Any = None
    cache: Dict[str, Any] = field(default_factory=dict, compare=False,
                                  repr=False)


@dataclass(frozen=True)
class Intent:
    changes: Tuple[str, ...] = ()
    keeps: Tuple[str, ...] = ()
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"changes": list(self.changes), "keeps": list(self.keeps),
                "source": self.source}


class Operator:
    name = ""
    family = ""
    target = ""
    action = ""
    intent: Optional[Intent] = None
    description = ""

    def recipes(self, document: Document,
                context: OperatorContext) -> List[Mutation]:
        return []

    def apply(self, document: Document, mutation: Mutation,
              context: OperatorContext) -> Optional[Document]:
        raise NotImplementedError

    def direction(self, mutation: Mutation) -> int:
        return 0

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "family": self.family,
                "target": self.target, "action": self.action,
                "intent": None if self.intent is None else
                self.intent.to_dict(), "description": self.description}


OPERATORS: Dict[str, Operator] = {}


def register(operator: Operator, *, replace: bool = False) -> Operator:
    if not operator.name:
        raise ValueError("an operator needs a name")
    if operator.name in OPERATORS and not replace:
        raise ValueError("operator %r is already registered" % operator.name)
    OPERATORS[operator.name] = operator
    return operator


def unregister(name: str) -> None:
    OPERATORS.pop(name, None)


def operator(name: str) -> Operator:
    found = OPERATORS.get(name)
    if found is None:
        raise KeyError("no operator %r is registered (have: %s)"
                       % (name, ", ".join(sorted(OPERATORS)) or "none"))
    return found


def recipes(document: Document, context: OperatorContext,
            names: Optional[Sequence[str]] = None) -> List[Mutation]:
    out: List[Mutation] = []
    for name in (names if names is not None else sorted(OPERATORS)):
        out.extend(operator(name).recipes(document, context))
    return out


def apply(mutation: Mutation, document: Document,
          context: OperatorContext) -> Tuple[Optional[Document], str]:
    try:
        found = operator(mutation.operator)
    except KeyError as error:
        return None, str(error.args[0])
    try:
        with deadline(APPLY_SECONDS, mutation.label()):
            mutated = found.apply(document, mutation, context)
    except (ArithmeticError, ValueError, TypeError, KeyError, IndexError,
            AttributeError, BasePolynomialError, NotImplementedError,
            RecursionError, Overtime) as error:
        return None, "%s failed: %s: %s" % (mutation.label(),
                                            type(error).__name__, error)
    if mutated is None:
        return None, "%s does not fire here" % mutation.label()
    problem = interface_change(context.original, mutated)
    if problem is not None:
        return None, "interface changed: %s" % problem
    return mutated, ""


def mutation_laws(mutation: Mutation, document: Document) -> Tuple[str, ...]:
    if mutation.site:
        return (mutation.site.split("/")[0],)
    if not mutation.param:
        return ()
    from ..document import params
    places = params(document)
    found = {location.split("/")[0]
             for name in mutation.param.split(",")
             for location, _spec in places.get(name, ())
             if "/" in location}
    return tuple(sorted(found))


def counts(mutations: Iterable[Mutation]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for mutation in mutations:
        out[mutation.operator] = out.get(mutation.operator, 0) + 1
    return out


__all__ = ["Intent", "Mutation", "OPERATORS", "Operator", "OperatorContext",
           "PATTERN", "SYNTAX", "apply", "counts",
           "mutation_laws", "operator", "recipes", "register", "unregister"]
