from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import sympy as sp

from ..solve.record import INPUTS, Record
from .document import Site, sites


@dataclass(frozen=True)
class Unit:
    site: Site
    flown: sp.Expr
    gate: Optional[Tuple[str, str]] = None

    @property
    def name(self) -> str:
        return self.site.name

    @property
    def law(self) -> str:
        return self.site.law

    @property
    def slot(self) -> str:
        return self.site.slot

    @property
    def inputs(self) -> Tuple[str, ...]:
        names = {symbol.name for symbol in self.flown.free_symbols}
        return tuple(name for name in INPUTS if name in names)

    def text(self) -> str:
        return str(self.flown)


def units_of(document: Mapping, record: Record) -> Dict[str, Unit]:
    terms = {term.name: term for term in record.terms}
    out: Dict[str, Unit] = {}
    for site in sites(document):
        term = terms.get(site.law)
        if term is None:
            raise ValueError("the parsed record has no law %r" % site.law)
        gate = None
        if site.slot == "magnitude":
            flown = term.magnitude
        elif site.slot == "threshold":
            if term.gate is None:
                raise ValueError("law %r writes a threshold but reads as "
                                 "ungated" % site.law)
            flown = term.gate.threshold
            gate = (term.gate.kind, term.gate.feature)
        else:
            flown = sp.Float(term.weight)
        out[site.name] = Unit(site, flown, gate)
    return out


def expressions(units: Mapping[str, Unit]) -> Dict[str, str]:
    return {name: unit.text() for name, unit in units.items()}


__all__ = ["Unit", "expressions", "units_of"]
