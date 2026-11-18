from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import sympy as sp

from ..solve.config import FamilyConfig
from ..solve.dynamics import Simulator
from ..solve.reach import Reach
from ..solve.record import Record, parse_record
from .domain import Domain
from .grammar import typecheck
from .singular import singular_point
from .sweep import GRID, SAMPLES, where
from .units import Unit, units_of

CHECKS = ("reads", "family", "grammar", "compiles", "speed", "real", "alive")

DEAD_RESPONSE = 1e-9


@dataclass(frozen=True)
class Validity:
    ok: bool
    check: str = ""
    reason: str = ""
    record: Optional[Record] = field(default=None, repr=False, compare=False)
    units: Mapping[str, Unit] = field(default_factory=dict, repr=False,
                                      compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"valid": self.ok, "check": self.check, "reason": self.reason}


def check(document: Mapping[str, Any], *, config: FamilyConfig,
          domains: Mapping[str, Domain], label: str = "variant") -> Validity:
    path = Path("%s.json" % label)
    try:
        record = parse_record(document, path, config.physics.record_params)
        units = units_of(document, record)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        return Validity(False, "reads", str(error))
    if record.target_key != config.name:
        return Validity(False, "family", "the record targets %r, the config "
                        "%r" % (record.target_key, config.name), record, units)

    for name, unit in units.items():
        problem = typecheck(unit.flown)
        if problem is not None:
            return Validity(False, "grammar", "%s: %s" % (name, problem),
                            record, units)

    try:
        simulator = Simulator(record, config)
    except (ValueError, KeyError, TypeError, SyntaxError, NameError) as error:
        return Validity(False, "compiles", str(error), record, units)
    try:
        Reach(config, 1).check_record(simulator.step_reach(), label)
    except ValueError as error:
        return Validity(False, "speed", str(error), record, units)

    terms = {term.name: term for term in record.terms}
    for name, unit in units.items():
        domain = domains.get(unit.law)
        if domain is None:
            return Validity(False, "reads", "law %r has no domain: it is not "
                            "a law of the original" % unit.law, record, units)
        gate = (terms[unit.law].gate if unit.slot == "magnitude" else None)
        try:
            found = _screen(unit, gate, domain)
        except (ValueError, TypeError, ArithmeticError) as error:
            return Validity(False, "real", "%s: %s" % (name, error), record,
                            units)
        if found is not None:
            kind, reason = found
            return Validity(False, kind, "%s %s" % (name, reason), record,
                            units)
    return Validity(True, "", "", record, units)


def _grid(axis: str, domain: Domain, count: int) -> Tuple[float, ...]:
    from .properties.response import axis_grid

    lo, hi = domain.span(axis) or domain.reach_span(axis)
    points = np.linspace(lo, hi, count)
    wide = domain.reach_span(axis)
    if wide is not None and wide != (lo, hi):
        points = np.union1d(points, axis_grid(axis, wide[0], wide[1], count))
    return tuple(float(x) for x in points)


def _gate_condition(gate) -> Any:
    feature = sp.Symbol(gate.feature)
    if gate.kind == "below":
        return sp.LessThan(feature, gate.threshold)
    return sp.StrictGreaterThan(feature, gate.threshold)


def _screen(unit: Unit, gate, domain: Domain
            ) -> Optional[Tuple[str, str]]:
    from ..solve.laws import Expression
    from ..solve.record import INPUTS

    reads = set(unit.inputs)
    if gate is not None:
        reads |= {gate.feature} | {s.name for s in
                                   gate.threshold.free_symbols}
    axes = tuple(name for name in INPUTS if name in reads)
    count = SAMPLES if len(axes) <= 1 else GRID
    grids = tuple(_grid(axis, domain, count) for axis in axes)
    if axes:
        mesh = np.meshgrid(*[np.asarray(g) for g in grids], indexing="ij")
        env = {a: m.ravel() for a, m in zip(axes, mesh)}
        shape = (int(mesh[0].size),)
    else:
        env, shape = {}, (1,)
    with np.errstate(all="ignore"):
        values = np.array(Expression(unit.flown)(env, shape), dtype=float)
        fires = np.ones(shape, dtype=bool)
        if gate is not None:
            threshold = Expression(gate.threshold)(env, shape)
            feature = env[gate.feature]
            fires = (feature <= threshold if gate.kind == "below"
                     else feature > threshold)
    bad = np.flatnonzero(fires & ~np.isfinite(values))
    if bad.size:
        point = [float(env[a][bad[0]]) for a in axes]
        return "real", "is not real and finite at %s" % (where(axes, point)
                                                         or "every point")
    guard = None if gate is None else _gate_condition(gate)
    between = singular_point(unit.flown, axes, grids, guard)
    if between is not None:
        return "real", ": %s" % between
    live = np.abs(values[fires & np.isfinite(values)])
    peak = float(np.max(live)) if live.size else 0.0
    if unit.slot == "magnitude" and peak < DEAD_RESPONSE:
        return "alive", ("responds with at most %.3g wherever it fires: the "
                         "law was deleted, not changed" % peak)
    if unit.slot == "weight" and peak == 0.0:
        return "alive", "is zero: the law was deleted, not changed"
    return None


__all__ = ["CHECKS", "DEAD_RESPONSE", "Validity", "check"]
