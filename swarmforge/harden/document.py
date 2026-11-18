from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import sympy as sp

SCHEMA = "swarmforge.algorithm.v1"

SLOTS: Tuple[str, ...] = ("magnitude", "threshold", "weight")

TERM_INTERFACE: Tuple[str, ...] = (
    "name", "role", "source", "signal", "direction", "aggregator",
    "measures", "anchor", "stochastic", "membership", "signal_speed",
)

DIRECTION_EDITS: Tuple[Tuple[str, str], ...] = (("away_unit", "around"),)

FIELD_DEFAULTS: Mapping[str, Any] = {
    "measures": "centre_to_centre", "anchor": "self", "stochastic": "none",
    "aggregator": "sum", "signal": "position", "role": "agent",
}

HEADER: Tuple[str, ...] = ("schema", "target_key", "roles", "base",
                           "arbitration")

Document = Dict[str, Any]


def strip(document: Mapping[str, Any]) -> Document:
    document = document.get("record", document)
    return {k: copy.deepcopy(v) for k, v in document.items()
            if not str(k).startswith("_")}


def load(path) -> Document:
    document = strip(json.loads(Path(path).read_text()))
    if document.get("schema") != SCHEMA:
        raise ValueError("%s is %r, not %r"
                         % (path, document.get("schema"), SCHEMA))
    return document


def canonical(document: Mapping[str, Any]) -> Document:
    out = strip(document)
    for term in out.get("terms") or ():
        for key, value in FIELD_DEFAULTS.items():
            term.setdefault(key, value)
        term.setdefault("signal_speed", None)
        if isinstance(term.get("activation"), dict):
            term["activation"].setdefault("feature", None)
    for raw in _expression_blocks(out):
        raw["e"] = sp.srepr(sp.sympify(raw["e"]))
    for _location, block in _param_blocks(out):
        for name in list(block):
            block[name] = Param.read(block[name]).to_raw()
    return out


def digest(document: Mapping[str, Any]) -> str:
    payload = json.dumps(strip(document), sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _expression_blocks(document: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for term in document.get("terms") or ():
        for slot in SLOTS:
            raw = _slot(term, slot)
            if raw:
                out.append(raw)
    return out


@dataclass(frozen=True)
class Site:
    law: str
    slot: str

    def __post_init__(self):
        if self.slot not in SLOTS:
            raise ValueError("slot %r (have: %s)" % (self.slot,
                                                     ", ".join(SLOTS)))

    @property
    def name(self) -> str:
        return "%s/%s" % (self.law, self.slot)

    @classmethod
    def parse(cls, name: str) -> "Site":
        law, _, slot = str(name).rpartition("/")
        if not law:
            raise ValueError("a site is '<law>/<slot>', got %r" % name)
        return cls(law, slot)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Param:
    value: Optional[float]
    tunable: bool
    bounds: Optional[Tuple[float, float]] = None

    @classmethod
    def read(cls, raw: Any) -> "Param":
        value, tunable = raw[0], bool(raw[1]) if len(raw) > 1 else False
        bounds = raw[2] if len(raw) > 2 else None
        return cls(None if value is None else float(value), tunable,
                   None if not bounds else (float(bounds[0]),
                                            float(bounds[1])))

    def to_raw(self) -> List[Any]:
        return [self.value, self.tunable,
                None if self.bounds is None else list(self.bounds)]


@dataclass(frozen=True)
class SiteExpression:
    site: Site
    expr: sp.Expr
    params: Mapping[str, Param]

    def substituted(self) -> sp.Expr:
        values = {sp.Symbol(name): spec.value
                  for name, spec in self.params.items()
                  if spec.value is not None}
        return self.expr.subs(values) if values else self.expr


def _term(document: Mapping[str, Any], law: str) -> Dict[str, Any]:
    for term in document.get("terms") or ():
        if str(term.get("name")) == law:
            return term
    raise KeyError("the record has no law %r" % law)


def _slot(term: Mapping[str, Any], slot: str) -> Optional[Dict[str, Any]]:
    if slot == "magnitude":
        return term.get("magnitude") or None
    if slot == "weight":
        return term.get("weight") or None
    activation = term.get("activation") or {}
    if (activation.get("kind") or "always") == "always":
        return None
    return activation.get("threshold") or None


def sites(document: Mapping[str, Any]) -> List[Site]:
    out = []
    for term in document.get("terms") or ():
        for slot in SLOTS:
            if _slot(term, slot):
                out.append(Site(str(term.get("name")), slot))
    return out


def read_site(document: Mapping[str, Any], site: Site) -> SiteExpression:
    raw = _slot(_term(document, site.law), site.slot)
    if not raw:
        raise KeyError("law %r carries no %s" % (site.law, site.slot))
    return SiteExpression(site, sp.sympify(raw["e"]),
                          {str(name): Param.read(spec)
                           for name, spec in (raw.get("p") or {}).items()})


def named_params(term: Mapping[str, Any]) -> Tuple[str, ...]:
    membership = term.get("membership") or {}
    names = [term.get("signal_speed")] + [membership.get(key)
                                          for key in ("range", "fov", "cap")]
    return tuple(str(n) for n in names if isinstance(n, str) and n)


def write_site(document: Mapping[str, Any], site: Site, expr: sp.Expr,
               params: Optional[Mapping[str, Param]] = None) -> Document:
    from ..solve.record import INPUTS

    out = strip(document)
    term = _term(out, site.law)
    raw = _slot(term, site.slot)
    if not raw:
        raise KeyError("law %r carries no %s to rewrite" % (site.law,
                                                            site.slot))
    given = dict(params or {})
    known = {str(name): Param.read(spec)
             for name, spec in (raw.get("p") or {}).items()}
    names = sorted(s.name for s in sp.sympify(expr).free_symbols
                   if s.name not in INPUTS)
    specs: Dict[str, Any] = {}
    for name in names:
        spec = given.get(name, known.get(name))
        if spec is None:
            raise ValueError("%s reads parameter %r, which has no spec: pass "
                             "one in params" % (site.name, name))
        specs[name] = spec.to_raw()
    for name in named_params(term):
        if name in known and name not in specs:
            specs[name] = known[name].to_raw()
    raw["e"] = sp.srepr(sp.sympify(expr))
    raw["p"] = specs
    return out


GATE_KINDS = ("always", "below", "above")


@dataclass(frozen=True)
class GateSpec:
    kind: str
    feature: Optional[str] = None
    threshold: Optional[sp.Expr] = None
    params: Mapping[str, Param] = None

    @property
    def gated(self) -> bool:
        return self.kind != "always"


def gate_of(document: Mapping[str, Any], law: str) -> GateSpec:
    activation = _term(document, law).get("activation") or {}
    kind = str(activation.get("kind") or "always")
    if kind == "always":
        return GateSpec("always", None, None, {})
    raw = activation.get("threshold") or {}
    return GateSpec(kind, activation.get("feature"),
                    sp.sympify(raw["e"]) if raw.get("e") is not None else None,
                    {str(n): Param.read(v)
                     for n, v in (raw.get("p") or {}).items()})


def set_gate(document: Mapping[str, Any], law: str, kind: str,
             feature: Optional[str] = None, threshold: Any = None,
             params: Optional[Mapping[str, Param]] = None) -> Document:
    from ..solve.record import INPUTS

    if kind not in GATE_KINDS:
        raise ValueError("gate kind %r (have: %s)" % (kind,
                                                      ", ".join(GATE_KINDS)))
    out = strip(document)
    term = _term(out, law)
    if kind == "always":
        term["activation"] = {"kind": "always", "threshold": None,
                              "feature": None}
        return out
    if threshold is None:
        raise ValueError("a %r gate needs a threshold" % kind)
    if feature is not None and feature not in INPUTS:
        raise ValueError("gate input %r (have: %s)"
                         % (feature, ", ".join(INPUTS)))
    expr = sp.sympify(threshold)
    known = gate_of(document, law).params or {}
    given = dict(params or {})
    specs = {}
    for name in sorted(s.name for s in expr.free_symbols
                       if s.name not in INPUTS):
        spec = given.get(name, known.get(name))
        if spec is None:
            raise ValueError("the threshold reads parameter %r, which has no "
                             "spec: pass one in params" % name)
        specs[name] = spec.to_raw()
    term["activation"] = {"kind": kind,
                          "threshold": {"e": sp.srepr(expr), "p": specs},
                          "feature": feature}
    return out


def _param_blocks(document: Mapping[str, Any]
                  ) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    for term in document.get("terms") or ():
        law = str(term.get("name"))
        for slot in ("threshold", "magnitude", "weight"):
            raw = _slot(term, slot)
            if raw and raw.get("p"):
                out.append(("%s/%s" % (law, slot), raw["p"]))
        bound = term.get("bound") or {}
        if bound.get("params"):
            out.append(("%s/bound" % law, bound["params"]))
    for index, bound in enumerate(document.get("global_bound") or ()):
        if bound and bound.get("params"):
            out.append(("global_bound[%d]" % index, bound["params"]))
    return out


def params(document: Mapping[str, Any]
           ) -> Dict[str, List[Tuple[str, Param]]]:
    out: Dict[str, List[Tuple[str, Param]]] = {}
    for location, block in _param_blocks(document):
        for name, spec in block.items():
            out.setdefault(str(name), []).append((location,
                                                  Param.read(spec)))
    return out


def namespace(document: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    return {name: places[-1][1].value
            for name, places in params(document).items()}


def tunable_params(document: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(name for name, places in params(document).items()
                 if all(p.tunable and p.value is not None
                        for _l, p in places))


def set_param(document: Mapping[str, Any], name: str,
              value: float) -> Document:
    out = strip(document)
    found = False
    for _location, block in _param_blocks(out):
        if name in block:
            spec = list(block[name])
            spec[0] = float(value)
            block[name] = spec
            found = True
    if not found:
        raise KeyError("the record writes no parameter %r" % name)
    return out


def _field(term: Mapping[str, Any], key: str) -> Any:
    return term.get(key, FIELD_DEFAULTS.get(key))


def _bound_kind(raw: Optional[Mapping[str, Any]]) -> Optional[str]:
    return None if not raw else str(raw.get("kind"))


def fixed_changes(before: Mapping[str, Any],
                  after: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    written, was_written = params(after), params(before)
    original = namespace(before)
    for name, places in was_written.items():
        if all(p.tunable for _l, p in places) or name not in written:
            continue
        there = {location: p.value for location, p in places}
        for location, spec in written[name]:
            expected = there.get(location, original[name])
            if spec.tunable:
                out.append("fixed parameter %r is marked tunable at %s"
                           % (name, location))
            elif spec.value != expected:
                out.append("fixed parameter %r is %r at %s, where the "
                           "original fixes it at %r"
                           % (name, spec.value, location, expected))
    return out


def interface_change(before: Mapping[str, Any], after: Mapping[str, Any],
                     *, fixed: bool = True) -> Optional[str]:
    for key in HEADER:
        if before.get(key) != after.get(key):
            return "%s changed from %r to %r" % (key, before.get(key),
                                                 after.get(key))
    old = [str(t.get("name")) for t in before.get("terms") or ()]
    new = [str(t.get("name")) for t in after.get("terms") or ()]
    if old != new:
        return "the laws changed from %s to %s" % (old, new)
    for was, now in zip(before.get("terms") or (), after.get("terms") or ()):
        name = str(was.get("name"))
        for key in TERM_INTERFACE:
            if _field(was, key) != _field(now, key):
                if (key == "direction" and (_field(was, key), _field(now, key))
                        in DIRECTION_EDITS):
                    continue
                return "law %r changed its %s from %r to %r" % (
                    name, key, _field(was, key), _field(now, key))
        if _bound_kind(was.get("bound")) != _bound_kind(now.get("bound")):
            return "law %r changed its bound from %r to %r" % (
                name, _bound_kind(was.get("bound")),
                _bound_kind(now.get("bound")))
        for slot in ("magnitude", "weight"):
            if bool(_slot(was, slot)) != bool(_slot(now, slot)):
                return "law %r %s its %s" % (
                    name, "lost" if _slot(was, slot) else "gained", slot)
    kinds = [_bound_kind(b) for b in before.get("global_bound") or ()]
    if kinds != [_bound_kind(b) for b in after.get("global_bound") or ()]:
        return "the global bounds changed"
    written, was_written = params(after), params(before)
    for term in after.get("terms") or ():
        for name in named_params(term):
            if name in was_written and name not in written:
                return ("law %r names parameter %r, which is no longer "
                        "written: its value is gone" % (term.get("name"),
                                                        name))
    if fixed:
        changes = fixed_changes(before, after)
        if changes:
            return changes[0]
    for name, places in written.items():
        values = {p.value for _l, p in places}
        if len(values) > 1 and len({p.value for _l, p in
                                    was_written.get(name, ())}) < 2:
            return ("parameter %r is written with different values %s; the "
                    "runtime flies the last one, %r, so the other edits do "
                    "nothing" % (name, sorted(values, key=str),
                                 places[-1][1].value))
    return None


__all__ = ["DIRECTION_EDITS", "Document", "FIELD_DEFAULTS", "GATE_KINDS", "GateSpec", "HEADER",
           "Param", "SCHEMA", "SLOTS", "Site", "SiteExpression",
           "TERM_INTERFACE", "canonical", "digest", "fixed_changes",
           "gate_of", "interface_change", "load", "named_params",
           "namespace", "params",
           "read_site", "set_gate", "set_param", "sites", "strip",
           "tunable_params", "write_site"]
