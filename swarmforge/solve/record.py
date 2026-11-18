from __future__ import annotations

import copyreg
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

import sympy as sp


def _mapping_proxy(items: Dict[str, Any]) -> MappingProxyType:
    return MappingProxyType(items)


copyreg.pickle(MappingProxyType,
               lambda proxy: (_mapping_proxy, (dict(proxy),)))

SCHEMA = "swarmforge.algorithm.v1"

DIST = sp.Symbol("dist")
CLEARANCE = sp.Symbol("clearance")
REL_SPEED = sp.Symbol("rel_speed")
SHILL_SPEED = sp.Symbol("shill_speed")

INPUTS: Tuple[str, ...] = ("dist", "clearance", "rel_speed", "shill_speed")
_INPUT_SYMBOLS = {DIST, CLEARANCE, REL_SPEED, SHILL_SPEED}

ADD_VELOCITY = "add_velocity"
VELOCITY_CMD = "velocity_cmd"
BASES = (ADD_VELOCITY, VELOCITY_CMD)

CENTRE_TO_CENTRE = "centre_to_centre"
CENTRE_TO_SURFACE = "centre_to_surface"

SOURCES = ("agents", "obstacles", "target", "bodies")
DIRECTIONS = ("toward", "away_unit", "away_raw", "match", "friction", "shill",
              "slide", "detour", "around")
AGGREGATORS = ("sum", "mean", "nearest")
GATE_KINDS = ("below", "above")
LIMIT_BOUNDS = ("clamp_speed", "clamp_accel")
RENORMALIZE = "renormalize"


@dataclass(frozen=True)
class Gate:
    kind: str
    feature: str
    threshold: sp.Expr


@dataclass(frozen=True)
class Bound:
    kind: str
    hi: Optional[float] = None
    lo: float = 0.0
    length: Optional[float] = None

    @classmethod
    def read(cls, raw: Mapping[str, Any], label: str) -> "Bound":
        kind = raw.get("kind")
        values = {name: (None if spec[0] is None else float(spec[0]))
                  for name, spec in (raw.get("params") or {}).items()}
        if kind in LIMIT_BOUNDS:
            hi: Optional[float] = None
            lo = 0.0
            for name, value in values.items():
                if value is None:
                    continue
                if "min" in name:
                    lo = value
                else:
                    hi = value if hi is None else min(hi, value)
            return cls("limit", hi=hi, lo=lo)
        if kind == RENORMALIZE:
            length = 1.0
            for value in values.values():
                if value is None:
                    raise ValueError("%s: renormalize bound has an unset "
                                     "parameter" % label)
                length *= value
            return cls("renormalize", length=length)
        raise ValueError("%s: bound kind %r is not modelled (have: %s, %s)"
                         % (label, kind, ", ".join(LIMIT_BOUNDS), RENORMALIZE))


@dataclass(frozen=True)
class Term:
    name: str
    role: str
    source: str
    signal: str
    direction: str
    aggregator: str
    measures: str
    range_param: Optional[str]
    fov_param: Optional[str]
    cap_param: Optional[str]
    magnitude: sp.Expr
    gate: Optional[Gate]
    weight: float
    bound: Optional[Bound]
    signal_speed: Optional[float]
    around: Optional[Tuple[float, float]] = None

    def reads(self) -> Tuple[str, ...]:
        used = set(self.magnitude.free_symbols)
        if self.gate is not None:
            used |= set(self.gate.threshold.free_symbols)
            used.add(sp.Symbol(self.gate.feature))
        return tuple(name for name in INPUTS if sp.Symbol(name) in used)


@dataclass(frozen=True)
class Record:
    path: Path
    target_key: str
    base: str
    roles: Tuple[str, ...]
    terms: Tuple[Term, ...]
    global_bounds: Tuple[Bound, ...]
    params: Mapping[str, Optional[float]]
    arbitration: Mapping[str, Any]
    notes: Tuple[str, ...]

    def unset_params(self) -> Tuple[str, ...]:
        return tuple(sorted(n for n, v in self.params.items() if v is None))


def _is_size(name: str) -> bool:
    return name == "size" or name.endswith("_size")


def _surface_dist(e: sp.Expr, surface: bool) -> sp.Expr:
    if not surface or any(_is_size(s.name) for s in e.free_symbols):
        return e
    return e.subs(DIST, CLEARANCE)


def _restore_clearance(e: sp.Expr) -> sp.Expr:
    sizes = [s for s in e.free_symbols if _is_size(s.name)]
    if not sizes:
        return e
    out = e.subs([(s, DIST - CLEARANCE) for s in sizes])
    return e if DIST in out.free_symbols else out


def _threshold_feature(e: sp.Expr) -> Tuple[sp.Expr, str]:
    feature = "dist"
    for size in [s for s in e.free_symbols if _is_size(s.name)]:
        if sp.simplify(sp.diff(e, size)) == 1:
            e, feature = sp.simplify(e - size), "clearance"
    return e, feature


def _params_of(raw: Optional[Mapping[str, Any]], key: str = "p"
               ) -> List[Tuple[str, Optional[float]]]:
    out = []
    for name, spec in ((raw or {}).get(key) or {}).items():
        out.append((str(name), None if spec[0] is None else float(spec[0])))
    return out


def _resolve(e: sp.Expr, params: Mapping[str, Optional[float]],
             label: str) -> sp.Expr:
    values = {}
    for symbol in e.free_symbols - _INPUT_SYMBOLS:
        value = params.get(symbol.name)
        if value is None:
            raise ValueError("%s reads %s, which has no value: name it in the "
                             "config's physics.record_params"
                             % (label, symbol.name))
        values[symbol] = value
    e = e.subs(values)
    if e.has(sp.logic.boolalg.ITE):
        e = e.rewrite(sp.Piecewise)
    return e


def release_sympy_caches() -> None:
    from sympy.polys import fields, rings
    for cache in (getattr(rings, "_ring_cache", None),
                  getattr(fields, "_field_cache", None)):
        if cache is not None:
            cache.clear()


def read_record(path: Path, fills: Optional[Mapping[str, float]] = None
                ) -> Record:
    from ..paths import relocate_path

    path = relocate_path(path)
    return parse_record(json.loads(path.read_text()), path, fills)


def parse_record(document: Mapping[str, Any], path: Path,
                 fills: Optional[Mapping[str, float]] = None) -> Record:
    path = Path(path)
    document = document.get("record", document)
    if document.get("schema") != SCHEMA:
        raise ValueError("%s is %r, not %r"
                         % (path, document.get("schema"), SCHEMA))
    base = document.get("base")
    if base not in BASES:
        raise ValueError("%s: base %r is not modelled (have: %s)"
                         % (path, base, ", ".join(BASES)))
    raw_terms = list(document.get("terms") or ())
    if not raw_terms:
        raise ValueError("%s carries no terms" % path)

    params: Dict[str, Optional[float]] = {}
    written: Dict[str, List[Optional[float]]] = {}

    def write(pairs):
        for name, value in pairs:
            params[name] = value
            written.setdefault(name, []).append(value)

    for term in raw_terms:
        activation = term.get("activation") or {}
        write(_params_of(activation.get("threshold")))
        write(_params_of(term.get("magnitude")))
        write(_params_of(term.get("weight")))
        write(_params_of(term.get("bound"), "params"))
    for bound in document.get("global_bound") or ():
        if bound:
            write(_params_of(bound, "params"))
    for term in raw_terms:
        membership = term.get("membership") or {}
        for key in ("range", "fov", "cap"):
            name = membership.get(key)
            if name and name not in params:
                params[str(name)] = None

    notes: List[str] = []
    for name, values in written.items():
        if len(set(values)) > 1:
            notes.append("parameter %r is written %d times with different "
                         "values %s; the runtime keeps the last, %r"
                         % (name, len(values), values, params[name]))
    for name, value in (fills or {}).items():
        if name not in params:
            raise ValueError("record_params names %r, which %s does not use"
                             % (name, path.name))
        if params[name] is not None:
            raise ValueError("record_params gives %r, but the record already "
                             "values it at %r; the record is the controller"
                             % (name, params[name]))
        params[name] = float(value)

    try:
        terms = tuple(_read_term(t, params, path) for t in raw_terms)
    finally:
        release_sympy_caches()
    names = [t.name for t in terms]
    if len(set(names)) != len(names):
        raise ValueError("%s: duplicate term names %s" % (path, names))
    bounds = tuple(Bound.read(b, "global bound") for b in
                   (document.get("global_bound") or ()) if b)
    return Record(
        path=path,
        target_key=str(document.get("target_key", "")),
        base=str(base),
        roles=tuple(document.get("roles") or ()),
        terms=terms,
        global_bounds=bounds,
        params=MappingProxyType(dict(params)),
        arbitration=MappingProxyType(dict(document.get("arbitration") or {})),
        notes=tuple(notes),
    )


def _read_term(raw: Mapping[str, Any], params: Mapping[str, Optional[float]],
               path: Path) -> Term:
    name = str(raw.get("name"))
    label = "%s: term %r" % (path.name, name)
    source = str(raw.get("source"))
    direction = str(raw.get("direction"))
    aggregator = str(raw.get("aggregator") or "sum")
    measures = str(raw.get("measures") or CENTRE_TO_CENTRE)
    if source not in SOURCES:
        raise ValueError("%s: source %r is not modelled (have: %s)"
                         % (label, source, ", ".join(SOURCES)))
    if direction not in DIRECTIONS:
        raise ValueError("%s: direction %r is not modelled (have: %s)"
                         % (label, direction, ", ".join(DIRECTIONS)))
    if aggregator not in AGGREGATORS:
        raise ValueError("%s: aggregator %r is not modelled (have: %s)"
                         % (label, aggregator, ", ".join(AGGREGATORS)))
    if measures not in (CENTRE_TO_CENTRE, CENTRE_TO_SURFACE):
        raise ValueError("%s: measures %r" % (label, measures))
    if (raw.get("anchor") or "self") != "self":
        raise ValueError("%s: anchor %r is not modelled (only 'self')"
                         % (label, raw.get("anchor")))
    if (raw.get("stochastic") or "none") != "none":
        raise ValueError("%s: a stochastic term is not modelled" % label)
    surface = measures == CENTRE_TO_SURFACE

    magnitude_raw = raw.get("magnitude")
    if not magnitude_raw:
        raise ValueError("%s has no magnitude" % label)
    magnitude = _restore_clearance(
        _surface_dist(sp.sympify(magnitude_raw["e"]), surface))
    magnitude = _resolve(magnitude, params, label + " magnitude")

    gate = None
    activation = raw.get("activation") or {}
    kind = activation.get("kind") or "always"
    if kind != "always":
        if kind not in GATE_KINDS:
            raise ValueError("%s: activation %r is not modelled" % (label, kind))
        if not activation.get("threshold"):
            raise ValueError("%s: a %r gate with no threshold" % (label, kind))
        threshold = sp.sympify(activation["threshold"]["e"])
        named = activation.get("feature")
        if named:
            if named not in INPUTS:
                raise ValueError("%s gates on %r (have: %s)"
                                 % (label, named, ", ".join(INPUTS)))
            threshold, feature = _surface_dist(threshold, surface), str(named)
        else:
            threshold, feature = _threshold_feature(threshold)
        gate = Gate(str(kind), feature,
                    _resolve(threshold, params, label + " threshold"))

    weight = 1.0
    if raw.get("weight"):
        value = _resolve(_surface_dist(sp.sympify(raw["weight"]["e"]), surface),
                         params, label + " weight")
        if value.free_symbols:
            raise ValueError("%s: weight reads %s; the runtime evaluates it on "
                             "parameters alone" % (label, value.free_symbols))
        weight = float(value)

    bound = Bound.read(raw["bound"], label) if raw.get("bound") else None

    signal_speed = None
    if raw.get("signal_speed"):
        signal_speed = params.get(str(raw["signal_speed"]))
        if signal_speed is None:
            raise ValueError("%s names signal_speed %r, which has no value"
                             % (label, raw["signal_speed"]))
    if direction == "shill" and signal_speed is None:
        raise ValueError("%s is a shill barrier with no signal_speed" % label)
    around = raw.get("around")
    if (direction == "around") != bool(around):
        raise ValueError("%s: direction %r with around %r" % (label, direction, around))

    term = Term(
        name=name, role=str(raw.get("role") or "agent"), source=source,
        signal=str(raw.get("signal") or "position"), direction=direction,
        aggregator=aggregator, measures=measures,
        range_param=(raw.get("membership") or {}).get("range"),
        fov_param=(raw.get("membership") or {}).get("fov"),
        cap_param=(raw.get("membership") or {}).get("cap"),
        magnitude=magnitude, gate=gate, weight=weight, bound=bound,
        signal_speed=signal_speed,
        around=(float(around[0]), float(around[1])) if around else None)
    if "shill_speed" in term.reads() and direction != "shill":
        raise ValueError("%s reads shill_speed but is not a shill barrier, so "
                         "the runtime would supply no such input" % label)
    return term


def named_records(config_record: Path, *, records_dir: Optional[Path] = None,
                  records: Tuple[str, ...] = (),
                  with_config: bool = False) -> Dict[str, Path]:
    from ..paths import relocate_metadata, relocate_path

    out: Dict[str, Path] = {}
    if with_config or not (records_dir or records):
        out["config"] = relocate_path(config_record)
    if records_dir:
        folder = relocate_path(records_dir)
        index = folder / "index.json"
        if index.is_file():
            for entry in relocate_metadata(json.loads(index.read_text()))["controllers"]:
                path = relocate_path(folder / entry["algorithm"])
                if path.is_file():
                    out[entry["id"]] = path
        else:
            for path in sorted(folder.glob("*/algorithm.json")):
                out[path.parent.name] = path
    for item in records:
        name, _, path = item.partition("=")
        if not path:
            name, path = Path(item).parent.name, item
        out[name] = relocate_path(path)
    return out


__all__ = ["ADD_VELOCITY", "Bound", "Gate", "INPUTS", "Record", "Term",
           "VELOCITY_CMD", "named_records", "parse_record", "read_record",
           "release_sympy_caches"]
