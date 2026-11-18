from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..solve.config import FamilyConfig
from ..solve.record import parse_record, release_sympy_caches
from .candidate import ORIGINAL
from .deadline import SCREEN_SECONDS, deadline
from .document import canonical, load
from .domain import Domain, domains_for
from .properties import (PROPERTIES, CommandScale, Profiler, Property,
                         PropertyProfile, compare)
from .units import expressions
from .validate import Validity, check


@dataclass
class Screen:
    label: str
    validity: Optional[Validity] = None
    expressions: Optional[Dict[str, Any]] = None
    profile: Optional[PropertyProfile] = None
    comparison: Any = None
    failed: Optional[Tuple[str, str, str]] = None
    probe: Any = None


def screen(document: Mapping[str, Any], *, label: str,
           config: FamilyConfig, domains: Mapping[str, Domain],
           profiler: Profiler, reference: PropertyProfile,
           properties: Sequence[Property]) -> Screen:
    out = Screen(label)
    started = time.time()
    try:
        with deadline(SCREEN_SECONDS, "screening %s" % label):
            validity = check(document, config=config, domains=domains,
                             label=label)
    except Exception as error:
        out.failed = ("validity", type(error).__name__, str(error))
        return out
    out.validity = validity
    if validity.units:
        out.expressions = expressions(validity.units)
    if not validity.ok:
        return out
    try:
        left = SCREEN_SECONDS - (time.time() - started)
        with deadline(max(left, 1.0), "screening %s" % label):
            out.profile = profiler.profile(validity.record, label)
            out.comparison = compare(reference, out.profile, properties)
    except Exception as error:
        out.profile = None
        out.failed = ("properties", type(error).__name__, str(error))
    return out


@dataclass
class Screener:
    config: FamilyConfig
    domains: Dict[str, Domain]
    profiler: Profiler
    reference: PropertyProfile
    properties: Tuple[Property, ...]

    @classmethod
    def build(cls, config: FamilyConfig, spec: Mapping[str, Any]
              ) -> "Screener":
        path = Path(spec["record_path"])
        original = canonical(load(path))
        record = parse_record(original, path, config.physics.record_params)
        domains = domains_for(record, config)
        properties = tuple(PROPERTIES)
        scale = CommandScale.of(record, config)
        profiler = Profiler(domains, properties, scale)
        validity = check(original, config=config, domains=domains,
                         label=ORIGINAL)
        reference = profiler.profile(validity.record, ORIGINAL)
        release_sympy_caches()
        return cls(config, domains, profiler, reference, properties)

    def __call__(self, document: Mapping[str, Any], label: str) -> Screen:
        return screen(document, label=label, config=self.config,
                      domains=self.domains, profiler=self.profiler,
                      reference=self.reference, properties=self.properties)


_SCREENERS: Dict[str, Screener] = {}
_PROBES: Dict[str, Any] = {}


def _probes(path: Optional[str]) -> Any:
    if path is None:
        return None
    if path not in _PROBES:
        _PROBES.clear()
        with open(path, "rb") as handle:
            _PROBES[path] = pickle.load(handle)
    return _PROBES[path]


def worker_screen(task: Tuple[Mapping[str, Any], str, Mapping[str, Any],
                              Optional[str]]) -> Screen:
    from .probe import read as read_probe
    from .weakness import _WORKER
    spec, label, document, probes = task
    config = _WORKER["config"]
    key = spec["record_path"]
    screener = _SCREENERS.get(key)
    if screener is None:
        screener = _SCREENERS[key] = Screener.build(config, spec)
    try:
        out = screener(document, label)
        probe_set = _probes(probes)
        if (probe_set is not None and out.failed is None
                and out.validity is not None and out.validity.ok):
            out.probe = read_probe(probe_set, out.validity.record, config)
    finally:
        release_sympy_caches()
    return out


def worker_probe(task: Tuple[Any, str]) -> Any:
    from .probe import read as read_probe
    from .weakness import _WORKER
    record, path = task
    try:
        return read_probe(_probes(path), record, _WORKER["config"])
    finally:
        release_sympy_caches()


__all__ = ["Screen", "Screener", "screen", "worker_probe", "worker_screen"]
