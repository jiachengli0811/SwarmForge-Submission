from __future__ import annotations

import dataclasses
import json
from typing import Dict, List, Optional

from ..classify import (ActivationIR, AlgorithmIR, BoundIR, MembershipIR, TermIR)


def _bound(d: Optional[Dict]) -> Optional[BoundIR]:
    return None if d is None else BoundIR(kind=d["kind"], params=dict(d["params"]))


def _term(d: Dict) -> TermIR:
    t = TermIR(name=d["name"])
    for k, v in d.items():
        if k == "membership":
            t.membership = MembershipIR(**v)
        elif k == "activation":
            t.activation = ActivationIR(**v)
        elif k == "bound":
            t.bound = _bound(v)
        elif hasattr(t, k):
            setattr(t, k, v)
    return t


def from_dict(d: Dict) -> AlgorithmIR:
    _check_vector_schema(d)
    algo = AlgorithmIR(target_key=d["target_key"])
    for k, v in d.items():
        if k == "terms":
            algo.terms = [_term(x) for x in v]
        elif k == "global_bounds":
            algo.global_bounds = [_bound(x) for x in v]
        elif k == "roles":
            algo.roles = tuple(v)
        elif hasattr(algo, k):
            setattr(algo, k, v)
    return algo


def is_record(d: Dict) -> bool:
    d = d.get("record", d)
    if not isinstance(d, dict):
        return False
    return str(d.get("schema", "")).startswith("swarmforge.algorithm") or \
        ("global_bound" in d and "terms" in d)


def _check_vector_schema(d: Dict) -> None:
    schema = d.get("schema")
    if schema == "swarmmodeling.exact-law.v1":
        raise ValueError("exact-law records run on swarmmodeling.runtime.exact.ExactLaw")
    if schema not in (None, "swarmforge.algorithm.v1"):
        raise ValueError(f"unsupported vector-law schema {schema!r}")


def _is_size(name: str) -> bool:
    return name == "size" or name.endswith("_size")


def _restore_clearance(e, used: set):
    import sympy as sp
    dist, clear = sp.Symbol("dist"), sp.Symbol("clearance")
    sizes = [x for x in e.free_symbols if _is_size(x.name)]
    if not sizes:
        return e
    out = e.subs([(x, dist - clear) for x in sizes])
    if dist in out.free_symbols:
        used.update(x.name for x in sizes)
        return e
    return out


def _rec_expr(r, params: Dict, used: set, restore: bool = True,
              surface: bool = False) -> Optional[str]:
    if r is None:
        return None
    import sympy as sp
    for name, spec in (r.get("p") or {}).items():
        params[name] = {"value": None if spec[0] is None else float(spec[0]),
                        "tunable": bool(spec[1])}
        if not _is_size(name):
            used.add(name)
    e = sp.sympify(r["e"])
    e = _surface_dist(e, surface)
    return str(_restore_clearance(e, used) if restore else e)


def _rec_bound(r, params: Dict, used: set) -> Optional[BoundIR]:
    if r is None:
        return None
    vals = {}
    for name, spec in (r.get("params") or {}).items():
        vals[name] = None if spec[0] is None else float(spec[0])
        params[name] = {"value": vals[name], "tunable": bool(spec[1])}
        used.add(name)
    return BoundIR(kind=r["kind"], params=vals)


def _rec_activation(a: Dict, params: Dict, used: set,
                    surface: bool = False) -> ActivationIR:
    import sympy as sp
    named = a.get("feature")
    text = _rec_expr(a.get("threshold"), params, used, restore=False)
    if not text:
        return ActivationIR(kind=a["kind"], feature=named)
    e = sp.sympify(text)
    if named:
        return ActivationIR(kind=a["kind"],
                            threshold=str(_surface_dist(e, surface)),
                            feature=named)
    feature = "dist"
    for x in [y for y in e.free_symbols if _is_size(y.name)]:
        if sp.simplify(sp.diff(e, x)) == 1:
            e, feature = sp.simplify(e - x), "clearance"
        else:
            used.add(x.name)
    return ActivationIR(kind=a["kind"], threshold=str(e), feature=feature)


def _surface_dist(e, surface: bool):
    import sympy as sp
    if not surface or any(_is_size(y.name) for y in e.free_symbols):
        return e
    return e.subs(sp.Symbol("dist"), sp.Symbol("clearance"))


def _rec_term(d: Dict, params: Dict, used: set) -> TermIR:
    m = d["membership"]
    surface = d.get("measures") == "centre_to_surface"
    return TermIR(
        name=d["name"], role=d["role"], source=d["source"],
        signal=d["signal"], direction=d["direction"], anchor=d["anchor"],
        aggregator=d["aggregator"],
        measures=d.get("measures", "centre_to_centre"),
        membership=MembershipIR(range_param=m.get("range"),
                                fov_param=m.get("fov"),
                                cap_param=m.get("cap"),
                                topology=m.get("topology")),
        activation=_rec_activation(d["activation"], params, used, surface),
        magnitude=_rec_expr(d.get("magnitude"), params, used, surface=surface),
        weight=_rec_expr(d.get("weight"), params, used, surface=surface),
        bound=_rec_bound(d.get("bound"), params, used),
        signal_speed=d.get("signal_speed"))


def from_record(d: Dict) -> AlgorithmIR:
    _check_vector_schema(d)
    d = d.get("record", d)
    _check_vector_schema(d)
    params: Dict[str, Dict] = {}
    used: set = set()
    algo = AlgorithmIR(target_key=d.get("target_key", "algorithm"))
    algo.roles = tuple(d.get("roles", ()))
    algo.base = d.get("base", algo.base)
    algo.terms = [_rec_term(t, params, used) for t in d.get("terms", [])]
    algo.global_bounds = [b for b in (_rec_bound(x, params, used)
                                      for x in d.get("global_bound", []))
                          if b is not None]
    arb = d.get("arbitration")
    if arb:
        algo.arbitration = {"role_rule": arb.get("role_rule"),
                            "planner": arb.get("planner"),
                            "target_selector": dict(arb.get("target_selector")
                                                    or {}),
                            "transition": arb.get("transition"),
                            "state": list(arb.get("state") or ())}
    unset = []
    for t in algo.terms:
        for name in (t.membership.range_param, t.membership.fov_param,
                     t.membership.cap_param):
            if name and name not in params:
                params[name] = {"value": None, "tunable": False}
                used.add(name)
                unset.append(name)
    restored = sorted(n for n in params if _is_size(n) and n not in used)
    algo.params = {n: v for n, v in params.items() if n not in restored}
    algo.notes = list(algo.notes) + ["loaded from a swarmforge.algorithm record"]
    if restored:
        algo.notes.append(
            "surface distance restored to `clearance` (the scene measures it); "
            "the record's frozen %s dropped" % ", ".join(restored))
    if unset:
        algo.notes.append(
            "sensing limits carry no value in a record: %s -- set them or the "
            "term sees every candidate" % ", ".join(sorted(set(unset))))
    return algo


def unset_params(algo: AlgorithmIR) -> list:
    return sorted(n for n, v in algo.params.items() if v.get("value") is None)


def load(path: str) -> AlgorithmIR:
    with open(path) as f:
        d = json.load(f)
    return from_record(d) if is_record(d) else from_dict(d)


def save(algo: AlgorithmIR, path: str) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(dataclasses.asdict(algo), indent=2))


def save_record(algo: AlgorithmIR, path: str, self_size: float = 0.0) -> None:
    from .. import emit
    from swarmforge.algorithm.serialize import save_algorithm
    save_algorithm(emit.to_swarmff(algo, self_size=self_size), path)


def term(self: AlgorithmIR, name: str) -> TermIR:
    for t in self.terms:
        if t.name == name:
            return t
    raise KeyError("no term %r (have: %s)"
                   % (name, ", ".join(t.name for t in self.terms)))


def param(self: AlgorithmIR, name: str) -> float:
    return self.params[name]["value"]


def set_param(self: AlgorithmIR, name: str, value: float,
              tunable: Optional[bool] = None) -> None:
    spec = self.params.setdefault(name, {"value": value, "tunable": True})
    spec["value"] = float(value)
    if tunable is not None:
        spec["tunable"] = bool(tunable)


def drop_term(self: AlgorithmIR, name: str) -> TermIR:
    t = term(self, name)
    self.terms.remove(t)
    return t


def copy(self: AlgorithmIR) -> AlgorithmIR:
    return from_dict(dataclasses.asdict(self))


def describe(self: AlgorithmIR) -> str:
    from .. import emit
    return emit.summary(self)


for _name, _fn in (("term", term), ("param", param), ("set_param", set_param),
                   ("drop_term", drop_term), ("copy", copy),
                   ("describe", describe),
                   ("save", lambda self, path: save(self, path))):
    setattr(AlgorithmIR, _name, _fn)
