from __future__ import annotations

import json
import os
from typing import Optional

import sympy as sp

from .algorithm import (Activation, ActivationKind, Aggregator, Algorithm, Anchor,
                        Arbitration, Base, Bound, BoundOp, Direction, Measures,
                        Membership, Role, Signal, Source, Stochastic, Term)
from .expr import Expr, ParamSpec

SCHEMA = "swarmforge.algorithm.v1"


def _spec_enc(s: ParamSpec):
    return [s.value, s.tunable, list(s.bounds) if s.bounds else None]


def _spec_dec(v) -> ParamSpec:
    return ParamSpec(float(v[0]), bool(v[1]), tuple(v[2]) if v[2] else None)


def _expr_enc(e: Optional[Expr]):
    if e is None:
        return None
    return {"e": sp.srepr(e.sympy),
            "p": {n: _spec_enc(s) for n, s in e.params.items()}}


def _expr_dec(r) -> Optional[Expr]:
    if r is None:
        return None
    return Expr(sp.sympify(r["e"]),
                {n: _spec_dec(v) for n, v in r["p"].items()})


def _mem_enc(m: Membership):
    return {"range": m.range_param, "fov": m.fov_param, "cap": m.cap_param,
            "topology": m.topology, "gate": m.gate, "mutable": m.mutable}


def _mem_dec(r) -> Membership:
    return Membership(r["range"], r["fov"], r["cap"], r["topology"], r["gate"],
                      r["mutable"])


def _act_enc(a: Activation):
    return {"kind": a.kind.value, "threshold": _expr_enc(a.threshold),
            "feature": a.feature}


def _act_dec(r) -> Activation:
    return Activation(ActivationKind(r["kind"]), _expr_dec(r["threshold"]),
                      feature=r.get("feature"))


def _bound_enc(b: Optional[BoundOp]):
    if b is None:
        return None
    return {"kind": b.kind.value,
            "params": {n: _spec_enc(s) for n, s in b.params.items()}}


def _bound_dec(r) -> Optional[BoundOp]:
    if r is None:
        return None
    return BoundOp(Bound(r["kind"]),
                   {n: _spec_dec(v) for n, v in r["params"].items()})


def _term_enc(t: Term):
    return {
        "name": t.name, "role": t.role.value, "source": t.source.value,
        "membership": _mem_enc(t.membership), "signal": t.signal.value,
        "direction": t.direction.value, "magnitude": _expr_enc(t.magnitude),
        "measures": t.measures.value,
        "anchor": t.anchor.value, "activation": _act_enc(t.activation),
        "aggregator": t.aggregator.value, "bound": _bound_enc(t.bound),
        "stochastic": t.stochastic.value, "weight": _expr_enc(t.weight),
        "signal_speed": t.signal_speed,
    }


def _term_dec(r) -> Term:
    return Term(
        r["name"], Role(r["role"]), Source(r["source"]), _mem_dec(r["membership"]),
        Signal(r["signal"]), Direction(r["direction"]), _expr_dec(r["magnitude"]),
        measures=Measures(r.get("measures", Measures.CENTRE_TO_CENTRE.value)),
        anchor=Anchor(r["anchor"]), activation=_act_dec(r["activation"]),
        aggregator=Aggregator(r["aggregator"]), bound=_bound_dec(r["bound"]),
        stochastic=Stochastic(r["stochastic"]), weight=_expr_dec(r["weight"]),
        signal_speed=r.get("signal_speed"))


def _arb_enc(a: Optional[Arbitration]):
    if a is None:
        return None
    return {"role_rule": a.role_rule, "planner": a.planner,
            "target_selector": dict(a.target_selector),
            "transition": a.transition, "state": list(a.state)}


def _arb_dec(r) -> Optional[Arbitration]:
    if r is None:
        return None
    return Arbitration(r["role_rule"], r["planner"], dict(r["target_selector"]),
                       r.get("transition"), tuple(r.get("state", ())))


def algorithm_to_record(alg: Algorithm) -> dict:
    return {
        "schema": SCHEMA,
        "target_key": alg.target_key,
        "roles": [r.value for r in alg.roles],
        "base": alg.base.value,
        "terms": [_term_enc(t) for t in alg.terms],
        "global_bound": [_bound_enc(b) for b in alg.global_bound],
        "arbitration": _arb_enc(alg.arbitration),
    }


def algorithm_from_record(rec: dict) -> Algorithm:
    rec = rec.get("record", rec)
    return Algorithm(
        target_key=rec["target_key"],
        roles=tuple(Role(v) for v in rec["roles"]),
        terms=[_term_dec(t) for t in rec["terms"]],
        base=Base(rec["base"]),
        global_bound=tuple(_bound_dec(b) for b in rec["global_bound"]),
        arbitration=_arb_dec(rec["arbitration"]))


def _is_yaml(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".yaml", ".yml")


def save_algorithm(alg: Algorithm, path: str) -> None:
    rec = algorithm_to_record(alg)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        if _is_yaml(path):
            import yaml
            yaml.safe_dump(rec, fh, sort_keys=False, default_flow_style=False)
        else:
            json.dump(rec, fh, indent=2)


def load_algorithm(path: str) -> Algorithm:
    with open(path) as fh:
        if _is_yaml(path):
            import yaml
            rec = yaml.safe_load(fh)
        else:
            rec = json.load(fh)
    return algorithm_from_record(rec)


def _demo() -> None:
    from .algorithm import identity, names
    for key in names():
        alg = identity(key)
        rt = algorithm_from_record(algorithm_to_record(alg))
        ok = rt.signature() == alg.signature()
        print("%-16s round-trip %s" % (key, "OK" if ok else "MISMATCH"))


if __name__ == "__main__":
    _demo()
