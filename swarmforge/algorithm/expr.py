from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import sympy as sp


class Input(Enum):
    DIST = "dist"
    REL_SPEED = "rel_speed"
    AXIS = "axis"
    SHILL_SPEED = "shill_speed"


DIST = sp.Symbol("dist")
REL_SPEED = sp.Symbol("rel_speed")
AXIS = sp.Symbol("axis")
SHILL_SPEED = sp.Symbol("shill_speed")


_MAX_NUMBER_BITS = 256
_MAX_FLOAT_BINARY_EXP = 512


def _number_is_tame(value: sp.Expr) -> bool:
    if value in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        return False
    if isinstance(value, sp.Float):
        sign, man, exp, bc = value._mpf_
        if man == 0:
            return True
        return abs(exp) <= _MAX_FLOAT_BINARY_EXP and bc <= _MAX_NUMBER_BITS
    if isinstance(value, sp.Integer):
        return abs(int(value)).bit_length() <= _MAX_NUMBER_BITS
    if isinstance(value, sp.Rational):
        return abs(int(value.p)).bit_length() <= _MAX_NUMBER_BITS and \
            abs(int(value.q)).bit_length() <= _MAX_NUMBER_BITS
    if getattr(value, "is_Number", False):
        try:
            v = float(value)
        except Exception:
            return False
        return v == v and v not in (float("inf"), float("-inf"))
    return True


def expression_is_tame(expr) -> bool:
    try:
        expr = sp.sympify(expr)
        for sub in sp.preorder_traversal(expr):
            if getattr(sub, "is_Number", False) and not _number_is_tame(sub):
                return False
        return True
    except Exception:
        return False


def sigma1(z):
    return z / sp.sqrt(1 + z ** 2)


def sigma_norm(x, eps):
    return (sp.sqrt(1 + eps * x ** 2) - 1) / eps


@dataclass(frozen=True)
class ParamSpec:
    value: float
    tunable: bool = False
    bounds: Optional[Tuple[float, float]] = None


@dataclass
class Expr:
    sympy: sp.Expr
    params: Dict[str, ParamSpec] = field(default_factory=dict)

    def inputs(self) -> List[Input]:
        names = {s.name for s in self.sympy.free_symbols}
        return [i for i in Input if i.value in names]

    def tunable_params(self) -> List[str]:
        return [n for n, s in self.params.items() if s.tunable]

    def _values(self) -> Dict[sp.Symbol, float]:
        return {sp.Symbol(n): s.value for n, s in self.params.items()}

    def eval(self, env: Dict[Input, float]) -> float:
        subs = self._values()
        subs.update({sp.Symbol(k.value): v for k, v in env.items()})
        return float(self.sympy.subs(subs))

    def compile(self) -> Callable[[Dict[Input, float]], float]:
        baked = self.sympy.subs(self._values())
        order = self.inputs()
        f = sp.lambdify([sp.Symbol(i.value) for i in order], baked,
                        modules=[{"Max": max, "Min": min}, "math"])
        return lambda env, _f=f, _o=order: _f(*[env[i] for i in _o])

    def compiled(self) -> Callable[[Dict[Input, float]], float]:
        fn = getattr(self, "_fn", None)
        if fn is None:
            fn = self.compile()
            object.__setattr__(self, "_fn", fn)
        return fn

    def __getstate__(self) -> Dict[str, object]:
        state = dict(self.__dict__)
        state.pop("_fn", None)
        return state

    def simplify(self) -> "Expr":
        return Expr(sp.simplify(self.sympy), dict(self.params))

    def diff(self, wrt: Input) -> "Expr":
        return Expr(sp.diff(self.sympy, sp.Symbol(wrt.value)), dict(self.params))

    def subs_params(self, **values) -> "Expr":
        params = dict(self.params)
        for n, v in values.items():
            old = params[n]
            params[n] = ParamSpec(float(v), old.tunable, old.bounds)
        return Expr(self.sympy, params)

    def values(self) -> Dict[str, float]:
        return {n: s.value for n, s in self.params.items()}

    def with_values(self, values: Dict[str, float]) -> "Expr":
        params = dict(self.params)
        for n, v in values.items():
            if n in params:
                old = params[n]
                params[n] = ParamSpec(float(v), old.tunable, old.bounds)
        return Expr(self.sympy, params)

    def rewrite(self, new_sympy,
                pool: Optional[Dict[str, "ParamSpec"]] = None) -> "Expr":
        e = sp.sympify(new_sympy)
        input_names = {i.value for i in Input}
        src = dict(self.params)
        if pool:
            for n, s in pool.items():
                src.setdefault(n, s)
        names = {sym.name for sym in e.free_symbols} - input_names
        order = [n for n in src if n in names] + sorted(names.difference(src))
        params: Dict[str, ParamSpec] = {
            n: src.get(n, ParamSpec(1.0, True)) for n in order}
        return Expr(e, params)

    def render(self) -> str:
        return str(self.sympy)

    def __repr__(self):
        return self.render()


class Params:
    def __init__(self):
        self._specs: Dict[str, ParamSpec] = {}

    def gain(self, name: str, value: float,
             bounds: Optional[Tuple[float, float]] = None) -> sp.Symbol:
        self._specs[name] = ParamSpec(float(value), True, bounds)
        return sp.Symbol(name)

    def fixed(self, name: str, value: float) -> sp.Symbol:
        self._specs[name] = ParamSpec(float(value), False)
        return sp.Symbol(name)

    def wrap(self, sympy_expr) -> Expr:
        e = sp.sympify(sympy_expr)
        names = {s.name for s in e.free_symbols}
        used = {n: s for n, s in self._specs.items() if n in names}
        return Expr(e, used)


if __name__ == "__main__":
    p = Params()
    rep = p.wrap(p.gain("p_rep", 0.1) * (p.fixed("r0_rep", 10.0) - DIST))
    print("expr:      ", rep.render())
    print("inputs:    ", [i.value for i in rep.inputs()])
    print("tunable:   ", rep.tunable_params())
    print("eval d=3:  ", rep.eval({Input.DIST: 3.0}), "(== 0.1*(10-3) = 0.7)")
    print("d/d(dist): ", rep.diff(Input.DIST).render(), "(sympy diff, for free)")
    f = rep.compile()
    print("compiled:  ", f({Input.DIST: 3.0}))
