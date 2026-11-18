from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import sympy as sp
from sympy.logic.boolalg import (And, BooleanFalse, BooleanTrue, ITE, Not, Or)

from . import interval as iv
from .interval import I, T

Env = Dict[str, I]


def compile_value(e: sp.Expr) -> Callable[[Env], I]:
    e = sp.sympify(e)
    if e.is_Symbol:
        name = e.name
        return lambda env: env[name]
    if e.is_Number or e.is_NumberSymbol:
        c = float(e)
        return lambda env: I(np.asarray(c))
    if isinstance(e, sp.Add):
        parts = [compile_value(a) for a in e.args]

        def add(env):
            out = parts[0](env)
            for p in parts[1:]:
                out = out + p(env)
            return out
        return add
    if isinstance(e, sp.Mul):
        coeff, rest = e.as_coeff_Mul()
        if coeff != 1 and rest != 1:
            c, inner = float(coeff), compile_value(rest)
            return lambda env: inner(env) * c
        parts = [compile_value(a) for a in e.args]

        def mul(env):
            out = parts[0](env)
            for p in parts[1:]:
                out = out * p(env)
            return out
        return mul
    if isinstance(e, sp.Pow):
        base, power = e.args
        if not power.is_Number:
            raise NotImplementedError("a power that is not a number: %s" % e)
        inner = compile_value(base)
        if power.is_Integer:
            n = int(power)
            return lambda env: iv.powi(inner(env), n)
        if power == sp.Rational(1, 2):
            return lambda env: iv.sqrt(inner(env))
        if power == sp.Rational(-1, 2):
            return lambda env: iv.recip(iv.sqrt(inner(env)))
        p = float(power)
        return lambda env: iv.powr(inner(env), p)
    if isinstance(e, sp.exp):
        inner = compile_value(e.args[0])
        return lambda env: iv.exp(inner(env))
    if isinstance(e, sp.Abs):
        inner = compile_value(e.args[0])
        return lambda env: iv.absolute(inner(env))
    if isinstance(e, (sp.Max, sp.Min)):
        parts = [compile_value(a) for a in e.args]
        pick = iv.maximum if isinstance(e, sp.Max) else iv.minimum

        def extreme(env):
            out = parts[0](env)
            for p in parts[1:]:
                out = pick(out, p(env))
            return out
        return extreme
    if isinstance(e, sp.Piecewise):
        branches = [(compile_value(value), compile_truth(cond))
                    for value, cond in e.args]

        def piecewise(env):
            lo = hi = None
            open_ = None
            for value, cond in branches:
                v, c = value(env), cond(env)
                shape = np.broadcast_shapes(v.shape, c.shape)
                if lo is None:
                    lo = np.full(shape, np.inf)
                    hi = np.full(shape, -np.inf)
                    open_ = np.ones(shape, dtype=bool)
                else:
                    shape = np.broadcast_shapes(shape, lo.shape)
                    lo, hi = np.broadcast_to(lo, shape), np.broadcast_to(hi, shape)
                    open_ = np.broadcast_to(open_, shape)
                may = open_ & ~c.no
                lo = np.where(may, np.minimum(lo, v.lo), lo)
                hi = np.where(may, np.maximum(hi, v.hi), hi)
                open_ = open_ & ~c.yes
            lo = np.where(open_, -np.inf, lo)
            hi = np.where(open_, np.inf, hi)
            return I(lo, hi)
        return piecewise
    raise NotImplementedError("no interval rule for %s (%s)" % (type(e), e))


def compile_truth(c) -> Callable[[Env], T]:
    if isinstance(c, BooleanTrue) or c is True:
        return lambda env: T.of(np.asarray(True))
    if isinstance(c, BooleanFalse) or c is False:
        return lambda env: T.of(np.asarray(False))
    if isinstance(c, And):
        parts = [compile_truth(a) for a in c.args]

        def both(env):
            out = parts[0](env)
            for p in parts[1:]:
                out = out & p(env)
            return out
        return both
    if isinstance(c, Or):
        parts = [compile_truth(a) for a in c.args]

        def either(env):
            out = parts[0](env)
            for p in parts[1:]:
                out = out | p(env)
            return out
        return either
    if isinstance(c, Not):
        inner = compile_truth(c.args[0])
        return lambda env: ~inner(env)
    if isinstance(c, ITE):
        cond, then, other = (compile_truth(a) for a in c.args)
        return lambda env: iv.select_t(cond(env), then(env), other(env))
    if isinstance(c, sp.Piecewise):
        branches = [(compile_truth(value), compile_truth(cond))
                    for value, cond in c.args]

        def chain(env):
            out = T.of(np.asarray(False))
            for value, cond in reversed(branches):
                out = iv.select_t(cond(env), value(env), out)
            return out
        return chain
    rules = {sp.StrictLessThan: iv.lt, sp.LessThan: iv.le,
             sp.StrictGreaterThan: iv.gt, sp.GreaterThan: iv.ge}
    for kind, rule in rules.items():
        if isinstance(c, kind):
            left, right = compile_value(c.lhs), compile_value(c.rhs)
            return lambda env, rule=rule: rule(left(env), right(env))
    raise NotImplementedError("no interval rule for the condition %s (%s)"
                              % (type(c), c))
