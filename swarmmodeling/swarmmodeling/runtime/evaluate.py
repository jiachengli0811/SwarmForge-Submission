from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..classify import AlgorithmIR, BoundIR, TermIR

Vec = Tuple[float, ...]


def vzero(n: int) -> Vec:
    return tuple([0.0] * n)


def vadd(a: Vec, b: Vec) -> Vec:
    return tuple(x + y for x, y in zip(a, b))


def vsub(a: Vec, b: Vec) -> Vec:
    return tuple(x - y for x, y in zip(a, b))


def vscale(a: Vec, k: float) -> Vec:
    return tuple(x * k for x in a)


def vnorm(a: Vec) -> float:
    return math.sqrt(sum(x * x for x in a))


def vunit(a: Vec) -> Vec:
    m = vnorm(a)
    return vscale(a, 1.0 / m) if m else tuple([0.0] * len(a))


def limit_magnitude(v: Vec, hi: Optional[float], lo: float = 0.0) -> Vec:
    m = vnorm(v)
    if hi is not None and m > hi:
        return vscale(v, hi / m)
    if lo and m < lo:
        if m == 0.0:
            raise ZeroDivisionError("min-magnitude clamp on a zero vector")
        return vscale(v, lo / m)
    return v


_LITERALS = {"true", "false"}
_CACHE: Dict[str, Tuple[Callable, Tuple[str, ...]]] = {}


def _symbol_names(text: str) -> set:
    import ast

    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError("expression %r does not parse: %s" % (text, error))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return used - called - _LITERALS


def compile_expr(text: str) -> Tuple[Callable, Tuple[str, ...]]:
    hit = _CACHE.get(text)
    if hit is not None:
        return hit
    import sympy as sp
    from sympy.core.function import AppliedUndef
    names = _symbol_names(text)
    e = sp.sympify(text, locals={n: sp.Symbol(n) for n in names})
    undefined = sorted({type(f).__name__ for f in e.atoms(AppliedUndef)})
    if undefined:
        raise ValueError(
            "expression %r calls %s, which sympy does not define"
            % (text, ", ".join(undefined)))
    if e.has(sp.logic.boolalg.ITE):
        e = e.rewrite(sp.Piecewise)
    args = tuple(sorted(s.name for s in e.free_symbols))
    f = sp.lambdify([sp.Symbol(a) for a in args], e,
                    modules=[{"Max": max, "Min": min}, "math"])
    if not args:
        f = (lambda _v=float(e): _v)
    _CACHE[text] = (f, args)
    return _CACHE[text]


def eval_expr(text: Optional[str], env: Dict[str, float]) -> float:
    if text is None or text == "":
        return 1.0
    f, args = compile_expr(text)
    try:
        return float(f(*[env[a] for a in args]))
    except KeyError as ex:
        raise KeyError("expression %r needs %s, which is neither an input "
                       "(dist/clearance/rel_speed) nor a parameter of this "
                       "algorithm -- add it with algo.set_param(%s, ...)"
                       % (text, ex.args[0], ex.args[0]))


class Scene:
    dims: int = 2

    def resolve(self, source: str, agent) -> List:
        raise NotImplementedError

    def param(self, name: Optional[str]) -> Optional[float]:
        raise NotImplementedError

    def position(self, obj) -> Vec:
        return tuple(obj.position)

    def velocity(self, obj) -> Vec:
        v = getattr(obj, "velocity", None)
        return tuple(v) if v is not None else tuple([0.0] * self.dims)

    def clearance(self, agent, other, dist: float) -> float:
        return dist - float(getattr(agent, "size", 0.0) or 0.0)

    def centroid(self) -> Vec:
        raise NotImplementedError("this algorithm has a centroid-anchored term")

    def gate(self, name: str, agent) -> bool:
        return True

    def role(self, agent) -> str:
        return "agent"


def shill_velocity(term: TermIR, agent, other, scene: Scene,
                   params: Dict[str, float]) -> Vec:
    speed = float(params.get(term.signal_speed) or 0.0)
    away = vunit(vsub(scene.position(agent), scene.position(other)))
    return vsub(vscale(away, speed), scene.velocity(agent))


def _features(agent, other, scene: Scene, term: Optional[TermIR] = None,
              params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    ap, op = scene.position(agent), scene.position(other)
    d = vnorm(vsub(op, ap))
    av, ov = scene.velocity(agent), scene.velocity(other)
    out = {"dist": d,
           "clearance": scene.clearance(agent, other, d),
           "rel_speed": vnorm(vsub(ov, av))}
    if term is not None and term.direction == "shill":
        out["shill_speed"] = vnorm(
            shill_velocity(term, agent, other, scene, params or {}))
    return out


def _angle_between(a: Vec, b: Vec) -> float:
    ma, mb = vnorm(a), vnorm(b)
    if ma == 0.0 or mb == 0.0:
        return 0.0
    c = sum(x * y for x, y in zip(a, b)) / (ma * mb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def narrow(members: List, term: TermIR, agent, scene: Scene) -> List:
    m = term.membership
    rng, fov, cap = scene.param(m.range_param), scene.param(m.fov_param), \
        scene.param(m.cap_param)
    out = []
    ap = scene.position(agent)
    for o in members:
        diff = vsub(scene.position(o), ap)
        if rng is not None and vnorm(diff) > rng:
            continue
        if fov is not None and _angle_between(scene.velocity(agent) or diff,
                                              diff) > fov:
            continue
        out.append(o)
    if cap is not None and cap > 0 and len(out) > cap:
        out.sort(key=lambda o: vnorm(vsub(scene.position(o), ap)))
        out = out[:int(round(cap))]
    return out


def fires(term: TermIR, env: Dict[str, float]) -> bool:
    a = term.activation
    if a.kind == "always":
        return True
    thr = eval_expr(a.threshold, env)
    feat = env[a.feature or "dist"]
    if a.kind == "below":
        return feat <= thr
    if a.kind == "above":
        return feat > thr
    raise NotImplementedError("activation kind %r" % a.kind)


def direction(term: TermIR, agent, other, scene: Scene,
              params: Optional[Dict[str, float]] = None) -> Vec:
    ap = scene.centroid() if term.anchor == "centroid" else scene.position(agent)
    op = scene.position(other)
    d = term.direction
    if d == "shill":
        return vunit(shill_velocity(term, agent, other, scene, params or {}))
    if d == "toward":
        return vunit(vsub(op, ap))
    if d == "away_unit":
        return vunit(vsub(ap, op))
    if d == "away_raw":
        return vsub(ap, op)
    if d == "match":
        if term.signal == "velocity":
            return vsub(scene.velocity(other), scene.velocity(agent))
        return vsub(op, ap)
    if d == "friction":
        return vunit(vsub(scene.velocity(other), scene.velocity(agent)))
    raise NotImplementedError("direction %r" % d)


def _bound_values(b: BoundIR) -> Tuple[Optional[float], float]:
    hi, lo = None, 0.0
    for name, v in b.params.items():
        if v is None:
            continue
        if "min" in name:
            lo = float(v)
        else:
            hi = float(v) if hi is None else min(hi, float(v))
    return hi, lo


def apply_bound(v: Vec, b: BoundIR) -> Vec:
    if b.kind in ("clamp_speed", "clamp_accel"):
        hi, lo = _bound_values(b)
        return limit_magnitude(v, hi, lo)
    if b.kind == "renormalize":
        target = 1.0
        for x in b.params.values():
            target *= float(x)
        return vscale(vunit(v), target) if vnorm(v) else v
    raise NotImplementedError("bound kind %r" % b.kind)


AGGREGATORS = ("sum", "mean", "nearest")


def term_vector(term: TermIR, agent, scene: Scene,
                params: Dict[str, float]) -> Vec:
    n = scene.dims
    if any(not scene.gate(g, agent) for g in term.gates):
        return vzero(n)
    here = scene.role(agent)
    if here != "agent" and term.role not in ("", "agent", here):
        return vzero(n)
    aggregator = term.aggregator or "sum"
    if aggregator not in AGGREGATORS:
        raise NotImplementedError(
            "term %r declares aggregator %r, which this runtime does not "
            "implement (have: %s).  Executing it as a sum would run a "
            "different law than the one recorded."
            % (term.name, aggregator, ", ".join(AGGREGATORS)))
    members = narrow(scene.resolve(term.source, agent), term, agent, scene)
    if aggregator == "nearest" and members:
        ap = scene.position(agent)
        members = [min(members,
                       key=lambda o: vnorm(vsub(scene.position(o), ap)))]
    acc = vzero(n)
    for o in members:
        env = dict(params)
        env.update(_features(agent, o, scene, term, params))
        if not fires(term, env):
            continue
        acc = vadd(acc, vscale(direction(term, agent, o, scene, params),
                               eval_expr(term.magnitude, env)))
    if aggregator == "mean":
        acc = vscale(acc, 1.0 / len(members)) if members else vzero(n)
    if term.bound is not None:
        acc = apply_bound(acc, term.bound)
    return vscale(acc, eval_expr(term.weight, params))


class Controller:
    def __init__(self, algo: AlgorithmIR, dims: int = 2):
        self.algo = algo
        self.dims = dims

    def params(self) -> Dict[str, float]:
        return {k: v["value"] for k, v in self.algo.params.items()
                if v.get("value") is not None}

    def force(self, agent, scene: Scene) -> Vec:
        p = self.params()
        total = vzero(self.dims)
        for t in self.algo.terms:
            total = vadd(total, term_vector(t, agent, scene, p))
        return total

    def command(self, agent, scene: Scene) -> Vec:
        v = self.force(agent, scene)
        if self.algo.base == "add_velocity":
            v = vadd(scene.velocity(agent), v)
        for b in self.algo.global_bounds:
            v = apply_bound(v, b)
        return v

    def per_term(self, agent, scene: Scene) -> Dict[str, Vec]:
        p = self.params()
        return {t.name: term_vector(t, agent, scene, p) for t in self.algo.terms}
