from __future__ import annotations

from typing import Dict, List

from .loop import SELF
from .sexpr import (S, Add, AndS, Clamp, Cmp, Cond, Const, Count, Dom, Fn, Idx, Mul, NotS, Obj,
                    OrS, Pow, Prop, Reduce, Sym, Unk, Vec)

SCHEMA = "swarmmodeling.exact-law.v1"

FUNCTIONS = {"norm", "unit", "sqrt", "exp", "abs", "dot", "min", "max", "cos", "sin", "tan",
             "acos", "asin", "atan2", "log", "floor", "ceil", "mod", "sign", "sum"}


class NotExact(ValueError):
    pass


def _root(domain: Dom) -> str:
    while isinstance(domain.base, Dom):
        domain = domain.base
    return domain.name


class _Graph:
    def __init__(self):
        self.nodes: List[Dict] = []
        self.index: Dict[int, int] = {}
        self.keep = []

    def add(self, node: Dict) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def __call__(self, s: S) -> int:
        hit = self.index.get(id(s))
        if hit is not None:
            return hit
        self.keep.append(s)
        node = self.build(s)
        i = self.add(node)
        self.index[id(s)] = i
        return i

    def domain(self, d: Dom) -> Dict:
        if d.elt is not None:
            raise NotExact("a mapped collection is not a domain: %r" % (d,))
        out = {"collection": _root(d), "vars": sorted({x.var for x in _chain(d) if x.var})}
        for name in ("guard", "cap", "key"):
            value = getattr(d, name)
            out[name] = self(value) if value is not None else None
        if isinstance(d.base, Dom) and d.base.guard is not None:
            out["base_guard"] = self(d.base.guard)
        return out

    def build(self, s: S) -> Dict:
        if isinstance(s, Const):
            return {"op": "const", "value": float(s.v)}
        if isinstance(s, Sym):
            if s.kind == "param":
                return {"op": "param", "name": s.name}
            if s.kind == "free":
                return {"op": "input", "name": s.name}
            raise NotExact("symbol %r" % (s,))
        if isinstance(s, Obj):
            if s == SELF:
                return {"op": "self"}
            if s.domain:
                return {"op": "element", "var": s.name, "collection": s.domain}
            raise NotExact("object %r" % (s,))
        if isinstance(s, Prop):
            if isinstance(s.base, Sym) and s.base.kind == "free":
                return {"op": "input", "name": "%s.%s" % (s.base.name, s.name)}
            return {"op": "field", "name": s.name, "args": [self(s.base)]}
        if isinstance(s, Vec):
            return {"op": "vec", "args": [self(x) for x in s.items]}
        if isinstance(s, Add):
            return {"op": "add", "args": [self(x) for x in s.parts]}
        if isinstance(s, Mul):
            return {"op": "mul", "args": [self(x) for x in s.parts]}
        if isinstance(s, Pow):
            return {"op": "pow", "args": [self(s.base), self(s.exp)]}
        if isinstance(s, Idx):
            return {"op": "index", "args": [self(s.base), self(s.i)]}
        if isinstance(s, Fn):
            if s.name not in FUNCTIONS:
                raise NotExact("function %s" % s.name)
            return {"op": "fn", "name": s.name, "args": [self(x) for x in s.args]}
        if isinstance(s, Cmp):
            return {"op": "cmp", "cmp": s.op, "args": [self(s.left), self(s.right)]}
        if isinstance(s, AndS):
            return {"op": "and", "args": [self(x) for x in s.parts]}
        if isinstance(s, OrS):
            return {"op": "or", "args": [self(x) for x in s.parts]}
        if isinstance(s, NotS):
            return {"op": "not", "args": [self(s.x)]}
        if isinstance(s, Cond):
            return {"op": "select", "args": [self(s.test), self(s.then), self(s.orelse)]}
        if isinstance(s, Clamp):
            return {"op": "clamp", "exact": bool(s.exact),
                    "args": [self(s.x), self(s.hi) if s.hi is not None else -1,
                             self(s.lo) if s.lo is not None else -1]}
        if isinstance(s, Reduce):
            return {"op": "reduce", "reduce": s.op, "var": s.var, "domain": self.domain(s.domain),
                    "guard": self(s.guard) if s.guard is not None else None,
                    "args": [self(s.body)]}
        if isinstance(s, Count):
            return {"op": "count", "var": s.var, "domain": self.domain(s.domain),
                    "guard": self(s.guard) if s.guard is not None else None, "args": []}
        if isinstance(s, Unk):
            raise NotExact("opaque value %s" % s.tag)
        raise NotExact("value %s" % type(s).__name__)


def _chain(d: Dom):
    while isinstance(d, Dom):
        yield d
        d = d.base


def exact_law(extraction) -> Dict:
    command = extraction.raw_command()
    if command is None:
        raise NotExact("no command")
    graph = _Graph()
    root = graph(command)
    params = {n.get("name"): extraction.result.params[n["name"]] for n in graph.nodes
              if n["op"] == "param" and n["name"] in extraction.result.params}
    missing = sorted({n["name"] for n in graph.nodes if n["op"] == "param"} - params.keys())
    if missing:
        raise NotExact("parameters without values: %s" % ", ".join(missing))
    names, loop = extraction.names, extraction.loop
    agents = next((c for c, s in names.sources.items() if s == "agents"), None)
    return {
        "schema": SCHEMA, "target_key": extraction.algorithm.target_key,
        "dimensions": extraction.dims, "nodes": graph.nodes, "command": root,
        "parameters": params,
        "inputs": sorted({n["name"] for n in graph.nodes if n["op"] == "input"}),
        "collections": sorted({n["domain"]["collection"] for n in graph.nodes if "domain" in n}),
        "self_collection": agents,
        "plant": {
            "integrates": loop.written,
            "integration": "displacement" if loop.kind == "state" and
            names.fields.get(loop.written) == "position" else "velocity" if loop.written else None,
            "update_order": extraction.update_order(),
        },
    }
