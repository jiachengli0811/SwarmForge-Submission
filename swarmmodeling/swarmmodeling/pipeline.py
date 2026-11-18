from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

from . import arrays, canonical
from .classify import (AlgorithmIR, Context, augment_driver, classify, fix_params,
                       membership_profile)
from .execute import Exec, Result
from .loop import SELF, Loop, find_loop
from .naming import Names, name_variables
from .program import Program
from .sexpr import Fn, Obj, Prop, Sym, Vec, subst
from .tick import CONTROLLED, Tick, find_tick


@dataclass
class Extraction:
    program: Program
    entry: str
    tick: Optional[Tick]
    result: Result
    loop: Loop
    names: Names
    algorithm: AlgorithmIR
    context: Context
    vectorized: bool = False
    raw: tuple = ()
    settings: dict = field(default_factory=dict)
    dims: int = 2
    unknown_fixed: List[str] = field(default_factory=list)

    def profile(self) -> dict:
        algo, loop, names = self.algorithm, self.loop, self.names
        plant = algo.plant or {}
        params = {n: d["value"] for n, d in algo.params.items()}
        step = plant.get("time_step") or (repr(loop.time_step) if loop.time_step is not None else None)
        dt = self.program.argument_value(self.entry, step) if step else None
        velocity = bool(plant) or loop.kind == "rate" or names.fields.get(loop.written) == "velocity" \
            or any(t.signal == "velocity" for t in algo.terms)
        return {
            "schema": "swarmmodeling.controller-profile.v1", "entry": self.entry,
            "dimensions": self.dims, "parameters": params,
            "specialization": dict(self.settings),
            "membership": membership_profile(loop.sink, names, self.context),
            "physics": {"dt": dt, "plant": {
                "integration": "velocity" if velocity else None,
                "update_order": self.update_order(),
                "max_accel": params.get(plant["max_accel"]) if plant else None}},
        }

    def raw_command(self):
        from .execute import _delta_of
        heap, ret = self.raw
        loop = self.loop
        if loop.kind == "return":
            return ret
        value = heap.get(Prop(SELF, loop.written)) if loop.written else None
        if value is None:
            return None
        if loop.kind == "state" and self.names.fields.get(loop.written) == "position":
            return _delta_of(value, Prop(SELF, loop.written))
        return value

    def update_order(self) -> Optional[str]:
        if self.vectorized:
            return "synchronous"
        if self.tick is not None and self.loop.written:
            return "sequential"
        return None

    def explain(self) -> dict:
        loop, names = self.loop, self.names
        return {
            "tick": self.tick.where if self.tick else "the entry alone (no driver loop calls it)",
            "loop": {"output": loop.desc, "reason": loop.reason, "fields": list(loop.fields),
                     "plant": list(loop.plant),
                     "time_step": repr(loop.time_step) if loop.time_step is not None else None,
                     "inputs": loop.inputs},
            "names": {"fields": names.fields, "sources": names.sources,
                      "membership": names.membership, "agent_argument": names.agent,
                      "body_radius": names.body_radius,
                      "unresolved_collections": sorted(names.collections - names.sources.keys()),
                      "reasons": names.reasons},
        }


def _bundle(heap, ret, calls):
    keys = list(heap)
    parts = [heap[k] for k in keys] + ([ret] if ret is not None else [])
    shape = [len(args) for _, args in calls]
    parts += [a for _, args in calls for a in args]
    return keys, Fn("_tick", tuple(parts)), shape


def _unbundle(keys, bundle, shape, calls, has_ret):
    parts = list(bundle.args)
    heap = {k: parts[i] for i, k in enumerate(keys)}
    at = len(keys)
    ret = parts[at] if has_ret else None
    at += 1 if has_ret else 0
    out_calls = []
    for (name, _), n in zip(calls, shape):
        out_calls.append((name, tuple(parts[at:at + n])))
        at += n
    return heap, ret, out_calls


def _rename(heap, ret, calls, mapping, trace=None):
    keys, bundle, shape = _bundle(heap, ret, calls)
    keys = [subst(k, mapping) for k in keys]
    extra = []
    if trace is not None:
        extra = [x for pair in trace.samples for x in pair] + trace.potentials + trace.occupancies
    bundle = subst(Fn("_tick", bundle.args + tuple(extra)), mapping)
    parts = bundle.args[:len(bundle.args) - len(extra)]
    out = _unbundle(keys, Fn("_tick", parts), shape, calls, ret is not None)
    if trace is None:
        return out
    extra = list(bundle.args[len(parts):])
    n = 2 * len(trace.samples)
    samples = list(zip(extra[0:n:2], extra[1:n:2]))
    potentials = extra[n:n + len(trace.potentials)]
    occupancies = extra[n + len(trace.potentials):]
    return out + (replace(trace, samples=samples, potentials=potentials,
                          occupancies=occupancies),)


def _vectorized(ret):
    items = ret.items if isinstance(ret, Vec) else (ret,)
    maps = [v for v in items if arrays.is_map(v)]
    return maps[0] if len(maps) == 1 else None


def extract(files: Sequence[str], entry: str, key: str, *, dims: int = 2,
            use_shortcuts: bool = True, fixed: Sequence[str] = (),
            peers: Optional[str] = None, settings: Optional[dict] = None) -> Extraction:
    program = Program()
    for path in files:
        program.load(path)
    program.finish()
    for name, value in (settings or {}).items():
        program.class_consts[name] = float(value)
    tick = find_tick(program, entry)
    result = Exec(program, use_shortcuts=use_shortcuts, dims=dims).run(entry, tick)
    heap, ret, calls = dict(result.env.heap), result.ret, list(result.actuations)
    trace = result.trace
    drawn_from = tick.collection if tick is not None else None
    if tick is not None:
        heap, ret, calls, trace = _rename(heap, ret, calls,
                                          {Obj(CONTROLLED, domain=tick.collection): SELF}, trace)
    elif ret is not None and _vectorized(ret) is not None:
        column = _vectorized(ret)
        drawn_from = column.name
        heap, ret, calls = _rename(heap, arrays.at(column, SELF), calls, {})
    raw = (dict(heap), ret)
    keys, bundle, shape = _bundle(heap, ret, calls)
    bundle = canonical.rewrite(bundle, result.notes)
    heap, ret, calls = _unbundle(keys, bundle, shape, calls, ret is not None)
    loop = find_loop(heap, ret, calls)
    function, _ = program.functions[entry]
    free = tuple(function.params) if tick is None and function.cls is None else ()
    names = name_variables(program, entry, heap, ret, calls, trace, result.params,
                           loop, drawn_from, free, peers)
    if names.agent:
        heap, ret, calls = _rename(heap, ret, calls, {Sym(names.agent, "free"): SELF})
        raw_heap, raw_ret, _ = _rename(raw[0], raw[1], [], {Sym(names.agent, "free"): SELF})
        raw = (raw_heap, raw_ret)
        loop = find_loop(heap, ret, calls)
        names.time_step = loop.time_step
    context = Context(result.params, result.notes, names, heap)
    algorithm = classify(loop, names, context, key)
    augment_driver(algorithm, program, entry, names)
    unknown = fix_params(algorithm, fixed)
    return Extraction(program, entry, tick, result, loop, names, algorithm, context,
                      vectorized=tick is None and drawn_from is not None, raw=raw,
                      settings=dict(settings or {}), dims=dims, unknown_fixed=unknown)
