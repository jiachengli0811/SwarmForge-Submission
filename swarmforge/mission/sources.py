from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..solve.config import FamilyConfig
from ..solve.effort import BODY, PAIR, TARGET, Sources, surface_point
from ..solve.record import Record
from .fly import Flight, fly
from .measure import _Field, surface_distance
from .store import read_case
from .verdict import Oracle, home, judge

Mission = Tuple[Dict[str, Any], int]


def read_missions(paths: Sequence[str]) -> List[Mission]:
    out = []
    for path in paths:
        document, meta = read_case(str(path))
        out.append((document, int(meta.spawn_seed)))
    return out


def fly_missions(config: FamilyConfig, record: Record,
                 missions: Sequence[Mission]) -> List[Flight]:
    oracle = Oracle.for_config(config)
    out = []
    for n, (document, spawn) in enumerate(missions):
        goal = (float(document["goal"][0]), float(document["goal"][1]))
        flight = fly(document, config, record, spawn_seed=spawn, every=1,
                     decides=lambda s, goal=goal: home(s, goal, oracle))
        if not flight.ok:
            raise ValueError("mission %d does not fly: %s" % (n, flight.error))
        out.append(flight)
    return out


def _nearest(apart: np.ndarray, anchors: Sequence[int], room: int
             ) -> List[int]:
    n = apart.shape[0]
    others = np.array([e for e in range(n) if e not in anchors], dtype=int)
    near = apart[list(anchors)][:, others].min(axis=0)
    return [int(e) for e in others[np.argsort(near, kind="stable")][:room]]


def cut(config: FamilyConfig, missions: Sequence[Mission],
        flights: Sequence[Flight], kind: str, drones: int) -> Sources:
    physics = config.physics
    seated = physics.targets.kind == "leader_formation"
    velocity = (physics.plant.integration == "velocity"
                or config.state.drone_speed is not None)
    square = physics.square_bodies
    settle = Oracle.for_config(config).settle_ticks
    m = int(drones)
    rows: Dict[str, List[Any]] = {k: [] for k in (
        "positions", "velocities", "origin", "target", "route", "mission",
        "tick")}

    def keep(n, tick, p, v, window, origin, target, route):
        rows["positions"].append(p[window])
        rows["velocities"].append(v[window])
        rows["origin"].append(origin)
        rows["target"].append(target)
        rows["route"].append(route)
        rows["mission"].append(n)
        rows["tick"].append(tick)

    for n, ((document, _), flight) in enumerate(zip(missions, flights)):
        field = _Field(document, square)
        goal = np.asarray(document["goal"], dtype=float)
        for snapshot in flight.snapshots:
            if snapshot.tick < settle:
                continue
            p = np.asarray(snapshot.positions, dtype=float).reshape(-1, 2)
            count = p.shape[0]
            if count < m or (seated and count != m):
                raise ValueError("a %d-drone window cannot be read from %d "
                                 "drones%s" % (m, count, " in seat order"
                                               if seated else ""))
            v = (np.asarray(snapshot.velocities, dtype=float).reshape(-1, 2)
                 if velocity else np.zeros_like(p))
            target = np.asarray(snapshot.aim if snapshot.aim is not None
                                else goal, dtype=float)
            route = np.asarray(snapshot.route, dtype=float)
            apart = np.linalg.norm(p[:, None] - p[None], axis=-1)
            every = list(range(count))
            if kind == TARGET:
                anchors = [(0,)] if seated else [(i,) for i in every]
                for a in anchors:
                    window = every if seated else (
                        list(a) + _nearest(apart, a, m - 1))
                    keep(n, snapshot.tick, p, v, window, target, target,
                         route)
                continue
            if kind == PAIR:
                apart_own = apart + np.diag(np.full(count, np.inf))
                pairs = sorted({tuple(sorted((i, int(np.argmin(
                    apart_own[i]))))) for i in every})
                for a in pairs:
                    window = every if seated else (
                        list(a) + _nearest(apart, a, m - 2))
                    keep(n, snapshot.tick, p, v, window,
                         0.5 * (p[a[0]] + p[a[1]]), target, route)
                continue
            centres, radii, gap, _ = field.seen_from(p)
            movers = snapshot.movers
            if movers:
                mc = np.array([c for c, _, _ in movers],
                              dtype=float).reshape(-1, 2)
                mr = np.array([r for _, r, _ in movers], dtype=float)
                centres = np.concatenate([centres, np.broadcast_to(
                    mc[None], (count,) + mc.shape)], axis=1)
                radii = np.concatenate([radii, np.broadcast_to(
                    mr[None], (count, mr.size))], axis=1)
                gap = np.concatenate([gap, surface_distance(
                    p[:, None], mc[None], mr[None], square)], axis=1)
            if gap.shape[1] == 0:
                continue
            k = np.argmin(gap, axis=1)
            pins = surface_point(p, centres[np.arange(count), k],
                                 radii[np.arange(count), k],
                                 np.zeros((count, 2)), square)
            for i in every:
                window = every if seated else (
                    [i] + _nearest(apart, (i,), m - 1))
                keep(n, snapshot.tick, p, v, window, pins[i], target, route)

    def stack(key, tail, dtype=float):
        return np.array(rows[key], dtype=dtype).reshape((-1,) + tail)
    return Sources(kind, stack("positions", (m, 2)),
                   stack("velocities", (m, 2)), stack("origin", (2,)),
                   stack("target", (2,)), stack("route", (2,)),
                   stack("mission", (), int), stack("tick", (), int))


def failure_sources(config: FamilyConfig, record: Record,
                    missions: Sequence[Mission], models: Mapping[str, Any]
                    ) -> Tuple[Dict[str, Sources], List[Tuple[int, str]]]:
    from ..solve.effort import pin_kind
    flights = fly_missions(config, record, missions)
    oracle = Oracle.for_config(config)
    decided = []
    for (document, _), flight in zip(missions, flights):
        goal = (float(document["goal"][0]), float(document["goal"][1]))
        outcome = judge(flight.snapshots, goal=goal, oracle=oracle,
                        completed=True)
        tick = (outcome.tick if outcome.tick is not None
                else flight.snapshots[-1].tick)
        decided.append((int(tick), outcome.verdict.value))
    ticks = np.array([t for t, _ in decided])
    out = {}
    for name, model in models.items():
        every = cut(config, missions, flights, pin_kind(model),
                    int(model.spec.drones))
        keep = every.tick == ticks[every.mission]
        if not keep.any():
            raise ValueError("no %s window at the missions' decisive ticks"
                             % every.kind)
        out[name] = Sources(every.kind, every.positions[keep],
                            every.velocities[keep], every.origin[keep],
                            every.target[keep], every.route[keep],
                            every.mission[keep], every.tick[keep])
    return out, decided


def decisive_ticks(config: FamilyConfig, missions: Sequence[Mission],
                   flights: Sequence[Flight]) -> List[Tuple[int, str]]:
    oracle = Oracle.for_config(config)
    out = []
    for (document, _), flight in zip(missions, flights):
        goal = (float(document["goal"][0]), float(document["goal"][1]))
        outcome = judge(flight.snapshots, goal=goal, oracle=oracle,
                        completed=True)
        tick = (outcome.tick if outcome.tick is not None
                else flight.snapshots[-1].tick)
        out.append((int(tick), outcome.verdict.value))
    return out


def reached_sources(config: FamilyConfig, record: Record,
                    missions: Sequence[Mission], models: Mapping[str, Any]
                    ) -> Tuple[Dict[str, Sources], List[Tuple[int, str]]]:
    from ..solve.effort import pin_kind
    flights = fly_missions(config, record, missions)
    decided = decisive_ticks(config, missions, flights)
    ticks = np.array([t for t, _ in decided])
    cache: Dict[Tuple[str, int], Sources] = {}
    out = {}
    for name, model in models.items():
        key = (pin_kind(model), int(model.spec.drones))
        if key not in cache:
            every = cut(config, missions, flights, *key)
            keep = every.tick <= ticks[every.mission]
            cache[key] = Sources(every.kind, every.positions[keep],
                                 every.velocities[keep], every.origin[keep],
                                 every.target[keep], every.route[keep],
                                 every.mission[keep], every.tick[keep])
        out[name] = cache[key]
    return out, decided


__all__ = ["BODY", "Mission", "PAIR", "TARGET", "cut", "decisive_ticks",
           "failure_sources", "fly_missions", "read_missions",
           "reached_sources"]
