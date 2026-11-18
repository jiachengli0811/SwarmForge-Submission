from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from .config import body_frame
from .dynamics import Simulator, World
from .laws import norm
from .record import Record
from .state import State

CHUNK = 20000


def plant_step(config) -> float:
    plant = config.physics.plant
    if plant.integration == "displacement":
        return float(plant.max_step)
    return float(config.state.drone_speed.hi) * config.physics.dt


def isotropic(config) -> bool:
    return getattr(config.physics.geometry, "peers", "point") != "square"


def _joint(w, dt: float) -> np.ndarray:
    b = w.positions.shape[0]
    return np.concatenate([w.positions.reshape(b, -1),
                           dt * w.velocities.reshape(b, -1)], axis=1)


class _Loop:
    def __init__(self, record: Record, config):
        self.sim = Simulator(record, config)
        self.dt = float(config.physics.dt)

    def world(self, state: State) -> World:
        w = World.of(state)
        w.obstacle_velocities = np.zeros_like(w.obstacle_velocities)
        return w


def _match(xs: np.ndarray, xo: np.ndarray, m: int) -> np.ndarray:
    b = xs.shape[0]
    ps = np.concatenate([xs[:, :2 * m].reshape(b, m, 2),
                         xs[:, 2 * m:].reshape(b, m, 2)], axis=2)
    po = np.concatenate([xo[:, :2 * m].reshape(b, m, 2),
                         xo[:, 2 * m:].reshape(b, m, 2)], axis=2)
    cost = np.sum((ps[:, :, None] - po[:, None]) ** 2, axis=-1)
    out = np.full((b, m), -1)
    rows = np.arange(b)
    for _ in range(m):
        i, j = np.unravel_index(cost.reshape(b, -1).argmin(axis=1), (m, m))
        out[rows, i] = j
        cost[rows, i, :] = np.inf
        cost[rows, :, j] = np.inf
    return out


def _action(loop: _Loop, x0: np.ndarray, state: State, segments: int,
            step_drift: Callable[[State], np.ndarray]) -> np.ndarray:
    b, m = state.positions.shape[:2]
    dg = (_joint(state, loop.dt) - x0) / segments
    step = np.sqrt(np.sum(dg * dg, axis=1))
    total = np.zeros(b)
    for k in range(segments):
        x = x0 + (k + 0.5) * dg
        mid = state.copy()
        mid.positions = x[:, :2 * m].reshape(b, m, 2)
        mid.velocities = x[:, 2 * m:].reshape(b, m, 2) / loop.dt
        d = step_drift(mid)
        total += np.sqrt(np.sum(d * d, axis=1)) * step - np.sum(d * dg, 1)
    return total


def _commanded_drift(loop: _Loop) -> Callable[[State], np.ndarray]:
    sim = loop.sim
    sync = sim.physics.plant.update_order == "synchronous"

    def drift(state: State) -> np.ndarray:
        w = loop.world(state)
        m = w.positions.shape[1]
        roles = sim.roles(m)
        if not sim.leader_formation:
            w.targets[:] = w.target[:, None, :]
        else:
            w.targets[:, 0] = w.target
        before = _joint(w, loop.dt)
        if sync:
            view = w.view()
            commands = [sim.controller.command(i, view, roles[i])
                        for i in range(m)]
            for i, c in enumerate(commands):
                sim._apply(w, i, c)
        else:
            for i in range(m):
                if sim.leader_formation and i == 1:
                    sim._slots(w)
                sim._apply(w, i, sim.controller.command(i, w.view(),
                                                        roles[i]))
        d = _joint(w, loop.dt) - before
        return np.where(np.isfinite(d), d, 0.0)
    return drift


BODY, PAIR, TARGET = "body", "pair", "target"


def pin_kind(model) -> str:
    from .reach import AgentReach, ObstacleReach
    if isinstance(model.reach, ObstacleReach):
        return BODY
    if isinstance(model.reach, AgentReach):
        return PAIR
    return TARGET


def rotates(config) -> bool:
    static = config.state.static
    walls = static is not None and bool(static.wall_share)
    return (isotropic(config) and not config.physics.square_bodies
            and not walls)


def surface_point(point: np.ndarray, centre: np.ndarray, radius: np.ndarray,
                  axis: np.ndarray, square: bool) -> np.ndarray:
    offset = point - centre
    if square:
        local, half, frame = body_frame(offset, radius, axis)
        local = np.clip(local, -half, half)
        if frame is None:
            return centre + local
        along, across = frame
        return centre + local[:, :1] * along + local[:, 1:] * across
    d = norm(offset)
    unit = offset / np.maximum(d, 1e-12)[:, None]
    return np.where((d > radius)[:, None], centre + radius[:, None] * unit,
                    point)


@dataclass
class Sources:
    kind: str
    positions: np.ndarray
    velocities: np.ndarray
    origin: np.ndarray
    target: np.ndarray
    route: np.ndarray
    mission: np.ndarray
    tick: np.ndarray

    @property
    def size(self) -> int:
        return int(self.positions.shape[0])


def bank_pin(state: State, config, kind: str) -> np.ndarray:
    if kind == TARGET:
        return state.target.copy()
    b, m = state.positions.shape[:2]
    rows = np.arange(b)
    p = state.positions
    if kind == PAIR:
        d = norm(p[:, :, None] - p[:, None])
        d[:, np.arange(m), np.arange(m)] = np.inf
        i, j = np.unravel_index(d.reshape(b, -1).argmin(axis=1), (m, m))
        return 0.5 * (p[rows, i] + p[rows, j])
    if state.obstacles.shape[1] == 0:
        raise ValueError("a body contact's state holds no body")
    gap = config.physics.body_gap(p[:, :, None], state.obstacles[:, None],
                                  state.obstacle_radii[:, None],
                                  state.obstacle_axes[:, None])
    i, k = np.unravel_index(gap.reshape(b, -1).argmin(axis=1), gap.shape[1:])
    return surface_point(p[rows, i], state.obstacles[rows, k],
                         state.obstacle_radii[rows, k],
                         state.obstacle_axes[rows, k],
                         config.physics.square_bodies)


def _turn(kind: str, origin, target, route, positions, turn: bool
          ) -> np.ndarray:
    b = origin.shape[0]
    if not turn:
        return np.repeat(np.eye(2)[None], b, axis=0)
    if kind == TARGET:
        ray = positions.mean(axis=1) - target
        fallback = -route
    else:
        ray = target - origin
        fallback = route
    fallback = np.where(norm(fallback)[:, None] > 0, fallback,
                        np.array([[1.0, 0.0]]))
    ray = np.where(norm(ray)[:, None] > 1e-12, ray, fallback)
    c, s = (ray / norm(ray)[:, None]).T
    return np.stack([np.stack([c, s], 1), np.stack([-s, c], 1)], 1)


def _features(positions, velocities, origin, rotation, dt: float,
              seated: bool) -> np.ndarray:
    p = np.einsum("bij,bmj->bmi", rotation, positions - origin[:, None])
    v = np.einsum("bij,bmj->bmi", rotation, velocities)
    if not seated:
        order = np.argsort(norm(p), axis=1, kind="stable")[:, :, None]
        p = np.take_along_axis(p, order.repeat(2, 2), 1)
        v = np.take_along_axis(v, order.repeat(2, 2), 1)
    b = p.shape[0]
    return np.concatenate([p.reshape(b, -1), dt * v.reshape(b, -1)], axis=1)


def _moved(sources: Sources, rows: np.ndarray, rotation: np.ndarray,
           origin: np.ndarray, back: np.ndarray, dt: float) -> np.ndarray:
    turn = np.einsum("bij,bjk->bik", back, rotation)
    p = np.einsum("bij,bmj->bmi", turn,
                  sources.positions[rows] - sources.origin[rows][:, None])
    p = p + origin[:, None]
    v = np.einsum("bij,bmj->bmi", turn, sources.velocities[rows])
    b = rows.size
    return np.concatenate([p.reshape(b, -1), dt * v.reshape(b, -1)], axis=1)


def _relabel(xs: np.ndarray, x: np.ndarray, m: int) -> np.ndarray:
    b = x.shape[0]
    take = _match(xs, x, m)[:, :, None].repeat(2, 2)
    p = np.take_along_axis(x[:, :2 * m].reshape(b, m, 2), take, 1)
    q = np.take_along_axis(x[:, 2 * m:].reshape(b, m, 2), take, 1)
    return np.concatenate([p.reshape(b, -1), q.reshape(b, -1)], axis=1)


def effort_from(record: Record, config, state: State, sources: Sources, *,
                candidates: int, coarse: int, segments: int) -> np.ndarray:
    if sources.size == 0:
        raise ValueError("the record's missions hold no %s window"
                         % sources.kind)
    m = state.positions.shape[1]
    if sources.positions.shape[1] != m:
        raise ValueError("the sources hold %d drones, the states %d"
                         % (sources.positions.shape[1], m))
    loop = _Loop(record, config)
    dt = loop.dt
    seated = config.physics.targets.kind == "leader_formation"
    turn = rotates(config)
    kind = sources.kind
    own = _turn(kind, sources.origin, sources.target, sources.route,
                sources.positions, turn)
    tree = cKDTree(_features(sources.positions, sources.velocities,
                             sources.origin, own, dt, seated))
    k = max(1, min(int(candidates), sources.size))
    drift = _commanded_drift(loop)
    out = np.empty(state.size)
    for lo in range(0, state.size, CHUNK):
        part = np.arange(lo, min(lo + CHUNK, state.size))
        x = state.take(part)
        origin = bank_pin(x, config, kind)
        rotation = _turn(kind, origin, x.target, x.route, x.positions, turn)
        back = np.transpose(rotation, (0, 2, 1))
        _, near = tree.query(_features(x.positions, x.velocities, origin,
                                       rotation, dt, seated), k)
        near = np.asarray(near).reshape(part.size, k)
        x_s = _joint(x, dt)
        best = np.full(part.size, np.inf)
        best_x = np.zeros((part.size, 4 * m))
        for c in range(k):
            x0 = _moved(sources, near[:, c], own[near[:, c]], origin, back,
                        dt)
            if not seated:
                x0 = _relabel(x_s, x0, m)
            e = _action(loop, x0, x, coarse, drift)
            better = e < best
            best[better] = e[better]
            best_x[better] = x0[better]
        out[part] = _action(loop, best_x, x, segments, drift)
    return out


def second_order(config) -> bool:
    plant = config.physics.plant
    return plant.integration == "velocity" and plant.max_accel is not None


def _accelerations(dp: np.ndarray, v0: np.ndarray, vs: np.ndarray,
                   ticks: np.ndarray, dt: float, question=None):
    t = ticks[:, None, None].astype(float)
    c1 = (vs - v0) / dt
    c2 = (dp - t * dt * v0) / dt ** 2
    s1 = t * (t + 1) / 2
    s2 = t * (t + 1) * (2 * t + 1) / 6
    det = t * s2 - s1 * s1
    single = det == 0
    safe = np.where(single, 1.0, det)
    alpha = np.where(single, c1, (s2 * c1 - s1 * c2) / safe)
    beta = np.where(single, 0.0, (t * c2 - s1 * c1) / safe)
    met = np.ones(ticks.shape, dtype=bool)
    one = single[:, 0, 0]
    if one.any():
        miss = _largest(dt * v0 + dt ** 2 * c1 - dp, question)
        met = np.where(one, miss <= 1e-9 * (1 + _largest(dp, question)),
                       met)
    return alpha, beta, met


def _apply(question, x: np.ndarray) -> np.ndarray:
    if question is None:
        return x
    return np.einsum("brm,bmk->brk", question[0], x)


def _largest(x: np.ndarray, question) -> np.ndarray:
    return np.max(norm(_apply(question, x)), axis=1)


def _within(alpha: np.ndarray, beta: np.ndarray, ticks: np.ndarray,
            a_max: float, question=None) -> np.ndarray:
    t = ticks[:, None, None]
    tol = a_max * (1 + 1e-6)
    return ((_largest(alpha + beta * t, question) <= tol)
            & (_largest(alpha + beta, question) <= tol))


def _override(loop: _Loop, config, x0: np.ndarray, state: State,
              ticks: np.ndarray, points: int,
              drift: Callable[[State], np.ndarray],
              question=None) -> np.ndarray:
    b, m = state.positions.shape[:2]
    dt = loop.dt
    plant = config.physics.plant
    p0 = x0[:, :2 * m].reshape(b, m, 2)
    v0 = x0[:, 2 * m:].reshape(b, m, 2) / dt
    ps, vs = state.positions, state.velocities
    second = second_order(config)
    ticks = np.maximum(np.round(ticks), 1).astype(int)
    T = ticks[:, None, None].astype(float)
    ok = np.ones(b, dtype=bool)
    if second:
        a_max = float(plant.max_accel)
        top = float(plant.max_step) / dt
        alpha, beta, met = _accelerations(ps - p0, v0, vs, ticks, dt,
                                          question)
        ok &= met & _within(alpha, beta, ticks, a_max, question)
        scale = a_max
    else:
        step = (ps - p0) / T
        scale = float(plant.max_step)
    n = np.minimum(ticks, points)
    total = np.zeros(b)
    involved = (None if question is None
                else np.any(question[0] != 0.0, axis=1))
    for k in range(int(n.max())):
        live = k < n
        tick = np.where(n == ticks, k,
                        np.floor((k + 0.5) * ticks / np.maximum(n, 1)))
        tt = np.where(live, tick, 0)[:, None, None].astype(float)
        mid = state.copy()
        if second:
            s_a = tt * alpha + beta * (tt * T - tt * (tt - 1) / 2)
            s_p = (alpha * tt * (tt + 1) / 2
                   + beta * ((T - tt) * tt * (tt + 1) / 2
                             + tt * (tt + 1) * (2 * tt + 1) / 6))
            mid.velocities = v0 + dt * s_a
            mid.positions = p0 + dt * (tt * v0 + dt * s_p)
            need = alpha + beta * (T - tt)
            speed = norm(mid.velocities)
            if involved is not None:
                speed = np.where(involved, speed, 0.0)
            fast = np.max(speed, axis=1) > top * (1 + 1e-6)
            ok &= ~(live & fast)
        else:
            mid.positions = p0 + tt * step
            mid.velocities = np.where(tt > 0, step / dt, v0)
            need = step
        d = drift(mid)
        law = (d[:, 2 * m:].reshape(b, m, 2) / dt ** 2 if second
               else d[:, :2 * m].reshape(b, m, 2))
        cost = np.sum(_apply(question, need - law) ** 2, axis=(1, 2))
        if question is not None:
            cost = cost * question[1]
        total += np.where(live, cost / scale ** 2, 0.0)
    return np.where(ok, total * ticks / n, np.inf)


_SCAN_RATIO, _SCAN_STEPS = 1.25, 64


def _shortest(config, x0: np.ndarray, state: State, dt: float,
              question=None) -> np.ndarray:
    b, m = state.positions.shape[:2]
    plant = config.physics.plant
    p0 = x0[:, :2 * m].reshape(b, m, 2)
    dp = state.positions - p0
    move = _largest(dp, question) / float(plant.max_step)
    out = np.maximum(1.0, np.ceil(move - 1e-9))
    if not second_order(config):
        return out
    a_max = float(plant.max_accel)
    v0 = x0[:, 2 * m:].reshape(b, m, 2) / dt
    vs = state.velocities
    turn = _largest(vs - v0, question) / (a_max * dt)
    ticks = np.maximum(out, np.ceil(turn - 1e-9))
    cap = float(config.mission.ticks)
    found = np.full(b, np.inf)
    for _ in range(_SCAN_STEPS):
        alpha, beta, met = _accelerations(dp, v0, vs, ticks, dt, question)
        ok = met & _within(alpha, beta, ticks, a_max, question) \
            & (ticks <= cap)
        found = np.where(np.isinf(found) & ok, ticks, found)
        if np.all(np.isfinite(found) | (ticks > cap)):
            break
        ticks = np.maximum(ticks + 1, np.ceil(ticks * _SCAN_RATIO))
    return found


def anchor_drones(state: State, config, kind: str) -> np.ndarray:
    b, m = state.positions.shape[:2]
    out = np.zeros((b, m), dtype=bool)
    rows = np.arange(b)
    p = state.positions
    if kind == TARGET or m == 1:
        out[:] = True
    elif kind == PAIR:
        d = norm(p[:, :, None] - p[:, None])
        d[:, np.arange(m), np.arange(m)] = np.inf
        i, j = np.unravel_index(d.reshape(b, -1).argmin(axis=1), (m, m))
        out[rows, i] = out[rows, j] = True
    else:
        gap = config.physics.body_gap(p[:, :, None], state.obstacles[:, None],
                                      state.obstacle_radii[:, None],
                                      state.obstacle_axes[:, None])
        out[rows, gap.min(axis=2).argmin(axis=1)] = True
    return out


def question_matrix(state: State, config, kind: str, mode: str):
    b, m = state.positions.shape[:2]
    mask = anchor_drones(state, config, kind)
    if kind == TARGET or m == 1:
        return np.full((b, 1, m), 1.0 / m), float(m)
    if kind == PAIR:
        rows = np.arange(b)
        first = mask.argmax(axis=1)
        second = np.where(mask & (np.arange(m)[None] != first[:, None]),
                          np.arange(m)[None], -1).max(axis=1)
        if mode == "question":
            q = np.zeros((b, 1, m))
            q[rows, 0, first] = 0.5
            q[rows, 0, second] = -0.5
            return q, 1.0
        q = np.zeros((b, 2, m))
        q[rows, 0, first] = 1.0
        q[rows, 1, second] = 1.0
        return q, 1.0
    q = np.zeros((b, 1, m))
    q[np.arange(b), 0, mask.argmax(axis=1)] = 1.0
    return q, 1.0


def _start_in_contact(config, x0: np.ndarray, state: State) -> np.ndarray:
    b, m = state.positions.shape[:2]
    if state.obstacles.shape[1] == 0:
        return np.zeros(b, dtype=bool)
    p0 = x0[:, :2 * m].reshape(b, m, 2)
    gap = config.physics.body_gap(p0[:, :, None], state.obstacles[:, None],
                                  state.obstacle_radii[:, None],
                                  state.obstacle_axes[:, None])
    return np.any(gap < 0.0, axis=(1, 2))


def override_from(record: Record, config, state: State, sources: Sources, *,
                  candidates: int, durations: int, coarse: int,
                  points: int, focus=None, pin: str = "target"
                  ) -> np.ndarray:
    if sources.size == 0:
        raise ValueError("the record reaches no %s window" % sources.kind)
    m = state.positions.shape[1]
    if sources.positions.shape[1] != m:
        raise ValueError("the sources hold %d drones, the states %d"
                         % (sources.positions.shape[1], m))
    loop = _Loop(record, config)
    dt = loop.dt
    seated = config.physics.targets.kind == "leader_formation"
    turn = rotates(config)
    kind = sources.kind
    centre = pin == "swarm" and kind == TARGET
    if centre:
        sources = replace(sources, origin=sources.positions.mean(axis=1))
    own = _turn(kind, sources.origin, sources.target, sources.route,
                sources.positions, turn)
    tree = cKDTree(_features(sources.positions, sources.velocities,
                             sources.origin, own, dt, seated))
    k = max(1, min(int(candidates), sources.size))
    drift = _commanded_drift(loop)
    cap = float(config.mission.ticks)
    out = np.empty(state.size)
    for lo in range(0, state.size, CHUNK):
        part = np.arange(lo, min(lo + CHUNK, state.size))
        x = state.take(part)
        origin = (x.positions.mean(axis=1) if centre
                  else bank_pin(x, config, kind))
        rotation = _turn(kind, origin, x.target, x.route, x.positions, turn)
        back = np.transpose(rotation, (0, 2, 1))
        _, near = tree.query(_features(x.positions, x.velocities, origin,
                                       rotation, dt, seated), k)
        near = np.asarray(near).reshape(part.size, k)
        x_s = _joint(x, dt)
        mode = "anchor" if focus is True else focus
        question = question_matrix(x, config, kind, mode) if mode else None
        best = np.full(part.size, np.inf)
        best_x = np.zeros((part.size, 4 * m))
        best_t = np.ones(part.size)
        for c in range(k):
            x0 = _moved(sources, near[:, c], own[near[:, c]], origin, back,
                        dt)
            if not seated:
                x0 = _relabel(x_s, x0, m)
            here = np.all(np.abs(x0 - x_s) <= 1e-9, axis=1)
            blocked = _start_in_contact(config, x0, x) & ~here
            first = _shortest(config, x0, x, dt, question)
            reachable = np.isfinite(first)
            first = np.where(reachable, first, 1.0)
            for j in range(max(1, int(durations))):
                ticks = first * 2 ** j
                e = _override(loop, config, x0, x, ticks, coarse, drift,
                              question)
                e = np.where(here, 0.0, e)
                e = np.where(blocked | ~reachable | (ticks > cap), np.inf, e)
                better = e < best
                best[better] = e[better]
                best_x[better] = x0[better]
                best_t[better] = ticks[better]
        fine = _override(loop, config, best_x, x, best_t, points, drift,
                         question)
        here = np.all(np.abs(best_x - x_s) <= 1e-9, axis=1)
        out[part] = np.where(np.isfinite(best),
                             np.where(here, 0.0, fine), np.inf)
    return out


__all__ = ["BODY", "CHUNK", "PAIR", "Sources", "TARGET", "anchor_drones",
           "bank_pin", "effort_from", "isotropic", "override_from",
           "pin_kind", "question_matrix", "plant_step", "rotates",
           "second_order", "surface_point"]
