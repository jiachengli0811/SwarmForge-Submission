from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..solve import effort as E
from ..solve.record import Record
from ..solve.state import State
from .sources import Mission, cut, decisive_ticks, fly_missions

RESPONSE, DISTANCE, DIRECTION = "response", "distance", "direction"
EFFORTS = (RESPONSE, DISTANCE, DIRECTION)


def keeps_velocity(config) -> bool:
    return (config.physics.plant.integration == "velocity"
            or config.state.drone_speed is not None)


def _authority(config, dt: float) -> float:
    plant = config.physics.plant
    if E.second_order(config):
        return float(plant.max_accel) * dt * dt
    return float(plant.max_step)


def _time_scale(config, dt: float) -> float:
    if E.second_order(config):
        plant = config.physics.plant
        return float(plant.max_step) / dt / float(plant.max_accel)
    return dt


def question_features(positions, velocities, origin, rotation, kind: str,
                      ts: float) -> np.ndarray:
    p = np.einsum("bij,bmj->bmi", rotation, positions - origin[:, None])
    v = np.einsum("bij,bmj->bmi", rotation, velocities) * ts
    order = np.argsort(E.norm(p), axis=1, kind="stable")[..., None]
    if kind == E.TARGET:
        p = np.take_along_axis(p - p.mean(axis=1, keepdims=True),
                               order.repeat(2, 2), 1)
        return np.concatenate([p.reshape(p.shape[0], -1), v.mean(axis=1)],
                              axis=1)
    p = np.take_along_axis(p, order.repeat(2, 2), 1)
    v = np.take_along_axis(v, order.repeat(2, 2), 1)
    if kind == E.BODY:
        return np.concatenate([p[:, 0], v[:, 0]], axis=1)
    r, dv, mv = p[:, 0] - p[:, 1], v[:, 0] - v[:, 1], 0.5 * (v[:, 0]
                                                             + v[:, 1])
    flip = (r[:, 0] < 0) | ((r[:, 0] == 0) & (r[:, 1] < 0))
    r = np.where(flip[:, None], -r, r)
    dv = np.where(flip[:, None], -dv, dv)
    return np.concatenate([r, dv, mv], axis=1)


def flight_states(config, missions: Sequence[Mission], flights,
                  decided) -> List[List[Tuple[np.ndarray, State]]]:
    from .measure import _Windows, surface_distance
    spec = config.model("timeout")
    keep_v = keeps_velocity(config)
    out = []
    for (document, _), flight, (last, _) in zip(missions, flights, decided):
        cutter = _Windows("timeout", config, spec, spec.horizon, document)
        for snap in flight.snapshots:
            if snap.tick > last:
                break
            p = np.asarray(snap.positions, dtype=float).reshape(-1, 2)
            n = p.shape[0]
            v = (np.asarray(snap.velocities, dtype=float).reshape(-1, 2)
                 if keep_v else np.zeros((n, 2)))
            movers = snap.movers
            mc = np.array([c for c, _, _ in movers],
                          dtype=float).reshape(-1, 2)
            mr = np.array([r for _, r, _ in movers], dtype=float)
            mv = np.array([u for _, _, u in movers],
                          dtype=float).reshape(-1, 2)
            sc, sr, ss, wall = cutter.field.seen_from(p)
            frame = (p, v, sc, sr, ss, wall,
                     np.linalg.norm(p[:, None] - sc, axis=-1), mc, mr, mv,
                     np.linalg.norm(p[:, None] - mc[None], axis=-1),
                     surface_distance(p[:, None], mc[None], mr[None],
                                      cutter.square),
                     np.linalg.norm(p[:, None] - p[None], axis=-1),
                     np.asarray(snap.aim if snap.aim is not None
                                else cutter.goal, dtype=float),
                     np.asarray(snap.route, dtype=float))
            cutter._window(snap.tick, tuple(range(n)), None, frame)
        out.append([(np.asarray(g.tick), g.state(ks, kd))
                    for (m, ks, kd), g in cutter.groups.items()])
    return out


def responses(loop, config, states: State, K: int) -> np.ndarray:
    dt = loop.dt
    drift = E._commanded_drift(loop)
    n, m = states.positions.shape[:2]
    d = 4 * m
    b = _authority(config, dt)
    out = np.empty((n, m, 2, K, d), dtype=np.float32)
    nudge = np.zeros((2 * m, d))
    for i in range(2 * m):
        nudge[i, i] = nudge[i, 2 * m + i] = b
    step = max(1, E.CHUNK // (4 * m))
    for lo in range(0, n, step):
        idx = np.arange(lo, min(lo + step, n))
        x = states.take(idx)
        z = E._joint(x, dt) + drift(x)
        z = np.concatenate([z[:, None] + nudge[None],
                            z[:, None] - nudge[None]], axis=1).reshape(-1, d)
        x = states.take(np.repeat(idx, 4 * m))
        for k in range(K):
            zz = z.reshape(idx.size, 2, 2 * m, d)
            out[idx, :, :, k] = (0.5 * (zz[:, 0] - zz[:, 1])
                                 ).reshape(idx.size, m, 2, d)
            if k + 1 < K:
                x.positions = z[:, :2 * m].reshape(-1, m, 2)
                x.velocities = (z[:, 2 * m:] / dt).reshape(-1, m, 2)
                z = z + drift(x)
    return out


def updates(loop, states: State) -> np.ndarray:
    drift = E._commanded_drift(loop)
    n, m = states.positions.shape[:2]
    out = np.empty((n, m, 4))
    for lo in range(0, n, E.CHUNK):
        idx = np.arange(lo, min(lo + E.CHUNK, n))
        d = drift(states.take(idx))
        out[idx, :, :2] = d[:, :2 * m].reshape(-1, m, 2)
        out[idx, :, 2:] = d[:, 2 * m:].reshape(-1, m, 2)
    return out


class Flow:
    def __init__(self, record: Record, config, missions: Sequence[Mission],
                 K: int, effort: str = RESPONSE):
        if effort not in EFFORTS:
            raise ValueError("no flow effort %r (have: %s)"
                             % (effort, ", ".join(EFFORTS)))
        self.config, self.missions, self.K = config, missions, int(K)
        self.effort = effort
        self.loop = E._Loop(record, config)
        self.dt = self.loop.dt
        self.flights = fly_missions(config, record, missions)
        self.decided = decisive_ticks(config, missions, self.flights)
        resp, move, pos = [], [], []
        for groups in flight_states(config, missions, self.flights,
                                    self.decided):
            top = max(int(t.max()) for t, _ in groups) + 1
            m = groups[0][1].positions.shape[1]
            R = (np.zeros((top, m, 2, self.K, 4 * m), dtype=np.float32)
                 if effort == RESPONSE else None)
            U = np.zeros((top, m, 4)) if effort == DIRECTION else None
            P = np.full((top, m, 2), np.nan)
            for ticks, st in groups:
                if R is not None:
                    R[ticks] = responses(self.loop, config, st, self.K)
                if U is not None:
                    U[ticks] = updates(self.loop, st)
                P[ticks] = st.positions
            resp.append(R)
            move.append(U)
            pos.append(P)
        self.offset = np.cumsum([0] + [p.shape[0] for p in pos])
        self.R = np.concatenate(resp) if effort == RESPONSE else None
        self.U = np.concatenate(move) if effort == DIRECTION else None
        self.P = pos

    def windows(self, kind: str, drones: int) -> Tuple[E.Sources, np.ndarray]:
        src = cut(self.config, self.missions, self.flights, kind, drones)
        keep = src.tick <= np.array([t for t, _ in self.decided])[src.mission]
        src = replace(src, **{f: getattr(src, f)[keep] for f in (
            "positions", "velocities", "origin", "target", "route", "mission",
            "tick")})
        if src.size == 0:
            raise ValueError("the record's flights reach no %s window before "
                             "the missions are decided" % kind)
        at = np.array([[int(np.argmin(np.linalg.norm(
            self.P[mi][t] - q, axis=1))) for q in p] for p, mi, t in zip(
                src.positions, src.mission, src.tick)],
            dtype=int).reshape(src.size, -1)
        if (self.config.physics.targets.kind == "leader_formation"
                and kind != E.TARGET):
            m = src.positions.shape[1]
            if kind == E.BODY:
                first = np.argmin(E.norm(src.positions - src.origin[:, None]),
                                  axis=1)[:, None]
            else:
                i, j = np.triu_indices(m, 1)
                mid = 0.5 * (src.positions[:, i] + src.positions[:, j])
                pick = np.argmin(E.norm(mid - src.origin[:, None]), axis=1)
                first = np.stack([i[pick], j[pick]], axis=1)
            rest = np.array([[e for e in range(m) if e not in f]
                             for f in first.tolist()], dtype=int)
            order = np.concatenate([first, rest], axis=1)
            take = order[..., None].repeat(2, 2)
            src = replace(src, positions=np.take_along_axis(src.positions,
                                                            take, 1),
                          velocities=np.take_along_axis(src.velocities,
                                                        take, 1))
            at = np.take_along_axis(at, order, 1)
        return src, at

    def gramians(self, kind: str, src: E.Sources, at: np.ndarray,
                 K: Optional[int] = None, chunk: int = 2048) -> np.ndarray:
        if self.R is None:
            raise ValueError("a %s flow reads no response" % self.effort)
        lags = self.K if K is None else min(int(K), self.K)
        vel = keeps_velocity(self.config)
        S = src.size
        m = self.P[0].shape[1]
        rows = np.arange(S)
        if kind == E.TARGET:
            w = at.shape[1]
            sigma = np.zeros((S, w, m))
            sigma[rows[:, None], np.arange(w)[None], at] = 1.0
            qv = sigma.mean(axis=1, keepdims=True)
            qp = np.zeros((S, 0, m)) if vel else sigma - qv
        elif kind == E.PAIR:
            qp = np.zeros((S, 1, m))
            qp[rows, 0, at[:, 0]], qp[rows, 0, at[:, 1]] = 0.5, -0.5
            qv, sigma = qp, 2.0 * qp
        else:
            qp = np.zeros((S, 1, m))
            qp[rows, 0, at[:, 0]] = 1.0
            qv, sigma = qp, qp
        eye = np.eye(2)
        n_p, n_v = qp.shape[1], (qv.shape[1] if vel else 0)
        r = 2 * (n_p + n_v)
        C = np.zeros((S, r, 4 * m))
        C[:, :2 * n_p, :2 * m] = np.einsum("snm,ij->snimj", qp, eye
                                           ).reshape(S, 2 * n_p, 2 * m)
        if vel:
            C[:, 2 * n_p:, 2 * m:] = np.einsum("snm,ij->snimj", qv, eye
                                               ).reshape(S, 2 * n_v, 2 * m)
        W = np.zeros((S, r, r))
        for lo in range(0, S, chunk):
            part = np.arange(lo, min(lo + chunk, S))
            for k in range(1, lags + 1):
                j = src.tick[part] - k
                live = j >= 0
                Rk = self.R[self.offset[src.mission[part]]
                            + np.maximum(j, 0), :, :, k - 1].astype(float)
                G = np.einsum("snm,smad->snad", sigma[part], Rk)
                CG = np.einsum("srd,snad->snar", C[part], G)
                W[part] += np.where(live[:, None, None],
                                    np.einsum("snar,snat->srt", CG, CG), 0.0)
        return W

    def updates(self, src: E.Sources, at: np.ndarray) -> np.ndarray:
        if self.U is None:
            raise ValueError("a %s flow keeps no updates" % self.effort)
        return self.U[(self.offset[src.mission] + src.tick)[:, None], at]


def _question(config, kind: str, m: int):
    vel = keeps_velocity(config)
    if kind == E.PAIR:
        qp, signs = np.r_[0.5, -0.5, np.zeros(m - 2)][None], (1.0, -1.0)
        qv = qp
    elif kind == E.BODY:
        qp, signs = np.r_[1.0, np.zeros(m - 1)][None], (1.0,)
        qv = qp
    else:
        qp = np.zeros((0, m)) if vel else np.eye(m) - 1.0 / m
        qv, signs = np.full((1, m), 1.0 / m), (1.0,)
    r = 2 * qp.shape[0] + (2 * qv.shape[0] if vel else 0)
    return qp, qv, signs, r


def _bank_question(config, x: State, kind: str, qp, qv, r: int):
    dt = float(config.physics.dt)
    b = x.size
    origin = (x.positions.mean(axis=1) if kind == E.TARGET
              else E.bank_pin(x, config, kind))
    rotation = E._turn(kind, origin, x.target, x.route, x.positions,
                       E.rotates(config))
    if kind == E.TARGET:
        qbp = np.broadcast_to(qp[None], (b,) + qp.shape)
        qbv = np.broadcast_to(qv[None], (b,) + qv.shape)
    else:
        qbp, _ = E.question_matrix(x, config, kind, "question")
        qbv = qbp
    zb = np.concatenate([
        np.einsum("bnm,bmj->bnj", qbp,
                  x.positions - origin[:, None]).reshape(b, -1),
        dt * np.einsum("bnm,bmj->bnj", qbv,
                       x.velocities).reshape(b, -1)], axis=1)[:, :r]
    return origin, rotation, zb


def question_scale(config, state: State, kind: str) -> np.ndarray:
    m = state.positions.shape[1]
    qp, qv, _, r = _question(config, kind, m)
    zb = np.concatenate([
        _bank_question(config, state.take(np.arange(
            lo, min(lo + E.CHUNK, state.size))), kind, qp, qv, r)[2]
        for lo in range(0, state.size, E.CHUNK)])
    n_p = 2 * qp.shape[0]
    out = np.ones(r)
    for block in (slice(0, n_p), slice(n_p, r)):
        if block.start < block.stop:
            spread = math.sqrt(float(np.mean(np.var(zb[:, block], axis=0))))
            out[block] = spread if spread > 0 else 1.0
    return out


def flow_efforts(config, state: State, src: E.Sources,
                 W: Optional[np.ndarray] = None, *, candidates: int,
                 ridge: float = 1e-9, update: Optional[np.ndarray] = None,
                 scale: Optional[np.ndarray] = None) -> np.ndarray:
    dt = float(config.physics.dt)
    turn_ok = E.rotates(config)
    kind = src.kind
    m = state.positions.shape[1]
    seated = config.physics.targets.kind == "leader_formation"
    qp, qv, signs, r = _question(config, kind, m)
    if W is not None:
        if W.shape[1] != r:
            raise ValueError("the reachability holds %d coordinates, the "
                             "question %d" % (W.shape[1], r))
        level = np.trace(W, axis1=1, axis2=2).mean() / r
    elif scale is None:
        scale = question_scale(config, state, kind)
    centre = kind == E.TARGET
    if centre:
        src = replace(src, origin=src.positions.mean(axis=1))
    own = E._turn(kind, src.origin, src.target, src.route, src.positions,
                  turn_ok)
    ts = _time_scale(config, dt)
    tree = cKDTree(question_features(src.positions, src.velocities,
                                     src.origin, own, kind, ts))
    k = max(1, min(int(candidates), src.size))
    n_p = qp.shape[0]
    blocks = np.arange(2 * n_p).reshape(n_p, 2)
    out = np.empty(state.size)
    for lo in range(0, state.size, E.CHUNK):
        part = np.arange(lo, min(lo + E.CHUNK, state.size))
        x = state.take(part)
        b = part.size
        origin, rotation, zb = _bank_question(config, x, kind, qp, qv, r)
        back = np.transpose(rotation, (0, 2, 1))
        _, near = tree.query(question_features(
            x.positions, x.velocities, origin, rotation, kind, ts), k)
        near = np.asarray(near).reshape(b, k)
        x_s = E._joint(x, dt)
        best = np.full(b, np.inf)
        for c in range(k):
            s = near[:, c]
            x0 = E._moved(src, s, own[s], origin, back, dt)
            turn = np.einsum("bij,bjk->bik", back, own[s])
            take = None
            if kind == E.TARGET and not seated:
                take = E._match(x_s, x0, m)
                x0 = E._relabel(x_s, x0, m)
            p0 = x0[:, :2 * m].reshape(b, m, 2) - origin[:, None]
            v0 = x0[:, 2 * m:].reshape(b, m, 2)
            zs = np.concatenate([
                np.einsum("nm,bmj->bnj", qp, p0).reshape(b, -1),
                np.einsum("nm,bmj->bnj", qv, v0).reshape(b, -1)],
                axis=1)[:, :r]
            if W is None:
                us = None
                if update is not None:
                    du = update[s]
                    dp = np.einsum("bij,bmj->bmi", turn, du[..., :2])
                    dv = np.einsum("bij,bmj->bmi", turn, du[..., 2:])
                    if take is not None:
                        pick = take[:, :, None].repeat(2, 2)
                        dp = np.take_along_axis(dp, pick, 1)
                        dv = np.take_along_axis(dv, pick, 1)
                    us = np.concatenate([
                        np.einsum("nm,bmj->bnj", qp, dp).reshape(b, -1),
                        np.einsum("nm,bmj->bnj", qv, dv).reshape(b, -1)],
                        axis=1)[:, :r] / scale
                for sg in signs:
                    d = (zb - sg * zs) / scale
                    e = np.einsum("bi,bi->b", d, d)
                    if us is not None:
                        size = np.sqrt(e * np.einsum("bi,bi->b", us, us))
                        cos = np.where(size > 0, sg * np.einsum(
                            "bi,bi->b", d, us) / np.where(size > 0, size, 1.0),
                            0.0)
                        e = e * (1.0 - 0.5 * cos)
                    best = np.minimum(best, e)
                continue
            Ws = W[s]
            if take is not None and n_p:
                idx = np.concatenate([
                    blocks[take].reshape(b, -1),
                    np.broadcast_to(np.arange(2 * n_p, r)[None],
                                    (b, r - 2 * n_p))], axis=1)
                Ws = np.take_along_axis(np.take_along_axis(
                    Ws, idx[:, :, None], 1), idx[:, None, :], 2)
            T = np.zeros((b, r, r))
            for blk in range(r // 2):
                T[:, 2 * blk:2 * blk + 2, 2 * blk:2 * blk + 2] = turn
            inv = np.linalg.inv(np.einsum("bij,bjk,blk->bil", T, Ws, T)
                                + ridge * level * np.eye(r)[None])
            for sg in signs:
                delta = zb - sg * zs
                best = np.minimum(best, np.einsum("bi,bij,bj->b", delta, inv,
                                                  delta))
        out[part] = best
    return out


__all__ = ["DIRECTION", "DISTANCE", "EFFORTS", "Flow", "RESPONSE",
           "flight_states", "flow_efforts", "keeps_velocity",
           "question_features", "question_scale", "responses", "updates"]
