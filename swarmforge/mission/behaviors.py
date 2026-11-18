from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..solve.config import FamilyConfig
from ..solve.dynamics import formation
from .snapshot import Snapshot
from .verdict import Oracle

PERSIST = 0.05
TAIL = 0.10
SLOW = 0.01
SATURATED = 0.98
NEAR = 1.0
CHATTER_TURNS = 3
SWING = 1.0
DISORDER = 0.5
MILLING = 0.5
COLLAPSE = 0.5
OVERSHOOT_NEAR = 2.0
OVERSHOOT_BACK = 1.0
ORBIT_RING = 3.0
FORMATION_BREAK = 1.0
STRETCH = 3.0


EVENTS = ("peer_collision", "body_collision", "mover_collision", "fragmentation",
          "isolation", "stray", "implosion",
          "disorder", "milling", "chatter", "hesitation", "backtrack", "deadlock", "livelock",
          "goal_overshoot", "goal_orbit", "outside_arena", "formation_break")


@dataclass(frozen=True)
class Settings:
    contact: float
    agent_radius: float
    obstacle_gap: float
    link: float
    step: float
    settle: int
    persist: int
    tail: int
    goal_radius: float
    arrive_fraction: float
    formation_robots: int = 0
    formation_spacing: float = 0.0

    @classmethod
    def for_config(cls, config: FamilyConfig) -> "Settings":
        if config.mission is None:
            raise ValueError("%s declares no mission block" % config.path)
        oracle = Oracle.for_config(config)
        ticks = int(config.mission.ticks)
        targets = config.physics.targets
        leader = targets.kind == "leader_formation"
        return cls(contact=oracle.agent_boundary,
                   agent_radius=oracle.agent_radius,
                   obstacle_gap=oracle.obstacle_gap,
                   link=float(config.state.window.drones),
                   step=float(config.physics.plant.max_step),
                   settle=oracle.settle_ticks,
                   persist=max(1, int(math.ceil(PERSIST * ticks))),
                   tail=max(1, int(math.ceil(TAIL * ticks))),
                   goal_radius=oracle.goal_radius,
                   arrive_fraction=oracle.arrive_fraction,
                   formation_robots=int(targets.formation_robots) if leader else 0,
                   formation_spacing=float(targets.formation_spacing) if leader else 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _first(mask: np.ndarray, ticks: np.ndarray) -> Optional[int]:
    found = np.flatnonzero(mask)
    return int(ticks[found[0]]) if found.size else None


def _sustained(mask: np.ndarray, ticks: np.ndarray, length: int
               ) -> Optional[int]:
    start = None
    for k, on in enumerate(mask):
        if not on:
            start = None
            continue
        if start is None:
            start = k
        if ticks[k] - ticks[start] >= length:
            return int(ticks[start])
    return None


def _fresh(near: np.ndarray) -> np.ndarray:
    apart = np.logical_or.accumulate(~near, axis=0)
    armed = np.concatenate([np.zeros_like(apart[:1]), apart[:-1]], axis=0)
    return near & armed


def reversals(P: np.ndarray, swing: float) -> np.ndarray:
    T, N = P.shape[:2]
    out = np.zeros((T, N), bool)
    for i in range(N):
        pivot, far = 0, 0
        for k in range(1, T):
            x = P[k, i]
            out_leg = P[far, i] - P[pivot, i]
            reach = float(np.hypot(*out_leg))
            if float(np.hypot(*(x - P[pivot, i]))) >= reach:
                far = k
                continue
            if reach < 1e-12:
                continue
            u = out_leg / reach
            back = P[far, i] - x
            along = float(back @ u)
            side = abs(float(back[0] * u[1] - back[1] * u[0]))
            if along >= swing and side <= along:
                out[far, i] = reach >= swing
                pivot, far = far, k
    return out


def _bursts(turns: np.ndarray, tk: np.ndarray, count: int, window: int) -> Optional[int]:
    first = None
    for i in range(turns.shape[1]):
        at = tk[np.flatnonzero(turns[:, i])]
        for j in range(len(at) - count + 1):
            if at[j + count - 1] - at[j] <= window:
                first = int(at[j]) if first is None else min(first, int(at[j]))
                break
    return first


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _components(adjacent: np.ndarray) -> np.ndarray:
    reach = adjacent.astype(np.float32)
    for _ in range(max(1, int(math.ceil(math.log2(max(2, adjacent.shape[1])))))):
        reach = (np.matmul(reach, reach) > 0).astype(np.float32)
    return np.rint(np.sum(1.0 / reach.sum(axis=2), axis=1)).astype(int)


def _body_gaps(snapshots: Sequence[Snapshot], P: np.ndarray,
               settings: Settings) -> Optional[np.ndarray]:
    if not any(s.obstacles for s in snapshots):
        return None
    out = np.full(P.shape[:2], np.inf)
    sizes = {len(s.obstacles) for s in snapshots}
    if len(sizes) == 1:
        bodies = np.array([[(c[0], c[1], r) for c, r in s.obstacles]
                           for s in snapshots], float)
        for a in range(0, len(snapshots), 256):
            b = min(len(snapshots), a + 256)
            delta = P[a:b, :, None, :] - bodies[a:b, None, :, :2]
            gap = (np.linalg.norm(delta, axis=-1) - bodies[a:b, None, :, 2]
                   - settings.agent_radius)
            out[a:b] = gap.min(axis=2)
        return out
    for k, s in enumerate(snapshots):
        if s.obstacles:
            bodies = np.array([(c[0], c[1], r) for c, r in s.obstacles], float)
            delta = P[k, :, None, :] - bodies[None, :, :2]
            out[k] = (np.linalg.norm(delta, axis=-1) - bodies[None, :, 2]
                      - settings.agent_radius).min(axis=1)
    return out


def _slots(aims: np.ndarray, leader: np.ndarray, settings: Settings,
           followers: int) -> np.ndarray:
    slots = formation(settings.formation_robots, aims, aims - leader,
                      settings.formation_spacing)
    out = np.full((aims.shape[0], followers, 2), np.nan)
    for i in range(min(followers, len(slots))):
        out[:, i] = slots[i] - aims
    return out


def measure(snapshots: Sequence[Snapshot], document: Dict[str, Any],
            settings: Settings) -> Dict[str, Any]:
    events: Dict[str, Optional[int]] = {name: None for name in EVENTS}
    if len(snapshots) < 2:
        return {"observed": None, "events": events, "metrics": {}}
    P = np.array([s.positions for s in snapshots], float)
    ticks = np.array([s.tick for s in snapshots], int)
    T, N = P.shape[:2]
    goal = np.array(document["goal"], float)
    need = max(1, int(math.ceil(settings.arrive_fraction * N)))
    to_goal = np.linalg.norm(P - goal, axis=-1)
    order = np.sort(to_goal, axis=1)
    front = order[:, need - 1]
    arrived = front <= settings.goal_radius
    started = np.flatnonzero(ticks >= settings.settle)
    if started.size == 0:
        return {"observed": None, "events": events, "metrics": {}}
    i0 = int(started[0])
    after = np.flatnonzero(arrived[i0:])
    i1 = i0 + int(after[0]) if after.size else T - 1
    seen = slice(i0, i1 + 1)
    tk = ticks[seen]
    gaps = np.maximum(1, np.diff(ticks))[:, None, None]
    S = np.diff(P, axis=0) / gaps
    S = np.concatenate([S[:1], S], axis=0)
    speed = np.linalg.norm(S, axis=-1) / settings.step
    moving = speed > SLOW
    metrics: Dict[str, Any] = {}
    c = settings.contact

    centre = P.mean(axis=1)
    spread = np.linalg.norm(P - centre[:, None], axis=-1).mean(axis=1)
    if N > 1:
        D = np.linalg.norm(P[:, :, None] - P[:, None, :], axis=-1)
        D[:, np.arange(N), np.arange(N)] = np.inf
        least = D.min(axis=(1, 2))
        near_peer = np.zeros(T, bool)
        near_peer[seen] = _fresh(D[seen] - c < NEAR * c).any(axis=(1, 2))
        events["peer_collision"] = _first(least[seen] < c, tk)
        metrics["peer_gap_min"] = _round((least[seen].min() - c) / c)
        R = P[:, None] - P[:, :, None]
        W = S[:, None] - S[:, :, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            closing = -np.sum(R * W, axis=-1) / D
            ttc = np.where((closing > 0) & (D <= settings.link),
                           (D - c) / closing, np.inf)
        best = ttc[seen].min()
        metrics["ttc_min"] = _round(max(0.0, best), 2) if np.isfinite(best) else None
        adjacent = D <= settings.link
        alone = ~adjacent.any(axis=2)
        adjacent[:, np.arange(N), np.arange(N)] = True
        groups = _components(adjacent)
        split = groups > groups[i0]
        events["fragmentation"] = _sustained(split[seen], tk, settings.persist)
        events["isolation"] = min((t for t in (
            _sustained(alone[seen, i], tk, settings.persist) for i in range(N)
            if not alone[i0, i]) if t is not None), default=None)
        metrics["groups_max"] = int(groups[seen].max())
        metrics["split_share"] = _round(split[seen].mean())
        metrics["isolated_share"] = _round(alone[seen].mean())
        collapse = spread < COLLAPSE * max(spread[i0], 1e-12)
        away_from_goal = np.linalg.norm(centre - goal, axis=-1) > settings.goal_radius
        events["implosion"] = _first((collapse & near_peer & away_from_goal)[seen], tk)
    radius0 = max(float(spread[i0]), c)
    away = np.linalg.norm(P - centre[:, None], axis=-1) / radius0
    if N > 2:
        events["stray"] = min((t for t in (_sustained((away[seen, i] > STRETCH), tk, settings.persist)
                                               for i in range(N)) if t is not None), default=None)
    metrics["stretch_max"] = _round(float(away[seen].max()))
    metrics["spread_min_ratio"] = _round(spread[seen].min() / max(spread[i0], 1e-12))
    metrics["spread_max_ratio"] = _round(spread[seen].max() / max(spread[i0], 1e-12))

    body = _body_gaps(snapshots, P, settings)
    if body is not None:
        clear = body - settings.obstacle_gap
        events["body_collision"] = _first(clear[seen].min(axis=1) < 0.0, tk)
        metrics["body_gap_min"] = _round(clear[seen].min() / c)
        hit = events["body_collision"]
        if hit is not None:
            snap = snapshots[int(np.flatnonzero(ticks == hit)[0])]
            if snap.movers:
                bodies = np.array([(q[0], q[1], r) for q, r in snap.obstacles], float)
                gap = (np.linalg.norm(P[ticks == hit][0][:, None] - bodies[None, :, :2], axis=-1)
                       - bodies[None, :, 2] - settings.agent_radius - settings.obstacle_gap)
                touched = bodies[np.flatnonzero((gap < 0.0).any(axis=0)), :2]
                movers = np.array([q for q, _r, _v in snap.movers], float)
                scale = 1e-6 * max(1.0, float(np.abs(movers).max()))
                if any(np.min(np.linalg.norm(movers - b, axis=-1)) <= scale for b in touched):
                    events["mover_collision"] = hit
    else:
        metrics["body_gap_min"] = None

    with np.errstate(divide="ignore", invalid="ignore"):
        U = S / np.linalg.norm(S, axis=-1, keepdims=True)
    U = np.where(moving[..., None], U, np.nan)
    count = moving.sum(axis=1)
    polar = np.where(count >= 2, np.linalg.norm(np.nansum(U, axis=1), axis=-1)
                     / np.maximum(count, 1), np.nan)
    rel = P - centre[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = rel / np.linalg.norm(rel, axis=-1, keepdims=True)
    cross = rhat[..., 0] * U[..., 1] - rhat[..., 1] * U[..., 0]
    rotation = np.where(count >= 3, np.abs(np.nansum(cross, axis=1))
                        / np.maximum(count, 1), np.nan)
    with np.errstate(invalid="ignore"):
        events["disorder"] = _sustained((polar < DISORDER)[seen], tk,
                                        settings.persist)
        events["milling"] = _sustained(((rotation > MILLING)
                                        & (polar < DISORDER))[seen], tk,
                                       settings.persist)
    metrics["polarization_mean"] = _round(np.nanmean(polar[seen])
                                          if np.isfinite(polar[seen]).any() else None)
    metrics["rotation_max"] = _round(np.nanmax(rotation[seen])
                                     if np.isfinite(rotation[seen]).any() else None)

    both = moving[1:] & moving[:-1]
    both[0] = False
    dot = np.sum(S[1:] * S[:-1], axis=-1)
    both = np.concatenate([np.zeros((1, N), bool), both], axis=0)
    turns = reversals(P[seen], SWING * c) & (to_goal[seen] > settings.goal_radius)
    counts = turns.sum(axis=0)
    metrics["reversals_max"] = int(counts.max())
    metrics["reversals_total"] = int(counts.sum())
    events["chatter"] = _bursts(turns, tk, CHATTER_TURNS, settings.persist)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.clip(dot / (np.linalg.norm(S[1:], axis=-1)
                                * np.linalg.norm(S[:-1], axis=-1)), -1.0, 1.0)
    turn = np.degrees(np.arccos(cosine)) / np.maximum(1, np.diff(ticks))[:, None]
    turn = np.concatenate([np.full((1, N), np.nan), np.where(both[1:], turn, np.nan)])
    metrics["turn_rate"] = _round(np.nanmean(turn[seen]) if np.isfinite(turn[seen]).any()
                                  else None, 2)
    change = np.linalg.norm(np.diff(S, axis=0), axis=-1) / settings.step
    change = np.concatenate([np.full((2, N), np.nan), change[1:]], axis=0)
    metrics["step_change"] = _round(np.nanmean(change[seen])
                                    if np.isfinite(change[seen]).any() else None)
    metrics["saturation_share"] = _round((speed[seen] >= SATURATED).mean())
    metrics["still_share"] = _round((~moving[seen]).mean())

    did_arrive = bool(arrived[i1])
    metrics["arrival_tick"] = int(ticks[i1]) if did_arrive else None
    metrics["stragglers"] = (_round(float((to_goal[i1] > settings.goal_radius).mean()))
                             if did_arrive else None)
    metrics["progress"] = _round(1.0 - front[i1] / max(front[i0], 1e-12))
    net = np.linalg.norm(centre[i0] - goal) - np.linalg.norm(centre[i1] - goal)
    path = np.linalg.norm(np.diff(centre[seen], axis=0), axis=-1).sum()
    metrics["path_ratio"] = _round(path / net) if net > 1e-9 * settings.step else None
    if not did_arrive and ticks[i1] - ticks[i0] >= settings.tail:
        k = int(np.searchsorted(ticks, ticks[i1] - settings.tail))
        k = max(k, i0)
        span = max(1, ticks[i1] - ticks[k])
        rate = (front[k] - front[i1]) / span / settings.step
        still = float(speed[k + 1:i1 + 1].mean()) if i1 > k else 0.0
        metrics["tail_progress"] = _round(rate)
        metrics["tail_speed"] = _round(still)
        if rate < SLOW:
            events["deadlock" if still < SLOW else "livelock"] = int(ticks[k])
    else:
        metrics["tail_progress"] = metrics["tail_speed"] = None
    diameter = max(2.0 * float(spread[i0]), c)
    end = ticks[i1] - settings.tail
    for k in range(i0, i1 + 1):
        j = int(np.searchsorted(ticks, ticks[k] + settings.persist))
        if j > i1 or ticks[j] > end:
            break
        stalled = (front[k] - front[j]) / max(1, ticks[j] - ticks[k]) / settings.step < SLOW
        if stalled and front[j:i1 + 1].min() <= front[j] - diameter:
            events["hesitation"] = int(ticks[k])
            break
    best = np.minimum.accumulate(front[seen])
    rise = front[seen] - best
    zone = OVERSHOOT_NEAR * settings.goal_radius
    events["backtrack"] = _first((best > zone) & (rise >= diameter), tk)
    metrics["backtrack_max"] = _round(float(np.where(best > zone, rise, 0.0).max()) / diameter)
    floor = np.minimum.accumulate(front[seen])
    back = ((floor <= OVERSHOOT_NEAR * settings.goal_radius)
            & (front[seen] >= floor + OVERSHOOT_BACK * settings.goal_radius))
    events["goal_overshoot"] = _first(back, tk)
    nearest = np.argsort(to_goal, axis=1)[:, :need]
    lead = np.take_along_axis(P, nearest[..., None], axis=1).mean(axis=1) - goal
    angle = np.arctan2(lead[:, 1], lead[:, 0])
    ring = front <= ORBIT_RING * settings.goal_radius
    turned = 0.0
    for k in range(i0 + 1, i1 + 1):
        if ring[k] and ring[k - 1]:
            turned += math.remainder(angle[k] - angle[k - 1], 2.0 * math.pi)
            if abs(turned) >= 2.0 * math.pi:
                events["goal_orbit"] = int(ticks[k])
                break
        else:
            turned = 0.0

    bounds = document.get("bounds")
    if bounds is not None:
        lo = -P
        hi = P - np.array(bounds, float)
        depth = np.maximum(lo, hi).max(axis=-1)
        out = depth > settings.agent_radius
        events["outside_arena"] = _first(out[seen].any(axis=1), tk)
        metrics["outside_share"] = _round(out[seen].mean())
        metrics["outside_depth_max"] = _round(max(0.0, depth[seen].max()) / c)
    if settings.formation_robots > 1 and N > 1:
        aims = [s.aim for s in snapshots]
        if all(a is not None for a in aims):
            A = np.array(aims, float)
            offsets = _slots(A, P[:, 0], settings, N - 1)
            shape = np.linalg.norm((P[:, 1:] - P[:, :1]) - offsets, axis=-1)
            error = np.nanmean(shape, axis=1) / settings.formation_spacing
            events["formation_break"] = _sustained((error > FORMATION_BREAK)[seen],
                                                   tk, settings.persist)
            metrics["formation_error_mean"] = _round(np.nanmean(error[seen]))
            metrics["formation_error_max"] = _round(np.nanmax(error[seen]))
    return {"observed": [int(ticks[i0]), int(ticks[i1])], "events": events,
            "metrics": metrics}


__all__ = ["EVENTS", "Settings", "measure", "reversals"]
