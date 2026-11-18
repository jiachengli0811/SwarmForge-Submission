from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..harden.features import SPACES, window_points
from ..harden.space import (SpaceSettings, describe_space, for_model,
                            load_space, space_bank, space_path)
from ..solve.bank import fingerprint
from ..solve.config import FamilyConfig
from ..solve.models import build_model
from ..solve.record import Record
from ..solve.state import State
from ..solve.verdict import CHARGED, Verdict
from ..solve.window import Reaches
from .snapshot import Snapshot

CRASH_KINDS = {"agent_collision": ("agent", "both"),
               "obstacle_collision": ("obstacle", "both"),
               "timeout": ()}


def failure_of(model: str, verdict: str, crash_kind: str = "") -> bool:
    if model == "timeout":
        return verdict == "timeout"
    return verdict == "crash" and crash_kind in CRASH_KINDS[model]


STALL_SHARE = 0.5
TAIL = 0.25
WEAK_SAMPLE = 1500

_CHARGED = np.array([int(v) for v in CHARGED])


def _nearest_on_polygon(points: np.ndarray, polygon: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    a = polygon
    b = np.roll(polygon, -1, axis=0)
    ab = b - a
    ap = points[:, None, :] - a[None]
    t = np.clip(np.sum(ap * ab, axis=-1)
                / np.maximum(np.sum(ab * ab, axis=-1), 1e-300), 0.0, 1.0)
    q = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(points[:, None, :] - q, axis=-1)
    nearest = q[np.arange(points.shape[0]), np.argmin(d, axis=1)]
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = (((a[:, 1] > y) != (b[:, 1] > y))
                 & (x < (b[:, 0] - a[:, 0]) * (y - a[:, 1])
                    / (b[:, 1] - a[:, 1]) + a[:, 0]))
    return nearest, np.sum(cross, axis=1) % 2 == 1


def surface_distance(points: np.ndarray, centres: np.ndarray,
                     radii: np.ndarray, square: bool) -> np.ndarray:
    offset = points - centres
    if square:
        outside = np.maximum(np.abs(offset) - radii[..., None], 0.0)
        return np.sqrt(np.sum(outside * outside, axis=-1))
    return np.linalg.norm(offset, axis=-1) - radii


class _Field:
    def __init__(self, document: Dict[str, Any], square: bool = False):
        self.square = square
        static = document.get("static", [])
        discs = [b for b in static if not b.get("polygon")]
        self.index = ([i for i, b in enumerate(static)
                       if not b.get("polygon")]
                      + [i for i, b in enumerate(static) if b.get("polygon")])
        self.centres = np.array([b["position"] for b in discs],
                                dtype=float).reshape(-1, 2)
        self.radii = np.array([float(b["radius"]) for b in discs],
                              dtype=float)
        self.walls = [np.array(b["polygon"], dtype=float)
                      for b in static if b.get("polygon")]

    @property
    def count(self) -> int:
        return self.radii.size + len(self.walls)

    def seen_from(self, points: np.ndarray):
        n = points.shape[0]
        centres = np.broadcast_to(self.centres[None],
                                  (n,) + self.centres.shape)
        radii = np.broadcast_to(self.radii[None], (n, self.radii.size))
        surface = surface_distance(points[:, None], self.centres[None],
                                   self.radii[None], self.square)
        if not self.walls:
            return centres, radii, surface, np.zeros(self.radii.size, bool)
        c, r, s = [centres], [radii], [surface]
        for polygon in self.walls:
            q, inside = _nearest_on_polygon(points, polygon)
            gap = np.where(inside, 0.0, np.linalg.norm(q - points, axis=-1))
            c.append(np.where(inside[:, None], points, q)[:, None])
            r.append(np.zeros((n, 1)))
            s.append(gap[:, None])
        wall = np.array([False] * self.radii.size + [True] * len(self.walls))
        return (np.concatenate(c, axis=1), np.concatenate(r, axis=1),
                np.concatenate(s, axis=1), wall)


@dataclass
class _Group:
    positions: List[np.ndarray] = field(default_factory=list)
    velocities: List[np.ndarray] = field(default_factory=list)
    obstacles: List[np.ndarray] = field(default_factory=list)
    obstacle_velocities: List[np.ndarray] = field(default_factory=list)
    obstacle_radii: List[np.ndarray] = field(default_factory=list)
    target: List[np.ndarray] = field(default_factory=list)
    route: List[np.ndarray] = field(default_factory=list)
    tick: List[int] = field(default_factory=list)
    truncated: List[int] = field(default_factory=list)
    wall: List[bool] = field(default_factory=list)
    members: List[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] \
        = field(default_factory=list)
    anchors: List[Tuple[int, ...]] = field(default_factory=list)
    body: List[str] = field(default_factory=list)

    def state(self, static: int, dynamic: int) -> State:
        def stack(rows, tail):
            return np.array(rows, dtype=float).reshape((len(rows),) + tail)
        m = self.positions[0].shape[0]
        k = static + dynamic
        return State(positions=stack(self.positions, (m, 2)),
                     velocities=stack(self.velocities, (m, 2)),
                     obstacles=stack(self.obstacles, (k, 2)),
                     obstacle_velocities=stack(self.obstacle_velocities,
                                               (k, 2)),
                     obstacle_radii=stack(self.obstacle_radii, (k,)),
                     dynamic=np.array([False] * static + [True] * dynamic),
                     target=stack(self.target, (2,)),
                     route=stack(self.route, (2,)))


def crash_body(snapshot: Snapshot, field_: "_Field", config: FamilyConfig
               ) -> str:
    physics = config.physics
    margin = physics.agent_radius + physics.obstacle_min_gap
    squares = physics.geometry.obstacles == "square"
    p = np.asarray(snapshot.positions, dtype=float).reshape(-1, 2)
    best, label = np.inf, ""
    for polygon in field_.walls:
        q, inside = _nearest_on_polygon(p, polygon)
        gap = np.where(inside, 0.0, np.linalg.norm(q - p, axis=-1)).min()
        if gap - margin < best:
            best, label = gap - margin, "wall"
    bodies = [("static", c, r) for c, r in zip(field_.centres, field_.radii)]
    bodies += [("dynamic", np.asarray(c, dtype=float), r)
               for c, r, _ in snapshot.movers]
    for kind, centre, radius in bodies:
        disc = float(np.min(np.linalg.norm(p - centre, axis=-1))) - radius
        gap, name = disc, kind
        if squares:
            half = np.array([radius, radius])
            square = np.array([centre - half, centre + [radius, -radius],
                               centre + half, centre + [-radius, radius]])
            q, inside = _nearest_on_polygon(p, square)
            square_gap = np.where(inside, 0.0,
                                  np.linalg.norm(q - p, axis=-1)).min()
            if square_gap - margin < 0.0 <= disc - margin:
                gap, name = square_gap, kind + "+corner"
        if gap - margin < best:
            best, label = gap - margin, name
    return label if best < 0.0 else ""


class _Windows:
    def __init__(self, model_name: str, config: FamilyConfig, spec,
                 horizon: int, document: Dict[str, Any]):
        physics = config.physics
        self.agent = model_name == "agent_collision"
        self.swarm = model_name == "timeout"
        self.m, self.ks, self.kd = (spec.drones, spec.static_obstacles,
                                    spec.dynamic_obstacles)
        self.reaches = Reaches.of(config, horizon)
        self.formation = physics.targets.kind == "leader_formation"
        self.velocity = physics.plant.integration == "velocity"
        self.boundary = physics.agent_boundary
        self.agent_ring = 2.0 * horizon * physics.plant.max_step
        self.body_ring = horizon * physics.plant.max_step
        self.horizon_time = horizon * physics.dt
        self.margin = physics.agent_radius + physics.obstacle_min_gap
        self.square = physics.square_bodies
        self.field = _Field(document, self.square)
        self.goal = tuple(document["goal"])
        self.groups: Dict[Tuple[int, int, int], _Group] = {}


    def add(self, snapshot: Snapshot) -> None:
        p = np.asarray(snapshot.positions, dtype=float).reshape(-1, 2)
        n = p.shape[0]
        v = (np.asarray(snapshot.velocities, dtype=float).reshape(-1, 2)
             if self.velocity else np.zeros((n, 2)))
        if self.formation and n > self.m:
            raise ValueError("a formation window keeps every drone, and the "
                             "model holds %d of %d" % (self.m, n))
        movers = snapshot.movers
        mc = np.array([c for c, _, _ in movers], dtype=float).reshape(-1, 2)
        mr = np.array([r for _, r, _ in movers], dtype=float)
        mv = np.array([u for _, _, u in movers], dtype=float).reshape(-1, 2)
        sc, sr, ss, wall = self.field.seen_from(p)
        static_centre = np.linalg.norm(p[:, None] - sc, axis=-1)
        dynamic_centre = np.linalg.norm(p[:, None] - mc[None], axis=-1)
        dynamic_surface = surface_distance(p[:, None], mc[None], mr[None],
                                           self.square)
        apart = np.linalg.norm(p[:, None] - p[None], axis=-1)
        target = np.asarray(snapshot.aim if snapshot.aim is not None
                            else self.goal, dtype=float)
        route = np.asarray(snapshot.route, dtype=float)
        frame = (p, v, sc, sr, ss, wall, static_centre, mc, mr, mv,
                 dynamic_centre, dynamic_surface, apart, target, route)

        if self.swarm:
            for i in range(n):
                self._window(snapshot.tick, (i,), None, frame)
            return
        if self.agent:
            first, second = np.triu_indices(n, 1)
            d = apart[first, second]
            ring = (d >= self.boundary) & (d < self.boundary + self.agent_ring)
            for i, j in zip(first[ring], second[ring]):
                self._window(snapshot.tick, (int(i), int(j)), None, frame)
            return
        clearance = ss - self.margin
        rows, bodies = np.nonzero((clearance >= 0.0)
                                  & (clearance < self.body_ring))
        for i, k in zip(rows, bodies):
            self._window(snapshot.tick, (int(i),), ("static", int(k)), frame)
        if mr.size:
            width = self.body_ring + self.horizon_time * np.linalg.norm(mv,
                                                                        axis=1)
            clearance = dynamic_surface - self.margin
            rows, bodies = np.nonzero((clearance >= 0.0)
                                      & (clearance < width[None]))
            for i, k in zip(rows, bodies):
                self._window(snapshot.tick, (int(i),), ("dynamic", int(k)),
                             frame)

    def _window(self, tick: int, drones: Tuple[int, ...],
                body: Optional[Tuple[str, int]], frame) -> None:
        (p, v, sc, sr, ss, wall, static_centre, mc, mr, mv, dynamic_centre,
         dynamic_surface, apart, target, route) = frame
        n = p.shape[0]
        anchors = list(drones)
        cut = 0
        if self.formation:
            keep = list(range(n))
        else:
            others = np.array([e for e in range(n) if e not in anchors],
                              dtype=int)
            near = apart[anchors][:, others].min(axis=0) if others.size \
                else np.zeros(0)
            inside = others[near <= self.reaches.drones]
            order = inside[np.argsort(near[near <= self.reaches.drones],
                                      kind="stable")]
            room = self.m - len(anchors)
            cut += max(0, order.size - room)
            keep = sorted(anchors + [int(e) for e in order[:room]])
        reader = np.array(keep if self.swarm else anchors)
        best = reader[np.argmin(ss[reader], axis=0)] if self.field.count \
            else np.zeros(0, dtype=int)
        columns = np.arange(self.field.count)
        reach = static_centre[best, columns] <= self.reaches.static_bodies
        order = columns[reach][np.argsort(ss[best, columns][reach],
                                          kind="stable")]
        static = [int(k) for k in order]
        dynamic: List[int] = []
        if mr.size:
            near = dynamic_centre[reader].min(axis=0)
            inside = np.flatnonzero(near <= self.reaches.dynamic_bodies)
            dynamic = [int(k) for k in inside[np.argsort(
                dynamic_surface[reader][:, inside].min(axis=0),
                kind="stable")]]
        if body is not None:
            kind, k = body
            chosen = static if kind == "static" else dynamic
            if k in chosen:
                chosen.remove(k)
            chosen.insert(0, k)
        cut += max(0, len(static) - self.ks) + max(0, len(dynamic) - self.kd)
        static, dynamic = static[:self.ks], dynamic[:self.kd]

        group = self.groups.setdefault((len(keep), len(static), len(dynamic)),
                                       _Group())
        group.positions.append(p[keep])
        group.velocities.append(v[keep])
        group.obstacles.append(np.concatenate(
            [sc[best[static], static].reshape(-1, 2), mc[dynamic]]))
        group.obstacle_velocities.append(np.concatenate(
            [np.zeros((len(static), 2)), mv[dynamic]]))
        group.obstacle_radii.append(np.concatenate(
            [sr[best[static], static], mr[dynamic]]))
        group.target.append(target)
        group.route.append(route)
        group.tick.append(int(tick))
        group.truncated.append(cut)
        group.wall.append(bool(np.any(wall[static])) if static else False)
        group.members.append((tuple(int(e) for e in keep), tuple(static),
                              tuple(dynamic)))
        group.anchors.append(tuple(int(e) for e in anchors))
        if body is None:
            body = (("static", static[0]) if static else
                    ("dynamic", dynamic[0]) if dynamic else None)
        if body is not None and body[0] == "static":
            body = ("static", self.field.index[body[1]])
        group.body.append("" if body is None else "%s:%d" % body)


@dataclass
class Reading:
    model: str
    ticks: int = 0
    windows: int = 0
    placed: int = 0
    in_space: bool = False
    first_in: Optional[int] = None
    in_ticks: int = 0
    doomed_ticks: int = 0
    infeasible: int = 0
    precedes_crash: Optional[bool] = None
    truncated: int = 0
    walls: int = 0
    crash_body: str = ""
    stall_ticks: int = 0
    stall_share: Optional[float] = None
    longest_stall: int = 0
    final_stall: Optional[float] = None
    cells: Dict[str, int] = field(default_factory=dict)
    charged_cells: Dict[str, int] = field(default_factory=dict)
    charged_first: Dict[str, int] = field(default_factory=dict)
    failure_cell: Optional[int] = None
    failure_charged: Optional[bool] = None
    failure_where: Optional[List[float]] = None
    failure_tick: Optional[int] = None
    failure_body: str = ""
    failure_point: Optional[List[float]] = None
    approach: Optional["Approach"] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {f.name: getattr(self, f.name)
               for f in dataclasses.fields(self) if f.name != "approach"}
        out["cells"] = dict(self.cells)
        out["charged_cells"] = dict(self.charged_cells)
        out["charged_first"] = dict(self.charged_first)
        return out


@dataclass
class Approach:
    distance: np.ndarray
    where: np.ndarray
    tick: np.ndarray
    drone: np.ndarray
    body: List[str]
    point: np.ndarray
    heading: np.ndarray
    weak_distance: Optional[np.ndarray] = None
    weak_where: Optional[np.ndarray] = None
    weak_body: Optional[List[str]] = None
    weak_heading: Optional[np.ndarray] = None
    weak_point: Optional[np.ndarray] = None
    weak_target: Optional[np.ndarray] = None


def cached_verdicts(model, bank_fingerprint: str, bank_path: Path,
                    states) -> np.ndarray:
    config = model.config
    physics = config.physics
    mission = config.mission
    fields = {
        "bank": bank_fingerprint, "model": model.name,
        "spec": dataclasses.asdict(model.spec),
        "physics": {"dt": physics.dt, "agent_radius": physics.agent_radius,
                    "agent_min_gap": physics.agent_min_gap,
                    "obstacle_min_gap": physics.obstacle_min_gap,
                    "plant": dataclasses.asdict(physics.plant),
                    "geometry": dataclasses.asdict(physics.geometry),
                    "targets": dataclasses.asdict(physics.targets),
                    "record_params": dict(physics.record_params)},
        "mission": (None if mission is None else
                    [mission.goal_radius, mission.arrive_fraction]),
        "record": hashlib.sha1(model.record.path.read_bytes()).hexdigest(),
    }
    if getattr(model, "judge", None):
        fields["judge"] = model.judge
    key = json.dumps(fields, sort_keys=True, default=str)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    path = Path(bank_path).parent / "verdicts" / ("%s-%s.npy"
                                                  % (model.name, digest))
    if path.is_file():
        return np.load(path)
    verdict = model.evaluate(states()).verdict
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npy")
    os.close(handle)
    try:
        np.save(tmp, verdict)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return verdict


def bank_verdicts(model, bank) -> np.ndarray:
    return cached_verdicts(model, bank.meta.get("fingerprint"), bank.path,
                           lambda: bank.state)


def pad(state: State, shape: Tuple[int, int, int],
        config: FamilyConfig) -> State:
    m, ks, kd = shape
    have_m, k = state.drones, state.obstacles.shape[1]
    mask = np.asarray(state.dynamic, dtype=bool)
    have_ks = int(np.sum(~mask))
    have_kd = k - have_ks
    if (have_m, have_ks, have_kd) == (m, ks, kd):
        return state
    w = config.state.window
    far = 20.0 * max(w.drones or 0.0, w.static_bodies or 0.0,
                     w.dynamic_bodies or 0.0, w.walls or 0.0, 1.0)
    b = state.size
    anchor = state.positions[:, :1]

    def spots(count, sign):
        steps = (np.arange(count) + 1.0)[None, :, None] * far
        return anchor + sign * np.concatenate([steps, steps * 0.5], axis=-1)

    def body(count, sign, radius):
        return (spots(count, sign), np.zeros((b, count, 2)),
                np.full((b, count), radius), np.zeros((b, count, 2)))

    static, dynamic = ~mask, mask
    radius_s = (config.state.static.radius.lo
                if config.state.static is not None else 0.0)
    radius_d = (config.state.dynamic.radius.lo
                if config.state.dynamic is not None else 0.0)
    ps, vs, rs, axs = body(ks - have_ks, -1.0, radius_s)
    pd, vd, rd, axd = body(kd - have_kd, np.array([1.0, -1.0]), radius_d)
    axes = state.obstacle_axes
    return State(
        positions=np.concatenate([state.positions,
                                  spots(m - have_m, 1.0)], axis=1),
        velocities=np.concatenate([state.velocities,
                                   np.zeros((b, m - have_m, 2))], axis=1),
        obstacles=np.concatenate([state.obstacles[:, static], ps,
                                  state.obstacles[:, dynamic], pd], axis=1),
        obstacle_velocities=np.concatenate(
            [state.obstacle_velocities[:, static], vs,
             state.obstacle_velocities[:, dynamic], vd], axis=1),
        obstacle_radii=np.concatenate(
            [state.obstacle_radii[:, static], rs,
             state.obstacle_radii[:, dynamic], rd], axis=1),
        dynamic=np.array([False] * ks + [True] * kd),
        target=state.target, route=state.route,
        obstacle_axes=np.concatenate([axes[:, static], axs, axes[:, dynamic],
                                      axd], axis=1))


def space_settings(config: FamilyConfig) -> SpaceSettings:
    from ..harden.config import read_harden

    scoring = read_harden(config.path).scoring
    if not scoring.region:
        raise ValueError("%s scores no weakness space (harden.scoring is %s): "
                         "missions are read in its space sampling's cells"
                         % (config.path, scoring.name))
    return scoring.space_settings()


def spaces_of(config: FamilyConfig, record: Record,
              models: Sequence[str]) -> Dict[str, "Space"]:
    settings = space_settings(config)
    return {m: Space(m, config, record, settings) for m in models}


_LONE = {"centroid_dist": "drone_cap", "t_mates_v_along": 0.0}


class Space:
    def __init__(self, model_name: str, config: FamilyConfig, record: Record,
                 settings: SpaceSettings):
        model = build_model(model_name, config, record)
        self.name = model_name
        self.config, self.record = config, record
        self.spec, self.horizon = model.spec, model.horizon
        free = build_model(model_name, config)
        self.settings = for_model(settings, free)
        sampling = config.sampling
        description = describe_space(free, self.settings, sampling.samples,
                                     sampling.seed)
        path = space_path(free, description)
        if not path.is_file():
            space_bank(free, self.settings)
        self.cells = load_space(path)
        verdict = cached_verdicts(model, fingerprint(description), path,
                                  lambda: space_bank(free, self.settings)[0]
                                  .state)
        charged = np.isin(verdict, _CHARGED)
        r = self.cells.cells
        self.share = (np.bincount(self.cells.cell, weights=charged,
                                  minlength=r)
                      / np.maximum(self.cells.count, 1))
        self.bank = str(path)
        self.states = int(charged.size)
        self.charged = int(charged.sum())
        self.weak_points = (self.cells.coords(self.cells.points[charged])
                            .astype(np.float64))
        self._weak_tree = (cKDTree(self.weak_points)
                           if self.weak_points.shape[0] else None)
        self.names = tuple(SPACES[model_name])
        self.unit = _rho(model)
        self.goal_cap = float(getattr(model.reach, "goal_reach",
                                      model.reach.reaches.target))
        self._tree = cKDTree(self.cells.centres)
        self._full = (self.spec.drones, self.spec.static_obstacles,
                      self.spec.dynamic_obstacles)
        static = config.state.static
        self._config = (config if static is None else dataclasses.replace(
            config, state=dataclasses.replace(
                config.state, static=dataclasses.replace(static,
                                                         min_spacing=0.0))))
        self._models: Dict[Tuple[int, int, int], Any] = {}
        self._field: Optional[_Field] = None
        self._last: Optional[Snapshot] = None
        self._shares: Dict[int, float] = {}
        self.last: Optional[Reading] = None
        self.trace: Optional[_Lead] = None

    @property
    def size(self) -> int:
        return int(self.cells.cells)

    def raw(self, quantiles: np.ndarray) -> Dict[str, float]:
        knots = self.cells.knots
        grid = np.linspace(0.0, 1.0, knots.shape[1])
        return {name: float(np.interp(q, grid, knots[j]))
                for j, (name, q) in enumerate(zip(self.names, quantiles))}

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.name, "bank": self.bank, "states": self.states,
                "charged": self.charged,
                "charged_share": round(self.charged / max(1, self.states), 6),
                "cells": self.size, "features": list(self.names),
                "share": [round(float(v), 6) for v in self.share]}

    def _model(self, shape: Tuple[int, int, int]):
        if shape not in self._models:
            m, ks, kd = shape
            self._models[shape] = build_model(
                self.name, self._config, self.record, drones=m,
                static_obstacles=ks, dynamic_obstacles=kd)
        return self._models[shape]

    def windows(self, snapshots: Sequence[Snapshot], document: Dict[str, Any],
                first: int = 0, last: Optional[int] = None
                ) -> Dict[Tuple[int, int, int], _Group]:
        cutter = _Windows(self.name, self.config, self.spec, self.horizon,
                          document)
        for snapshot in snapshots:
            if snapshot.tick < first or (last is not None
                                         and snapshot.tick > last):
                continue
            cutter.add(snapshot)
        return cutter.groups

    def points(self, state: State) -> np.ndarray:
        m, ks, kd = self._full
        if self.name == "timeout":
            shape = (state.drones, ks, kd)
        else:
            shape = (max(state.drones, m), ks, kd)
        padded = pad(state, shape, self.config)
        model = self._model(shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = window_points(model, padded)
        if self.name == "timeout" and state.drones < 2:
            for name, value in _LONE.items():
                if name in self.names:
                    j = self.names.index(name)
                    rho = _rho(model)
                    out[:, j] = (np.log1p(float(model.reach.reaches.drones)
                                          / rho)
                                 if value == "drone_cap" else value)
        return out

    def judge(self, group: _Group, shape: Tuple[int, int, int]
              ) -> Tuple[np.ndarray, np.ndarray]:
        state = group.state(shape[1], shape[2])
        outcome = self._model(shape).evaluate(state)
        verdict = outcome.verdict
        if self.name == "obstacle_collision" and self._field is not None \
                and self._field.walls and np.any(group.wall):
            verdict = self._wall_contact(verdict, outcome.positions,
                                         np.array(group.wall))
        return verdict, self.points(state)

    def _wall_contact(self, verdict: np.ndarray, end: np.ndarray,
                      walls: np.ndarray) -> np.ndarray:
        physics = self.config.physics
        margin = physics.agent_radius + physics.obstacle_min_gap
        rows = np.flatnonzero(walls & (verdict == int(Verdict.SAFE)))
        if rows.size == 0:
            return verdict
        points = end[rows].reshape(-1, 2)
        hit = np.zeros(points.shape[0], dtype=bool)
        for polygon in self._field.walls:
            nearest, inside = _nearest_on_polygon(points, polygon)
            gap = np.where(inside, 0.0,
                           np.linalg.norm(nearest - points, axis=-1))
            hit |= gap < margin
        hit = hit.reshape(rows.size, -1).any(axis=1)
        out = verdict.copy()
        out[rows[hit]] = int(Verdict.VIOLATES)
        return out

    def read(self, snapshots: Sequence[Snapshot], document: Dict[str, Any], *,
             settle: int, verdict: str = "", verdict_tick: Optional[int] = None,
             crash_kind: str = "") -> Reading:
        self.last = self._read(snapshots, document, settle, verdict,
                               verdict_tick, crash_kind)
        return self.last

    def _read(self, snapshots, document, settle, verdict, verdict_tick,
              crash_kind) -> Reading:
        first = max(0, int(settle) - self.horizon)
        decided = verdict in ("crash", "success") and verdict_tick is not None
        last = int(verdict_tick) - 1 if decided else None
        reading = Reading(self.name)
        self.trace = None
        self._shares = {}
        self._field = _Field(document, self.config.physics.square_bodies)
        self._last = next((s for s in snapshots
                           if verdict_tick is not None
                           and s.tick == int(verdict_tick)), None)
        ticks = {s.tick for s in snapshots
                 if s.tick >= first and (last is None or s.tick <= last)}
        reading.ticks = len(ticks)
        rows: Dict[str, List[Any]] = {k: [] for k in (
            "tick", "verdict", "points", "where", "truncated", "wall",
            "drone", "body", "anchors", "heading")}
        for shape, group in self.windows(snapshots, document, first,
                                         last).items():
            codes, points = self.judge(group, shape)
            rows["tick"].append(np.array(group.tick))
            rows["verdict"].append(codes)
            rows["points"].append(points)
            rows["where"].append(np.array([p.mean(axis=0)
                                           for p in group.positions]))
            rows["truncated"].append(np.array(group.truncated))
            rows["wall"].append(np.array(group.wall))
            rows["drone"].append(np.array([a[0] for a in group.anchors]))
            rows["heading"].append(np.array([v.mean(axis=0)
                                             for v in group.velocities]))
            rows["body"] += list(group.body)
            rows["anchors"] += list(group.anchors)
        size = self.size
        if not rows["tick"]:
            reading.approach = _no_approach(size, len(self.names))
            return self._crash(reading, verdict, verdict_tick, crash_kind,
                               _Lead.empty())
        tick = np.concatenate(rows["tick"])
        codes = np.concatenate(rows["verdict"])
        points = np.concatenate(rows["points"])
        where = np.concatenate(rows["where"])
        reading.windows = int(tick.size)
        reading.truncated = int(np.sum(np.concatenate(rows["truncated"]) > 0))
        reading.walls = int(np.sum(np.concatenate(rows["wall"])))
        reading.infeasible = int(np.sum(codes == int(Verdict.INFEASIBLE)))
        charged = np.isin(codes, _CHARGED)
        doomed = np.unique(tick[codes == int(Verdict.DOOMED)])
        reading.doomed_ticks = int(doomed.size)
        flagged = np.unique(tick[charged])
        if self.name == "timeout":
            flagged = self._stalls(reading, tick, charged, sorted(ticks))
        reading.in_space = bool(flagged.size)
        reading.in_ticks = int(flagged.size)
        reading.first_in = int(flagged[0]) if flagged.size else None
        placed = np.all(np.isfinite(points), axis=1)
        reading.placed = int(placed.sum())
        cell = np.full(tick.size, -1, dtype=int)
        z = np.zeros((tick.size, points.shape[1]))
        if placed.any():
            z[placed] = self.cells.coords(points[placed])
            cell[placed] = self._tree.query(z[placed])[1]
        counts = charged & placed
        if self.name == "timeout":
            counts &= np.isin(tick, flagged)
        for j, n in zip(*np.unique(cell[placed], return_counts=True)):
            reading.cells[str(int(j))] = int(n)
        for j in np.unique(cell[counts]):
            these = counts & (cell == j)
            reading.charged_cells[str(int(j))] = int(these.sum())
            reading.charged_first[str(int(j))] = int(tick[these].min())
        reading.approach = self._approach(
            z[placed], where[placed], tick[placed],
            np.concatenate(rows["drone"])[placed],
            [b for b, p in zip(rows["body"], placed) if p],
            np.concatenate(rows["heading"])[placed])
        self._weak(reading.approach, z[placed], cell[placed], where[placed],
                   [b for b, p in zip(rows["body"], placed) if p],
                   np.concatenate(rows["heading"])[placed],
                   counts[placed], tick.size)
        lead = _Lead(tick=tick, charged=counts, cell=cell, where=where,
                     points=points, body=rows["body"],
                     anchors=rows["anchors"], placed=placed)
        self.trace = lead
        return self._crash(reading, verdict, verdict_tick, crash_kind, lead,
                           flagged, doomed)

    def _approach(self, z: np.ndarray, where: np.ndarray, tick: np.ndarray,
                  drone: np.ndarray, body: List[str],
                  heading: np.ndarray) -> Approach:
        size = self.size
        if z.shape[0] == 0:
            return _no_approach(size, len(self.names))
        distance, index = cKDTree(z).query(self.cells.centres, k=1)
        return Approach(distance=distance.astype(np.float32),
                        where=where[index].astype(np.float32),
                        tick=tick[index].astype(np.int32),
                        drone=drone[index].astype(np.int32),
                        body=[body[i] for i in index],
                        point=z[index].astype(np.float32),
                        heading=heading[index].astype(np.float32))

    def _weak(self, approach: Approach, z: np.ndarray, cell: np.ndarray,
              where: np.ndarray, body: List[str], heading: np.ndarray,
              charged: np.ndarray, salt: int) -> None:
        size = self.size
        f = len(self.names)
        approach.weak_distance = np.full(size, np.inf, dtype=np.float32)
        approach.weak_where = np.zeros((size, 2), dtype=np.float32)
        approach.weak_body = [""] * size
        approach.weak_heading = np.zeros((size, 2), dtype=np.float32)
        approach.weak_point = np.full((size, f), np.nan, dtype=np.float32)
        approach.weak_target = np.full((size, f), np.nan, dtype=np.float32)
        if self._weak_tree is None or z.shape[0] == 0:
            return
        rows = np.arange(z.shape[0])
        if rows.size > WEAK_SAMPLE:
            rng = np.random.default_rng(salt)
            rows = np.union1d(np.flatnonzero(charged),
                              rng.choice(rows, WEAK_SAMPLE, replace=False))
        d, j = self._weak_tree.query(z[rows], k=1)
        order = np.lexsort((d, cell[rows]))
        first = np.ones(order.size, dtype=bool)
        first[1:] = cell[rows][order][1:] != cell[rows][order][:-1]
        for i in order[first]:
            r = int(cell[rows[i]])
            w = rows[i]
            approach.weak_distance[r] = d[i]
            approach.weak_where[r] = where[w]
            approach.weak_body[r] = body[w]
            approach.weak_heading[r] = heading[w]
            approach.weak_point[r] = z[w]
            approach.weak_target[r] = self.weak_points[j[i]]

    def _stalls(self, reading: Reading, tick: np.ndarray,
                charged: np.ndarray, read: List[int]) -> np.ndarray:
        ticks, index = np.unique(tick, return_inverse=True)
        share = (np.bincount(index, weights=charged.astype(float))
                 / np.bincount(index))
        stalled = ticks[share >= STALL_SHARE]
        self._shares = dict(zip(ticks.tolist(), share.tolist()))
        reading.stall_ticks = int(stalled.size)
        reading.stall_share = (round(stalled.size / len(read), 6)
                               if read else None)
        run = best = 0
        stalled_set = set(stalled.tolist())
        for t in read:
            run = run + 1 if t in stalled_set else 0
            best = max(best, run)
        reading.longest_stall = int(best)
        tail = read[-max(1, int(round(TAIL * len(read)))):] if read else []
        reading.final_stall = (round(sum(t in stalled_set for t in tail)
                                     / len(tail), 6) if tail else None)
        return stalled

    def _crash(self, reading: Reading, verdict: str,
               verdict_tick: Optional[int], crash_kind: str, lead: "_Lead",
               flagged: np.ndarray = np.zeros(0, dtype=int),
               doomed: np.ndarray = np.zeros(0, dtype=int)) -> Reading:
        if self.name == "timeout":
            if verdict == "timeout":
                reading.precedes_crash = bool(reading.final_stall)
                tail = sorted(self._shares)[-max(1, int(round(
                    TAIL * len(self._shares)))):] if self._shares else []
                self._attribute(reading, lead, np.isin(lead.tick, tail))
            return reading
        if (verdict == "crash" and verdict_tick is not None
                and crash_kind in CRASH_KINDS[self.name]):
            if self.name == "obstacle_collision" and self._last is not None:
                reading.crash_body = crash_body(self._last, self._field,
                                                self.config)
            decided = int(verdict_tick)
            settled = set(int(t) for t in doomed) - set(int(t)
                                                        for t in flagged)
            while decided - 1 in settled:
                decided -= 1
            reading.precedes_crash = bool(np.any(
                (flagged >= decided - self.horizon)
                & (flagged < int(verdict_tick))))
            window = ((lead.tick >= decided - self.horizon)
                      & (lead.tick < int(verdict_tick)))
            touching = self._touching()
            if touching and window.any():
                mine = np.array([bool(set(a) & touching)
                                 for a in lead.anchors], dtype=bool)
                if np.any(window & mine):
                    window &= mine
            self._attribute(reading, lead, window)
        return reading

    def _touching(self) -> set:
        snapshot = self._last
        if snapshot is None:
            return set()
        physics = self.config.physics
        p = np.asarray(snapshot.positions, dtype=float).reshape(-1, 2)
        if self.name == "agent_collision":
            apart = (np.linalg.norm(p[:, None] - p[None], axis=-1)
                     + np.eye(p.shape[0]) * 1e12)
            return set(np.flatnonzero((apart < physics.agent_boundary)
                                      .any(axis=1)).tolist())
        margin = physics.agent_radius + physics.obstacle_min_gap
        _, _, surface, _ = self._field.seen_from(p)
        gaps = [surface]
        for c, r, _ in snapshot.movers:
            gaps.append(surface_distance(p[:, None], np.asarray(c)[None, None],
                                         np.array([[r]]),
                                         self._field.square))
        gap = np.concatenate(gaps, axis=1) if gaps else np.zeros((len(p), 0))
        if gap.shape[1] == 0:
            return set()
        return set(np.flatnonzero((gap - margin < 0.0).any(axis=1)).tolist())

    def _attribute(self, reading: Reading, lead: "_Lead",
                   window: np.ndarray) -> None:
        usable = window & lead.placed
        if not np.any(usable):
            return
        charged = usable & lead.charged
        pool = charged if np.any(charged) else usable
        labels = lead.cell[pool]
        latest = lead.tick[pool]
        counts = Counter(labels.tolist())
        top = max(counts.values())
        best = max((j for j in counts if counts[j] == top),
                   key=lambda j: latest[labels == j].max())
        chosen = np.flatnonzero(pool)[labels == best]
        pick = chosen[np.argmax(lead.tick[chosen])]
        reading.failure_cell = int(best)
        reading.failure_charged = bool(np.any(charged))
        reading.failure_where = [round(float(v), 6) for v in lead.where[pick]]
        reading.failure_tick = int(lead.tick[pick])
        reading.failure_body = lead.body[pick]
        reading.failure_point = [round(float(v), 6)
                                 for v in lead.points[pick]]


@dataclass
class _Lead:
    tick: np.ndarray
    charged: np.ndarray
    cell: np.ndarray
    where: np.ndarray
    points: np.ndarray
    body: List[str]
    anchors: List[Tuple[int, ...]]
    placed: np.ndarray

    @classmethod
    def empty(cls) -> "_Lead":
        z = np.zeros(0, dtype=int)
        return cls(z, z.astype(bool), z, np.zeros((0, 2)), np.zeros((0, 0)),
                   [], [], z.astype(bool))


def _no_approach(size: int, features: int) -> Approach:
    return Approach(distance=np.full(size, np.inf, dtype=np.float32),
                    where=np.zeros((size, 2), dtype=np.float32),
                    tick=np.full(size, -1, dtype=np.int32),
                    drone=np.full(size, -1, dtype=np.int32),
                    body=[""] * size,
                    point=np.full((size, features), np.nan, dtype=np.float32),
                    heading=np.zeros((size, 2), dtype=np.float32))


def coarse_regions(centres: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from scipy.cluster.vq import kmeans2

    k = max(1, min(int(k), centres.shape[0]))
    _, label = kmeans2(np.asarray(centres, dtype=float), k, iter=50,
                       minit="++", seed=np.random.default_rng([seed, 0xC0A]))
    return label.astype(int)


def _rho(model) -> float:
    physics = model.config.physics
    plant = physics.plant
    h = model.spec.horizon
    if plant.integration == "velocity" and plant.max_accel is not None:
        return plant.max_accel * physics.dt ** 2 * h * (h + 1) / 2.0
    return h * plant.max_step


__all__ = ["Approach", "CRASH_KINDS", "Reading", "STALL_SHARE", "Space",
           "TAIL", "bank_verdicts", "cached_verdicts", "coarse_regions",
           "crash_body", "failure_of", "pad", "space_settings", "spaces_of"]
