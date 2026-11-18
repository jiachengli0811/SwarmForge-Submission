from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

from ..solve.bank import _ARRAYS, Bank, describe, fingerprint, load_bank
from ..solve.config import read_config
from ..solve.models import build_model
from ..solve.regime import MODELS as REGIME_MODELS
from ..solve.state import State, concat
from .features import FEATURES_VERSION, SPACES, window_points

SPACE_VERSION = 1

KNOTS = 1001
NEIGHBOUR = 10
PER_CELL_FIT = 50
FIT_ROWS = 50000


@dataclass(frozen=True)
class SpaceSettings:
    cells: int
    pool: int
    draws: int
    regime: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        if not self.regime:
            out.pop("regime")
        return out


def for_model(settings: SpaceSettings, model) -> SpaceSettings:
    if settings.regime and model.name not in REGIME_MODELS:
        return dataclasses.replace(settings, regime=0.0)
    return settings


@dataclass
class Space:
    names: Tuple[str, ...]
    cell: np.ndarray
    mass: np.ndarray
    count: np.ndarray
    points: np.ndarray
    centres: np.ndarray
    knots: np.ndarray

    @property
    def cells(self) -> int:
        return int(self.mass.size)

    def coords(self, points: np.ndarray) -> np.ndarray:
        return quantile_coords(points, self.knots)

    def assign(self, points: np.ndarray) -> np.ndarray:
        return cKDTree(self.centres).query(self.coords(points))[1] \
            .astype(np.int32)

    def cell_weights(self, tilt: float) -> np.ndarray:
        held = self.count > 0
        w = np.where(held, np.power(np.maximum(self.mass, 1e-300), tilt), 0.0)
        return w / w.sum()

    def state_weights(self, tilt: float) -> np.ndarray:
        return (self.cell_weights(tilt) / np.maximum(self.count, 1))[self.cell]


def quantile_coords(points: np.ndarray, knots: np.ndarray) -> np.ndarray:
    out = np.empty(points.shape, dtype=np.float32)
    for j in range(points.shape[1]):
        lo = np.searchsorted(knots[j], points[:, j], side="left")
        hi = np.searchsorted(knots[j], points[:, j], side="right")
        out[:, j] = 0.5 * (lo + hi) / knots.shape[1]
    return out


def two_nn(z: np.ndarray) -> float:
    d = cKDTree(z).query(z, k=3)[0]
    mu = d[:, 2] / np.maximum(d[:, 1], 1e-12)
    mu = mu[np.isfinite(mu) & (mu > 1.0)]
    return float(mu.size / max(np.sum(np.log(mu)), 1e-12))


def _round_seed(seed: int, round_: int) -> int:
    return int(np.random.SeedSequence([int(seed), 0x5ACE, int(round_)])
               .generate_state(1)[0])


_DRAW: Dict[str, Any] = {}


def _draw_init(config_path: str, sampling, name: str) -> None:
    config = dataclasses.replace(read_config(config_path), sampling=sampling)
    _DRAW["model"] = build_model(name, config)


def _draw_round(job: Tuple[int, int, int, float]
                ) -> Tuple[State, np.ndarray, Dict[str, Any]]:
    seed, round_, n, regime = job
    model = _DRAW["model"]
    state, info = model.sample(n, _round_seed(seed, round_), regime=regime)
    return state, window_points(model, state), info


def _fit_cells(points: np.ndarray, cells: int, seed: int
               ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    knots = np.quantile(points, np.linspace(0.0, 1.0, KNOTS), axis=0) \
        .T.astype(np.float64)
    z = quantile_coords(points, knots).astype(np.float64)
    rng = np.random.default_rng([int(seed), 0xCE11])
    fit = z[rng.choice(z.shape[0], min(FIT_ROWS, z.shape[0]), replace=False)]
    dimension = two_nn(fit)
    centre = fit.mean(axis=0)
    _, _, axes = np.linalg.svd(fit - centre, full_matrices=False)
    dims = int(np.clip(round(dimension), 2, z.shape[1]))
    y = (z - centre) @ axes[:dims].T
    r = cKDTree(y).query(y, k=NEIGHBOUR + 1)[0][:, NEIGHBOUR]
    rare = dims * np.log(np.maximum(r, 1e-9))
    lo, hi = np.quantile(rare, [0.01, 0.99])
    rare = np.clip(rare, lo, hi)
    size = min(z.shape[0], PER_CELL_FIT * cells)
    flat = np.argpartition(-(rare + rng.gumbel(size=rare.size)), size - 1)[:size]
    centres, label = kmeans2(z[flat], cells, iter=50, minit="++",
                             seed=np.random.default_rng([int(seed), 0xC1]))
    used = np.unique(label)
    info = {"intrinsic_dimension": dimension, "density_dims": dims,
            "density_spread": float(hi - lo), "fit_windows": int(size),
            "empty_centres": int(cells - used.size)}
    return knots, centres[used], info


def cell_rank(cell: np.ndarray) -> np.ndarray:
    order = np.argsort(cell, kind="stable")
    sorted_cells = cell[order]
    rank = np.empty(cell.size, dtype=np.int64)
    rank[order] = np.arange(cell.size) - np.searchsorted(
        sorted_cells, sorted_cells, side="left")
    return rank


def _fill(cell: np.ndarray, room: np.ndarray, reserve: np.ndarray
          ) -> Tuple[np.ndarray, np.ndarray]:
    rank = cell_rank(cell)
    into = rank < room[cell]
    held = ~into & (rank < room[cell] + reserve[cell])
    return np.flatnonzero(into), np.flatnonzero(held)


def draw_space(model, settings: SpaceSettings, samples: int, seed: int,
               workers: int = 1) -> Tuple[State, Space, Dict[str, Any]]:
    names = SPACES[model.name]
    config = model.config
    settings = for_model(settings, model)
    started = time.perf_counter()
    jobs = iter((seed, r, settings.pool, settings.regime)
                for r in range(max(1, settings.draws // settings.pool)))
    _DRAW["model"] = model
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=workers, initializer=_draw_init,
            initargs=(str(config.path), config.sampling, model.name))
    try:
        first = _draw_round(next(jobs))
        knots, centres, fit = _fit_cells(first[1], settings.cells, seed)
        tree = cKDTree(centres)
        cells = centres.shape[0]
        if cells > samples:
            raise ValueError("%d cells for a sampling of %d states" % (cells,
                                                                   samples))
        base, rest = divmod(samples, cells)
        quota = np.full(cells, base, dtype=np.int64)
        quota[:rest] += 1
        counts = np.zeros(cells, dtype=np.int64)
        held = np.zeros(cells, dtype=np.int64)
        seen = np.zeros(cells, dtype=np.int64)
        kept: List[Tuple[State, np.ndarray, np.ndarray]] = []
        spare: List[Tuple[State, np.ndarray, np.ndarray]] = []
        infos: List[Dict[str, Any]] = []

        def take(result):
            state, points, info = result
            cell = tree.query(quantile_coords(points, knots))[1]
            np.add.at(seen, cell, 1)
            infos.append(info)
            into, extra = _fill(cell, quota - counts, quota - held)
            for part, store, tally in ((into, kept, counts),
                                       (extra, spare, held)):
                if part.size:
                    store.append((state.take(part), points[part],
                                  cell[part]))
                    np.add.at(tally, cell[part], 1)

        take(first)
        while np.any(counts < quota):
            batch = [job for _, job in zip(range(max(1, workers)), jobs)]
            if not batch:
                break
            results = (pool.map(_draw_round, batch) if pool is not None
                       else map(_draw_round, batch))
            for result in results:
                take(result)
    finally:
        if pool is not None:
            pool.shutdown()
    state, points, cell = _water_fill(kept, spare, samples)
    count = np.bincount(cell, minlength=cells)
    read = int(seen.sum())
    space = Space(names=tuple(names), cell=cell.astype(np.int32),
                  mass=seen / max(read, 1), count=count,
                  points=points.astype(np.float32), centres=centres,
                  knots=knots)
    shares = np.array([i["contested_share"] for i in infos])
    errors = np.array([i["contested_share_se"] for i in infos])
    info = {"focus": infos[0]["focus"], "kept": int(state.size),
            "draws": int(read), "rounds": len(infos),
            "candidates": int(sum(i["candidates"] for i in infos)),
            "contested_share": float(shares.mean()),
            "contested_share_se": float(np.sqrt(np.mean(errors ** 2)
                                                / errors.size)),
            "cells": int(cells), "quota": int(quota.max()),
            "full_cells": int(np.sum(count >= quota)),
            "least_count": int(count.min()),
            **({"regime": settings.regime,
                "regime_kept": float(np.mean([i["regime_kept"]
                                              for i in infos]))}
               if settings.regime else {}),
            "seconds": time.perf_counter() - started, **fit}
    return state, space, info


def _water_fill(kept, spare, samples):
    state = concat([k[0] for k in kept])
    points = np.concatenate([k[1] for k in kept])
    cell = np.concatenate([k[2] for k in kept])
    short = samples - state.size
    if short <= 0 or not spare:
        return state, points, cell
    extra_state = concat([s[0] for s in spare])
    extra_points = np.concatenate([s[1] for s in spare])
    extra_cell = np.concatenate([s[2] for s in spare])
    have = np.bincount(extra_cell)
    lo, hi = 0, int(have.max())
    while lo < hi:
        mid = (lo + hi) // 2
        if np.minimum(have, mid).sum() >= short:
            hi = mid
        else:
            lo = mid + 1
    pick = np.flatnonzero(cell_rank(extra_cell) < lo)[:short]
    return (concat([state, extra_state.take(pick)]),
            np.concatenate([points, extra_points[pick]]),
            np.concatenate([cell, extra_cell[pick]]))


_SPACE = ("space_cell", "space_mass", "space_count", "space_points",
          "space_centres", "space_knots")


def describe_space(model, settings: SpaceSettings, samples: int,
                   seed: int) -> Dict[str, Any]:
    description = describe(model, samples, seed)
    description["space"] = {"version": SPACE_VERSION,
                            "features_version": FEATURES_VERSION,
                            "features": list(SPACES[model.name]),
                            **settings.to_dict()}
    return description


def space_path(model, description: Dict[str, Any]) -> Path:
    return (Path(model.config.sampling.bank_dir) / model.config.name
            / ("%s-space-%s.npz" % (model.name, fingerprint(description))))


def save_space(state: State, space: Space, meta: Dict[str, Any],
               path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(state, name) for name in _ARRAYS}
    arrays.update(space_cell=space.cell, space_mass=space.mass,
                  space_count=space.count, space_points=space.points,
                  space_centres=space.centres, space_knots=space.knots)
    arrays["meta"] = np.array(json.dumps(dict(meta, space_names=list(
        space.names)), sort_keys=True))
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npz")
    os.close(handle)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_space(path) -> Space:
    from ..paths import relocate_path

    with np.load(relocate_path(Path(path)), allow_pickle=False) as data:
        missing = [k for k in _SPACE if k not in data.files]
        if missing:
            raise ValueError("%s is not a space sampling: it has no %s"
                             % (path, ", ".join(missing)))
        meta = json.loads(str(data["meta"]))
        return Space(names=tuple(meta["space_names"]),
                     cell=np.array(data["space_cell"]),
                     mass=np.array(data["space_mass"]),
                     count=np.array(data["space_count"]),
                     points=np.array(data["space_points"]),
                     centres=np.array(data["space_centres"]),
                     knots=np.array(data["space_knots"]))


def space_bank(model, settings: SpaceSettings, samples: Optional[int] = None,
               seed: Optional[int] = None, workers: int = 1
               ) -> Tuple[Bank, Space]:
    sampling = model.config.sampling
    samples = sampling.samples if samples is None else int(samples)
    seed = sampling.seed if seed is None else int(seed)
    settings = for_model(settings, model)
    description = describe_space(model, settings, samples, seed)
    path = space_path(model, description)
    if path.is_file():
        bank = load_bank(path)
        if bank.meta.get("fingerprint") != fingerprint(description):
            raise ValueError("%s holds a sampling drawn from something else; "
                             "remove it" % path)
        return bank, load_space(path)
    state, space, info = draw_space(model, settings, samples, seed, workers)
    meta = {"fingerprint": fingerprint(description),
            "description": description, "sampler": info,
            "seconds": info["seconds"],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_space(state, space, meta, path)
    return Bank(state=state, meta=meta, path=path, created=True), space


__all__ = ["SPACE_VERSION", "Space", "SpaceSettings", "cell_rank", "describe_space",
           "draw_space", "for_model", "load_space", "quantile_coords", "save_space",
           "space_bank", "space_path", "two_nn"]
