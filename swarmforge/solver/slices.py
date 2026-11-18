from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

from swarmforge.solve.dynamics import World
from swarmforge.solve.verdict import Verdict

from .common import OUT, model_of, parse_cases
from .engine import UNDECIDED, Engine
from .paving import POSITION, VELOCITY, Slice, pave

_MODELS = {}


def _model(family: str, name: str):
    key = (family, name)
    if key not in _MODELS:
        model = model_of(family, name)
        _MODELS[key] = (model, Engine(model), model.bank().state)
    return _MODELS[key]


def critical_drone(model, state) -> int:
    world = World.of(state)
    model.simulator.tick(world)
    if model.name == "timeout":
        target = state.target[:, None]
        gain = (np.linalg.norm(state.positions - target, axis=-1)
                - np.linalg.norm(world.positions - target, axis=-1))
        return int(np.argmin(gain[0]))
    first, _ = model.pair_members(state)
    margin = model.margins(world, state)[0]
    margin = np.where(np.isnan(margin), np.inf, margin)
    return int(first[int(np.argmin(margin))])


def kinds(model) -> list:
    config = model.config
    out = [POSITION]
    if (config.physics.plant.integration == "velocity"
            and config.state.drone_speed is not None):
        out.append(VELOCITY)
    return out


def square(model, state, kind: str, drone: int, steps: float):
    physics = model.config.physics
    if kind == POSITION:
        half = steps * physics.plant.max_step
        cx, cy = state.positions[0, drone]
    else:
        half = physics.plant.max_step / physics.dt
        cx = cy = 0.0
    return (float(cx - half), float(cx + half), float(cy - half), float(cy + half))


def choose(family: str, name: str, per_verdict: int, seed: int) -> list:
    model, _, state = _model(family, name)
    verdict = model.evaluate(state).verdict
    rng = np.random.default_rng(seed)
    out = []
    for kind in (Verdict.VIOLATES, Verdict.SAFE):
        rows = np.flatnonzero(verdict == int(kind))
        pick = rng.choice(rows, size=min(per_verdict, rows.size), replace=False)
        out += [(int(i), kind.name) for i in np.sort(pick)]
    return out


def run(task) -> dict:
    family, name, index, bank_verdict, kind, options = task
    model, engine, bank = _model(family, name)
    base = bank.take(np.array([index]))
    drone = critical_drone(model, base)
    piece = Slice(base, kind, drone, square(model, base, kind, drone,
                                            options["steps"]))
    x0, x1, y0, y1 = piece.box

    started = time.perf_counter()
    paving = pave(engine, piece, max_depth=options["max_depth"])
    solver_s = time.perf_counter() - started
    grid = paving.raster()
    r = grid.shape[0]

    rng = np.random.default_rng([options["seed"], index, int(kind == VELOCITY)])
    n = options["samples"]
    x, y = rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)
    started = time.perf_counter()
    sampled = model.evaluate(piece.states(x, y)).verdict
    sampling_s = time.perf_counter() - started
    col = np.minimum(((x - x0) / (x1 - x0) * r).astype(int), r - 1)
    row = np.minimum(((y - y0) / (y1 - y0) * r).astype(int), r - 1)
    cell = grid[row, col]
    wrong = (cell != UNDECIDED) & (cell != sampled)

    violating = grid == int(Verdict.VIOLATES)
    labels, found = ndimage.label(violating | (grid == UNDECIDED))
    real = np.unique(labels[violating])
    pieces = int(real.size)
    area = np.bincount(labels[violating], minlength=found + 1) / r ** 2
    piece_area = area[real]
    hits = sampled == int(Verdict.VIOLATES)
    seen = {}
    for size in (100, 1000, 10000, n):
        at = labels[row[:size][hits[:size]], col[:size][hits[:size]]]
        seen[str(size)] = int(np.intersect1d(at, real).size)

    def where(value):
        cx = min(int((value[0] - x0) / (x1 - x0) * r), r - 1)
        cy = min(int((value[1] - y0) / (y1 - y0) * r), r - 1)
        return int(grid[cy, cx])
    here = (base.positions if kind == POSITION else base.velocities)[0, drone]
    inside = bool(x0 <= here[0] <= x1 and y0 <= here[1] <= y1)

    codes = [v for v in Verdict]
    out = {
        "family": family, "model": name, "state": index, "kind": kind,
        "drone": drone, "bank_verdict": bank_verdict, "box": list(piece.box),
        "max_depth": paving.max_depth,
        "solver_share": {v.name: paving.share(int(v)) for v in codes},
        "undecided_share": paving.share(UNDECIDED),
        "cells_asked": paving.asked, "cells": int(paving.code.size),
        "solver_seconds": solver_s,
        "samples": n, "sampling_seconds": sampling_s,
        "sampled_share": {v.name: float(np.mean(sampled == int(v)))
                          for v in codes},
        "sampled_violates_first": {
            str(size): float(np.mean(sampled[:size] == int(Verdict.VIOLATES)))
            for size in (100, 1000, 10000)},
        "sampled_in_decided": int(np.sum(cell != UNDECIDED)),
        "contradictions": int(np.sum(wrong)),
        "undecided_sampled": {v.name: int(np.sum((cell == UNDECIDED)
                                                 & (sampled == int(v))))
                              for v in codes},
        "pieces": int(pieces),
        "piece_areas": [float(a) for a in np.sort(piece_area)[::-1][:20]],
        "pieces_seen": seen,
        "base_cell": where(here) if inside else None,
    }
    folder = OUT / "slices" / family / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / ("%06d-%s.npz" % (index, kind)), grid=grid,
        depth=paving.depth, ix=paving.ix, iy=paving.iy, code=paving.code,
        x=x[:20000].astype(np.float64), y=y[:20000].astype(np.float64),
        sampled=sampled[:20000], box=np.array(piece.box),
        base=np.array(here), meta=np.array(json.dumps(out)))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--per-verdict", type=int, default=6)
    parser.add_argument("--steps", type=float, default=4.0)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=str(OUT / "slices.json"))
    args = parser.parse_args(argv)
    options = {"steps": args.steps, "max_depth": args.max_depth,
               "samples": args.samples, "seed": args.seed}
    tasks = []
    for family, name in parse_cases(args.cases):
        model, _, _ = _model(family, name)
        for index, verdict in choose(family, name, args.per_verdict, args.seed):
            for kind in kinds(model):
                tasks.append((family, name, index, verdict, kind, options))
    print("%d slices" % len(tasks), flush=True)
    rows = []
    started = time.perf_counter()
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(run, tasks):
            rows.append(row)
            print("%-14s %-18s %6d %-8s %-8s solver V %.4f ? %.4f | sampled V "
                  "%.4f | wrong %d | %d cells %.1fs"
                  % (row["family"], row["model"], row["state"],
                     row["bank_verdict"], row["kind"],
                     row["solver_share"]["VIOLATES"], row["undecided_share"],
                     row["sampled_share"]["VIOLATES"], row["contradictions"],
                     row["cells_asked"], row["solver_seconds"]), flush=True)
    rows.sort(key=lambda r: (r["family"], r["model"], r["kind"], r["state"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"options": options, "per_verdict": args.per_verdict,
         "seconds": time.perf_counter() - started, "slices": rows},
        indent=1) + "\n")
    print("wrote %s (%d slices, %.0fs)" % (args.out, len(rows),
                                           time.perf_counter() - started))
    return 1 if any(r["contradictions"] for r in rows) else 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.exit(main())
