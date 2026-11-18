from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .common import OUT, model_of, parse_cases
from .engine import UNDECIDED, Boxes, Engine

CHUNK = 20000


def steps(model):
    physics, st = model.config.physics, model.config.state
    step = physics.plant.max_step
    return step, (step / physics.dt if st.drone_speed is not None else 0.0)


def classify(engine: Engine, state, position_half=0.0, velocity_half=0.0
             ) -> np.ndarray:
    out = np.empty(state.size, dtype=np.int8)
    for lo in range(0, state.size, CHUNK):
        part = state.take(slice(lo, lo + CHUNK))
        out[lo:lo + CHUNK] = engine.classify(
            Boxes.around(part, position_half, velocity_half))
    return out


def check(family: str, name: str, states: int, radius: float, points: int,
          seed: int) -> dict:
    model = model_of(family, name)
    engine = Engine(model)
    bank = model.bank()
    state = bank.state
    started = time.perf_counter()
    truth = model.evaluate(state).verdict
    pipeline_s = time.perf_counter() - started
    started = time.perf_counter()
    mine = classify(engine, state)
    engine_s = time.perf_counter() - started
    decided = mine != UNDECIDED
    out = {"family": family, "model": name, "states": int(state.size),
           "point_undecided": int(np.sum(~decided)),
           "point_wrong": int(np.sum(decided & (mine != truth))),
           "pipeline_seconds": pipeline_s, "engine_seconds": engine_s}

    rng = np.random.default_rng(seed)
    pick = rng.choice(state.size, size=min(states, state.size), replace=False)
    base = state.take(pick)
    pstep, vstep = steps(model)
    boxed = classify(engine, base, radius * pstep, radius * vstep)
    wrong = inside = 0
    for _ in range(points):
        moved = base.copy()
        moved.positions = base.positions + rng.uniform(
            -1, 1, base.positions.shape) * radius * pstep
        moved.velocities = base.velocities + rng.uniform(
            -1, 1, base.velocities.shape) * radius * vstep
        verdict = model.evaluate(moved).verdict
        sure = boxed != UNDECIDED
        wrong += int(np.sum(sure & (verdict != boxed)))
        inside += int(np.sum(sure))
    out.update(box_radius=radius, boxes=int(pick.size),
               box_decided=int(np.sum(boxed != UNDECIDED)),
               box_points=inside, box_wrong=wrong)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--states", type=int, default=2000)
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(OUT / "check.json"))
    args = parser.parse_args(argv)
    rows = []
    for family, name in parse_cases(args.cases):
        row = check(family, name, args.states, args.radius, args.points,
                    args.seed)
        rows.append(row)
        print("%-14s %-18s points: %d undecided, %d wrong of %d   boxes "
              "r=%g: %d/%d decided, %d wrong of %d points   (%.1fs pipeline, "
              "%.1fs engine)"
              % (family, name, row["point_undecided"], row["point_wrong"],
                 row["states"], row["box_radius"], row["box_decided"],
                 row["boxes"], row["box_wrong"], row["box_points"],
                 row["pipeline_seconds"], row["engine_seconds"]), flush=True)
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
    bad = sum(r["point_wrong"] + r["box_wrong"] for r in rows)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
