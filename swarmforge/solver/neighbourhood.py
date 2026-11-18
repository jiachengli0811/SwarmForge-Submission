from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from swarmforge.solve.verdict import Verdict

from .check import classify, steps
from .common import OUT, model_of, parse_cases
from .engine import Engine


def study(family: str, name: str, states: int, halvings: int, seed: int) -> dict:
    model = model_of(family, name)
    engine = Engine(model)
    bank = model.bank().state
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(bank.size, size=min(states, bank.size),
                              replace=False))
    base = bank.take(pick)
    truth = model.evaluate(base).verdict
    pstep, vstep = steps(model)
    radius = np.zeros(base.size)
    for k in range(halvings, -1, -1):
        r = 2.0 ** -k
        same = classify(engine, base, r * pstep, r * vstep) == truth
        radius = np.where(same, r, radius)
    m = base.drones
    out = {"family": family, "model": name, "states": int(base.size),
           "coordinates": int(2 * m * (2 if vstep else 1)),
           "smallest_radius": 2.0 ** -halvings, "by_verdict": {}}
    for verdict in (Verdict.VIOLATES, Verdict.SAFE):
        r = radius[truth == int(verdict)]
        if not r.size:
            continue
        q = np.quantile(r, [0.25, 0.5, 0.75])
        out["by_verdict"][verdict.name] = {
            "states": int(r.size), "none": float(np.mean(r == 0.0)),
            "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2]),
            "max": float(r.max())}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--states", type=int, default=2000)
    parser.add_argument("--halvings", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(OUT / "neighbourhood.json"))
    args = parser.parse_args(argv)
    rows = []
    for family, name in parse_cases(args.cases):
        row = study(family, name, args.states, args.halvings, args.seed)
        rows.append(row)
        for verdict, s in row["by_verdict"].items():
            print("%-14s %-18s %-8s n=%4d  %2d coords  radius (steps): median "
                  "%.2g  q25 %.2g  q75 %.2g  max %.2g  none %.1f%%"
                  % (family, name, verdict, s["states"], row["coordinates"],
                     s["median"], s["q25"], s["q75"], s["max"],
                     100 * s["none"]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
