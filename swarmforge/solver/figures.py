from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from swarmforge.solve.verdict import Verdict

from .common import OUT, model_of
from .engine import UNDECIDED

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
COLOUR = {int(Verdict.SAFE): "#e9e8e2", int(Verdict.VIOLATES): "#e34948",
          int(Verdict.DOOMED): "#eda100", int(Verdict.INFEASIBLE): "#52514e",
          int(Verdict.SINGULAR): "#6250d6", UNDECIDED: "#2a78d6"}
LABEL = {int(Verdict.SAFE): "safe", int(Verdict.VIOLATES): "violates",
         int(Verdict.DOOMED): "doomed (physics)",
         int(Verdict.INFEASIBLE): "infeasible (starts in contact)",
         int(Verdict.SINGULAR): "singular", UNDECIDED: "undecided by the solver"}
PRIORITY = (int(Verdict.VIOLATES), UNDECIDED, int(Verdict.DOOMED),
            int(Verdict.INFEASIBLE), int(Verdict.SAFE))
SHOWN = 1024


def reduce(grid: np.ndarray, side: int = SHOWN) -> np.ndarray:
    r = grid.shape[0]
    if r <= side:
        return grid
    s = r // side
    blocks = grid.reshape(side, s, side, s)
    out = np.full((side, side), int(Verdict.SAFE), dtype=np.int8)
    for code in reversed(PRIORITY):
        out[np.any(blocks == code, axis=(1, 3))] = code
    return out


def draw(path: Path) -> str:
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    x0, x1, y0, y1 = data["box"]
    model = model_of(meta["family"], meta["model"])
    physics = model.config.physics
    position = meta["kind"] == "position"
    unit = physics.plant.max_step if position else physics.plant.max_step / physics.dt
    cx, cy = (data["base"] if position else (0.0, 0.0))
    extent = [(x0 - cx) / unit, (x1 - cx) / unit, (y0 - cy) / unit, (y1 - cy) / unit]

    order = [int(Verdict.SAFE), int(Verdict.VIOLATES), int(Verdict.DOOMED),
             int(Verdict.INFEASIBLE), int(Verdict.SINGULAR), UNDECIDED]
    index = {code: i for i, code in enumerate(order)}
    cmap = ListedColormap([COLOUR[c] for c in order])
    shown = reduce(data["grid"])
    image = np.vectorize(index.get)(shown)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 6.1), facecolor=SURFACE)
    axes[0].imshow(image, origin="lower", extent=extent, cmap=cmap, vmin=0,
                   vmax=len(order) - 1, interpolation="nearest")
    sampled = data["sampled"]
    axes[1].set_facecolor(SURFACE)
    axes[1].scatter((data["x"] - cx) / unit, (data["y"] - cy) / unit, s=1.2,
                    c=[COLOUR[int(v)] for v in sampled], linewidths=0)
    bx, by = (data["base"][0] - cx) / unit, (data["base"][1] - cy) / unit
    for ax, title in zip(axes, ("Solver: proved cells", "Sampling: %d points"
                                % sampled.size)):
        ax.plot([bx], [by], marker="o", ms=9, mfc="none", mec=INK, mew=1.4)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
        what = ("drone %d %s (steps from the sampled state)" if position
                else "drone %d %s (share of top speed)")
        ax.set_xlabel(what % (meta["drone"], "x" if position else "vx"),
                      fontsize=9, color="#52514e")
        ax.set_ylabel(what % (meta["drone"], "y" if position else "vy"),
                      fontsize=9, color="#52514e")
    present = [c for c in order if np.any(shown == c) or np.any(sampled == c)]
    fig.legend(handles=[Patch(facecolor=COLOUR[c], edgecolor="#c3c2b7",
                              label=LABEL[c]) for c in present],
               loc="lower center", ncol=len(present), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "%s / %s, sampled state %d (%s), %s slice.  Violates: solver %.2f%% to "
        "%.2f%%, sampled %.2f%%"
        % (meta["family"], meta["model"], meta["state"],
           meta["bank_verdict"].lower(), meta["kind"],
           100 * meta["solver_share"]["VIOLATES"],
           100 * (meta["solver_share"]["VIOLATES"] + meta["undecided_share"]),
           100 * meta["sampled_share"]["VIOLATES"]),
        fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    out = (OUT / "figures" / meta["family"] / meta["model"]
           / (path.stem + ".png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return str(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--only", default="", help="a substring of the path")
    args = parser.parse_args(argv)
    paths = sorted(p for p in (OUT / "slices").rglob("*.npz")
                   if args.only in str(p))
    with Pool(args.workers) as pool:
        for i, out in enumerate(pool.imap_unordered(draw, paths)):
            if i % 50 == 0:
                print(i, out, flush=True)
    print("%d figures" % len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
