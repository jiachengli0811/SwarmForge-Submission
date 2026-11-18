from __future__ import annotations

import collections
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import FAMILIES, OUT

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
FAMILY_COLOUR = dict(zip(FAMILIES, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")))
FAMILY_MARK = dict(zip(FAMILIES, ("o", "s", "^", "D")))


def ci_width(p: float, n: int) -> float:
    return 2 * 1.96 * float(np.sqrt(max(p * (1 - p), 0.0) / n))


def table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def main(argv=None) -> int:
    check = json.loads((OUT / "check.json").read_text())
    doc = json.loads((OUT / "slices.json").read_text())
    near = json.loads((OUT / "neighbourhood.json").read_text())
    slices = doc["slices"]
    n = doc["options"]["samples"]
    lines = []

    lines.append("## 1. The engine against the pipeline, on the sampled states\n")
    lines.append(table(
        ["family", "model", "sampled states", "undecided at a point",
         "wrong at a point", "boxes decided (r = %g steps)" % check[0]["box_radius"],
         "points inside decided boxes", "wrong"],
        [[r["family"], r["model"], r["states"], r["point_undecided"],
          r["point_wrong"], "%d / %d" % (r["box_decided"], r["boxes"]),
          r["box_points"], r["box_wrong"]] for r in check]))

    groups = collections.defaultdict(list)
    for r in slices:
        groups[(r["family"], r["model"], r["kind"])].append(r)
    lines.append("\n## 2. Slices: solver regions against sampling\n")
    rows = []
    for (family, model, kind), g in groups.items():
        lo = np.array([r["solver_share"]["VIOLATES"] for r in g])
        un = np.array([r["undecided_share"] for r in g])
        sa = np.array([r["sampled_share"]["VIOLATES"] for r in g])
        inside = sum(1 for a, b, c in zip(lo, un, sa)
                     if a - ci_width(c, n) / 2 <= c <= a + b + ci_width(c, n) / 2)
        mixed = [r for r in g if r["pieces"] > 0
                 and r["solver_share"]["VIOLATES"] + r["undecided_share"] < 1.0]
        rows.append([
            family, model, kind, len(g),
            "%.2f%%" % (100 * lo.mean()), "%.3f%%" % (100 * np.median(un)),
            "%.2f%%" % (100 * un.max()), "%.2f%%" % (100 * sa.mean()),
            "%.3f%%" % (100 * np.median([ci_width(c, n) for c in sa])),
            "%d / %d" % (inside, len(g)),
            sum(r["contradictions"] for r in g),
            "%.1f" % np.median([r["solver_seconds"] for r in g]),
            "%.1f" % np.median([r["sampling_seconds"] for r in g]),
            len(mixed)])
    lines.append(table(
        ["family", "model", "slice", "slices", "violates, proved (mean)",
         "undecided (median)", "undecided (max)", "violates, sampled (mean)",
         "sampling 95% width (median)", "sampled inside solver bounds",
         "contradictions", "solver s (median)", "sampling s (median)",
         "slices with a boundary"], rows))
    total = sum(r["sampled_in_decided"] for r in slices)
    lines.append("\n%d slices; %d sampled points fell in decided cells, %d "
                 "contradicted the cell's verdict; %d sampled SINGULAR."
                 % (len(slices), total, sum(r["contradictions"] for r in slices),
                    round(sum(r["sampled_share"]["SINGULAR"] for r in slices) * n)))

    lines.append("\n## 3. Violation regions a sample size would have missed\n")
    rows = []
    for (family, model, kind), g in groups.items():
        pieces = sum(r["pieces"] for r in g)
        if not pieces:
            continue
        smallest = min(min(r["piece_areas"]) for r in g if r["pieces"])
        rows.append([family, model, kind, pieces,
                     *("%d" % sum(r["pieces"] - r["pieces_seen"][str(s)]
                                  for r in g) for s in (100, 1000, 10000, n)),
                     "%.3g%%" % (100 * smallest)])
    lines.append(table(
        ["family", "model", "slice", "regions (all slices)",
         "missed by 100 points", "by 1,000", "by 10,000", "by %d" % n,
         "smallest region (share of the square)"], rows))

    lines.append("\n## 4. Full state: the largest certified box round a sampled state\n")
    rows = []
    for r in near:
        for verdict, s in r["by_verdict"].items():
            rows.append([r["family"], r["model"], verdict.lower(), s["states"],
                         r["coordinates"], "%.2g" % s["q25"], "%.2g" % s["median"],
                         "%.2g" % s["q75"], "%.1f%%" % (100 * s["none"])])
    lines.append(table(
        ["family", "model", "sampled verdict", "states", "coordinates widened",
         "radius q25 (steps)", "median", "q75", "none certified"], rows))
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7.2, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for family in FAMILIES:
        g = [r for r in slices if r["family"] == family
             and r["undecided_share"] > 0 and r["sampled_share"]["VIOLATES"] > 0]
        ax.scatter([100 * ci_width(r["sampled_share"]["VIOLATES"], n) for r in g],
                   [100 * r["undecided_share"] for r in g], s=26,
                   marker=FAMILY_MARK[family], color=FAMILY_COLOUR[family],
                   edgecolors=SURFACE, linewidths=0.6, label=family, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo, hi = 1e-3, 100.0
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("Sampling: width of the 95%% interval at %s points (%% of the "
                  "slice)" % format(n, ","), fontsize=9, color="#52514e")
    ax.set_ylabel("Solver: undecided share at depth %d (%% of the slice)"
                  % doc["options"]["max_depth"], fontsize=9, color="#52514e")
    ax.set_title("Uncertainty left on the violating share, per slice",
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "overview.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print((OUT / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
