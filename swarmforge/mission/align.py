from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .measure import failure_of
from .store import Result, read_results

GROUPS = ("this", "other", "success")

ALIGN_FIELDS = (
    "record", "model", "group", "flights", "in_space", "in_space_rate",
    "weak", "weak_rate", "median_risk", "precision_in", "recall_in",
    "accuracy_in", "precision_weak", "recall_weak", "accuracy_weak",
    "auc_risk", "precedes_crash", "median_lead_in")


def group_of(result: Result, model: str) -> str:
    if failure_of(model, result.verdict, result.crash_kind):
        return "this"
    return "success" if result.verdict == "success" else "other"


def auc(positive: Sequence[float], negative: Sequence[float]) -> Optional[float]:
    if not len(positive) or not len(negative):
        return None
    pos = np.asarray(positive, dtype=float)[:, None]
    neg = np.asarray(negative, dtype=float)[None, :]
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def _rates(hit: np.ndarray, truth: np.ndarray) -> Dict[str, Optional[float]]:
    tp = int(np.sum(hit & truth))
    fp = int(np.sum(hit & ~truth))
    fn = int(np.sum(~hit & truth))
    tn = int(np.sum(~hit & ~truth))
    return {"precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "accuracy": (tp + tn) / truth.size if truth.size else None}


def _median(values: List[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def _weak(share: Sequence[float]) -> set:
    share = np.asarray(share, dtype=float)
    if share.sum() <= 0.0:
        return set()
    order = np.argsort(-share, kind="stable")
    cumulative = np.cumsum(share[order]) / share.sum()
    return set(order[:int(np.searchsorted(cumulative, 0.5) + 1)].tolist())


def align(results: Sequence[Result], model: str,
          share: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
    flown = [r for r in results if r.verdict != "error" and model in r.reach]
    rows: List[Dict[str, Any]] = []
    record = flown[0].record if flown else ""
    share = np.asarray(share if share is not None else [], dtype=float)
    weak = _weak(share) if share.size else set()

    def cells(r: Result) -> List[int]:
        return [int(c) for c in r.reach[model].get("cells", {})]

    def risk(r: Result) -> float:
        return max((float(share[c]) for c in cells(r) if c < share.size),
                   default=0.0)

    def entered(r: Result) -> bool:
        return any(c in weak for c in cells(r))

    def group_row(name: str, members: List[Result]) -> Dict[str, Any]:
        inside = sum(1 for r in members if r.reach[model]["in_space"])
        hit = sum(1 for r in members if entered(r))
        n = len(members)
        return {"record": record, "model": model, "group": name,
                "flights": n, "in_space": inside,
                "in_space_rate": round(inside / n, 4) if n else None,
                "weak": hit, "weak_rate": round(hit / n, 4) if n else None,
                "median_risk": _median([risk(r) for r in members])}

    for name in GROUPS:
        rows.append(group_row(name, [r for r in flown
                                     if group_of(r, model) == name]))
    total = group_row("all", flown)
    truth = np.array([group_of(r, model) == "this" for r in flown])
    for key, test in (("in", lambda r: bool(r.reach[model]["in_space"])),
                      ("weak", entered)):
        hit = np.array([test(r) for r in flown], dtype=bool)
        for name, value in _rates(hit, truth).items():
            total["%s_%s" % (name, key)] = (None if value is None
                                            else round(value, 4))
    score = [risk(r) for r in flown]
    area = auc([s for s, t in zip(score, truth) if t],
               [s for s, t in zip(score, truth) if not t])
    total["auc_risk"] = None if area is None else round(area, 4)
    crashes = [r for r in flown if group_of(r, model) == "this"]
    before = [r.reach[model]["precedes_crash"] for r in crashes
              if r.reach[model].get("precedes_crash") is not None]
    total["precedes_crash"] = (round(sum(before) / len(before), 4)
                               if before else None)
    total["median_lead_in"] = _median([
        r.verdict_tick - r.reach[model]["first_in"] for r in crashes
        if r.reach[model].get("first_in") is not None])
    rows.append(total)
    return rows


def alignment_rows(results: Dict[str, Sequence[Result]],
                   spaces: Dict[str, Dict[str, Dict[str, Any]]]
                   ) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, flown in results.items():
        for model in sorted({m for r in flown for m in r.reach}):
            share = spaces.get(name, {}).get(model, {}).get("share")
            rows.extend(align(flown, model, share))
    return rows


def write_alignment(directory: str, rows: List[Dict[str, Any]]) -> str:
    path = os.path.join(directory, "alignment.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALIGN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k])
                             for k in ALIGN_FIELDS})
    return path


def text(rows: List[Dict[str, Any]]) -> str:
    def cell(value, spec="%.3f"):
        return "-" if value is None else (spec % value if isinstance(
            value, float) else str(value))
    out = ["%-12s %-18s %-12s %7s %9s %9s %9s" % (
        "record", "model", "group", "flights", "in_space", "weak",
        "med_risk")]
    for row in rows:
        out.append("%-12s %-18s %-12s %7s %9s %9s %9s" % (
            row["record"][:12], row["model"], row["group"], row["flights"],
            cell(row["in_space_rate"]), cell(row["weak_rate"]),
            cell(row["median_risk"], "%.4f")))
        if row["group"] == "all":
            out.append(
                "%14s in_space P %s R %s | weak P %s R %s | AUC risk %s | "
                "precedes %s | lead in %s" % (
                    "", cell(row.get("precision_in")),
                    cell(row.get("recall_in")),
                    cell(row.get("precision_weak")),
                    cell(row.get("recall_weak")),
                    cell(row.get("auc_risk")),
                    cell(row.get("precedes_crash")),
                    cell(row.get("median_lead_in"), "%.0f")))
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("campaign", type=Path,
                        help="a swarmforge.mission.run output directory")
    args = parser.parse_args(argv)
    results: Dict[str, List[Result]] = {}
    spaces: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for index in sorted(args.campaign.glob("*/index.json")):
        if index.parent.name == "missions":
            continue
        header, flown = read_results(str(index.parent))
        name = header.get("record", index.parent.name)
        results[name] = flown
        spaces[name] = header.get("spaces", {})
    rows = alignment_rows(results, spaces)
    path = write_alignment(str(args.campaign), rows)
    print(text(rows))
    print("-> %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["ALIGN_FIELDS", "GROUPS", "align", "alignment_rows", "auc",
           "group_of", "text", "write_alignment"]
