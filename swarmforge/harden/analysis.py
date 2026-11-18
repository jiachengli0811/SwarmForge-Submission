from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

Transition = Tuple[str, str, str, str]


@dataclass
class Row:
    id: str
    deltas: Dict[str, float]
    transitions: Tuple[Transition, ...]
    distances: Dict[str, float] = field(default_factory=dict)


def rows_of_session(session) -> Tuple[List[str], List[Row]]:
    baseline = session.baseline.score
    rows = []
    for candidate in session.archive.measured():
        if candidate.id == session.baseline.id or candidate.total is None:
            continue
        deltas = {name: candidate.score.scores[name] - baseline.scores[name]
                  for name in candidate.score.scores}
        deltas["total"] = candidate.total - baseline.total
        comparison = candidate.comparison
        rows.append(Row(
            candidate.id, deltas,
            tuple(comparison.qualitative()) if comparison else (),
            {p: comparison.max_distance(p) or 0.0
             for p in comparison.properties} if comparison else {}))
    return list(session.harden.objective.models), rows


def rows_of_run(directory) -> Tuple[List[str], List[Row]]:
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    baseline = index["baseline"]["score"]
    models = list(baseline["scores"])
    rows = []
    for entry in index["candidates"]:
        if entry["stage"] != "measured" or entry["id"] == "baseline" \
                or not entry.get("file"):
            continue
        meta = json.loads((directory / entry["file"]).read_text())["_meta"]
        score = meta.get("score") or {}
        if not score.get("complete"):
            continue
        deltas = {name: score["scores"][name] - baseline["scores"][name]
                  for name in models}
        deltas["total"] = score["total"] - baseline["total"]
        comparison = meta.get("comparison") or {}
        rows.append(Row(
            entry["id"], deltas,
            tuple(tuple(t) for t in comparison.get("qualitative") or ()),
            {k: v or 0.0 for k, v in
             (comparison.get("max_distance") or {}).items()}))
    return models, rows


def transitions(models: Sequence[str],
                rows: Sequence[Row]) -> List[Dict[str, Any]]:
    found: Dict[Transition, List[Row]] = {}
    for row in rows:
        for move in set(row.transitions):
            found.setdefault(move, []).append(row)
    out = []
    for move, having in found.items():
        ids = {row.id for row in having}
        lacking = [row for row in rows if row.id not in ids]
        record: Dict[str, Any] = {
            "law": move[0], "facet": move[1], "before": move[2],
            "after": move[3], "candidates": len(having),
            "share": len(having) / max(1, len(rows))}
        for name in list(models) + ["total"]:
            with_it = float(np.mean([r.deltas[name] for r in having]))
            record["%s_with" % name] = with_it
            record["%s_without" % name] = (
                float(np.mean([r.deltas[name] for r in lacking]))
                if lacking else None)
        record["best_total"] = min(r.deltas["total"] for r in having)
        record["examples"] = ",".join(sorted(ids)[:5])
        out.append(record)
    out.sort(key=lambda r: (-r["candidates"], r["total_with"]))
    return out


def groups(rows: Sequence[Row]) -> List[Dict[str, Any]]:
    found: Dict[Tuple[Transition, ...], List[Row]] = {}
    for row in rows:
        found.setdefault(tuple(sorted(set(row.transitions))), []).append(row)
    out = []
    for moves, members in found.items():
        totals = [r.deltas["total"] for r in members]
        out.append({
            "candidates": len(members), "best_total": min(totals),
            "mean_total": float(np.mean(totals)),
            "best": min(members, key=lambda r: r.deltas["total"]).id,
            "transitions": "; ".join("%s.%s: %s -> %s" % m for m in moves)
            or "(qualitatively the original)",
            "members": ",".join(sorted(r.id for r in members)[:10])})
    out.sort(key=lambda g: g["best_total"])
    return out


def preservation(rows: Sequence[Row],
                 facets: Sequence[str]) -> Dict[str, float]:
    if not rows:
        return {}
    return {facet: sum(1 for r in rows
                       if not any(t[1] == facet for t in r.transitions))
            / len(rows) for facet in facets}


def _write(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    header: List[str] = []
    for record in records:
        header += [k for k in record if k not in header]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for record in records:
            writer.writerow({k: ("%.6g" % v if isinstance(v, float) else v)
                             for k, v in record.items()})
    return str(path)


def write(directory, models: Sequence[str],
          rows: Sequence[Row]) -> Dict[str, str]:
    directory = Path(directory)
    return {"transitions": _write(directory / "transitions.csv",
                                  transitions(models, rows)),
            "groups": _write(directory / "groups.csv", groups(rows))}


def text(models: Sequence[str], rows: Sequence[Row],
         facets: Sequence[str], top: int = 15) -> str:
    lines = ["%d measured candidates; weakness change against the original "
             "(negative is better)" % len(rows)]
    if not rows:
        return lines[0]
    kept = preservation(rows, facets)
    lines += ["", "  facet kept by    " + "  ".join("%s %.0f%%" % (f, 100 * v)
                                                   for f, v in kept.items())]
    lines += ["", "  %4s  %-40s  %9s %9s  %s" % ("n", "transition", "with",
                                                 "without", "best")]
    for record in transitions(models, rows)[:top]:
        lines.append("  %4d  %-40s  %+9.4f %9s  %+.4f" % (
            record["candidates"],
            ("%s.%s: %s -> %s" % (record["law"], record["facet"],
                                  record["before"], record["after"]))[:40],
            record["total_with"],
            "-" if record["total_without"] is None
            else "%+.4f" % record["total_without"], record["best_total"]))
    found = groups(rows)
    lines += ["", "  %d qualitative signatures among them; the best:"
              % len(found)]
    for group in found[:min(top, 5)]:
        lines.append("  %4d  best %+.4f (%s)  %s" % (
            group["candidates"], group["best_total"], group["best"],
            group["transitions"][:110]))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("run", type=Path)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args(argv)
    models, rows = rows_of_run(args.run)
    index = json.loads((args.run / "index.json").read_text())
    facets = [p["name"] for p in index.get("properties") or ()]
    print(text(models, rows, facets, args.top))
    written = write(args.run, models, rows)
    print("\nwrote %s" % ", ".join(written.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["Row", "groups", "preservation", "rows_of_run", "rows_of_session",
           "text", "transitions", "write"]
