from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

from .candidate import ORIGINAL

if TYPE_CHECKING:
    from .session import Session


def _pct(value) -> str:
    return "-" if value is None else "%8.4f%%" % (100.0 * value)


def text(session: "Session", top: int = 10) -> str:
    models = list(session.harden.objective.models)
    weights = session.harden.objective.weights
    lines: List[str] = ["=" * 78,
                        "Hardening %s (%s): %d candidates, %d of %d "
                        "evaluations, %.1fs"
                        % (session.config.name, session.record_path.name,
                           len(session.archive) - 1, session.budget.used,
                           session.budget.max_evaluations, session.seconds),
                        "=" * 78, "  Domains each law is read over"]
    for domain in session.domains.values():
        lines.append("    %s" % domain.describe())

    lines += ["", "  Sampling (every candidate is read on the same states)"]
    for name, bank in session.evaluator.describe().items():
        lines.append("    %-20s %7d %s states  %s"
                     % (name, bank["samples"], bank["focus"], bank["path"]))

    baseline = session.baseline
    lines += ["", "  The original, charged share per model (weight)"]
    for name in models:
        lines.append("    %-20s %s  (x%g)" % (name, _pct(
            baseline.score.of(name)), weights[name]))
    lines.append("    %-20s %.6f" % ("weighted total", baseline.total))

    stages = session.archive.stage_counts()
    lines += ["", "  Stages, the original included: %s" % ", ".join(
        "%s %d" % (k, v) for k, v in sorted(stages.items()))]
    lines += _search(session)

    ranked = [c for c in session.archive.ranked() if c.id != ORIGINAL]
    if ranked:
        lines += ["", "  Best %d measured (lower total is better)"
                  % min(top, len(ranked))]
        lines.append("    %-10s %-10s %s  %-10s %s"
                     % ("id", "parent", " ".join("%9s" % m[:9]
                                                  for m in models),
                        "total", "properties changed"))
        for candidate in ranked[:top]:
            lines.append("    %-10s %-10s %s  %-10.6f %s"
                         % (candidate.id, candidate.parent,
                            " ".join("%9.5f" % (candidate.score.of(m) or 0.0)
                                     for m in models),
                            candidate.total,
                            candidate.comparison.summary()[:60]
                            if candidate.comparison else ""))
    shifted: Dict[str, int] = {}
    drifted: Dict[str, int] = {}
    moves: Dict[Tuple[str, str, str, str], List[str]] = {}
    profiled = 0
    for candidate in session.archive:
        if candidate.comparison is None or candidate.id == ORIGINAL:
            continue
        profiled += 1
        for _subject, name, change in candidate.comparison.changes():
            if change.shifted:
                shifted[name] = shifted.get(name, 0) + 1
            elif change.changed:
                drifted[name] = drifted.get(name, 0) + 1
        for row in candidate.comparison.qualitative():
            moves.setdefault(row, []).append(candidate.id)
    if profiled:
        lines += ["", "  Against the original, over %d profiled candidates: "
                  "laws whose class shifted / that only drifted" % profiled]
        for name in [p.name for p in session.properties]:
            lines.append("    %-12s %5d %5d" % (name, shifted.get(name, 0),
                                                drifted.get(name, 0)))
    if moves:
        lines += ["", "  Qualitative changes, most common first"]
        for (subject, name, before, after), ids in sorted(
                moves.items(), key=lambda kv: -len(kv[1]))[:top]:
            lines.append("    %3d x %-14s %-10s %s -> %s"
                         % (len(ids), subject, name, before[:44],
                            after[:44]))
    return "\n".join(lines)


def _spearman(a, b) -> float:
    import numpy as np

    ranks = [np.argsort(np.argsort(np.asarray(x, dtype=float)))
             for x in (a, b)]
    if np.std(ranks[0]) == 0 or np.std(ranks[1]) == 0:
        return float("nan")
    return float(np.corrcoef(ranks[0], ranks[1])[0, 1])


def _search(session: "Session") -> List[str]:
    from .search import property_changes

    pool = session.archive.retained()
    if len(pool) < 2:
        return []
    why: Dict[str, int] = {}
    for candidate in pool:
        reason = str(candidate.annotations.get("accepted") or "kept")
        why[reason] = why.get(reason, 0) + 1
    lines = ["", "  Search: %d rounds; the pool holds %d (%s)"
             % (len(session.rounds), len(pool), ", ".join(
                 "%s %d" % kv for kv in sorted(why.items())))]
    guided = [c for c in session.archive.measured()
              if c.parent and c.annotations.get("guide")
              and c.total is not None]
    if len(guided) >= 3:
        predicted = [c.annotations["guide"]["gain"] for c in guided]
        found = [session.archive.get(c.parent).total - c.total
                 for c in guided]
        lines.append("  Guided: %d measured children were predicted by a "
                     "probe of their parent's states -- Spearman %.2f with "
                     "what measuring them found; probes cost %.1f "
                     "measurements"
                     % (len(guided), _spearman(predicted, found),
                        session.budget.probed))
    best = min(pool, key=lambda c: (c.total if c.total is not None
                                    else float("inf"), c.id))
    lines.append("  The best of the pool, from the original:")
    for candidate in session.archive.lineage(best):
        label = (candidate.mutation.label() if candidate.mutation
                 else "(original)")
        lines.append("    %-10s %-50s total %.6f  %d changed"
                     % (candidate.id, label[:50], candidate.total,
                        property_changes(candidate)))
    return lines


__all__ = ["text"]
