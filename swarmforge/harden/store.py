from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

from .candidate import ORIGINAL, STAGES, Candidate
from .document import Document, strip
from .weakness import VERDICTS

if TYPE_CHECKING:
    from .session import Session

CANDIDATE_SCHEMA = "swarmforge.harden.candidate.v1"
INDEX_SCHEMA = "swarmforge.harden.index.v1"
VERDICTS_FORMAT = 1

CANDIDATES_DIR = "candidates"
VERDICTS_DIR = "verdicts"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w") as handle:
        json.dump(payload, handle, indent=1, default=str)
    os.replace(partial, path)


def _meta(candidate: Candidate) -> Dict[str, Any]:
    meta = dict(candidate.to_dict(), schema=CANDIDATE_SCHEMA)
    if candidate.profile is not None and candidate.id != ORIGINAL:
        touched = ({s.subject for s in candidate.comparison.touched}
                   if candidate.comparison
                   else set(candidate.profile.subjects))
        meta["profile"] = {name: subject.to_dict() for name, subject in
                           candidate.profile.subjects.items()
                           if name in touched}
    return meta


def write_candidate(directory, document: Document,
                    candidate: Candidate) -> str:
    directory = Path(directory)
    payload = strip(document)
    payload["_meta"] = _meta(candidate)
    path = directory / CANDIDATES_DIR / ("%s.json" % candidate.id)
    _atomic_json(path, payload)
    write_verdicts(directory, candidate)
    return str(path)


REACH_INDEX = "__reach_index"
REACH_VALUES = "__reach_values"
REACH_SCORE = "__reach_score"


def write_verdicts(directory, candidate: Candidate) -> Optional[str]:
    models = getattr(candidate.weakness, "models", None) or {}
    arrays = {name: reading.verdicts() for name, reading in models.items()
              if reading.codes}
    if not arrays:
        return None
    for name, reading in models.items():
        weighted = reading.reach() if reading.codes else None
        if weighted is not None:
            arrays[name + REACH_INDEX] = weighted[0]
            arrays[name + REACH_VALUES] = weighted[1]
            arrays[name + REACH_SCORE] = np.array([reading.score])
    folder = Path(directory) / VERDICTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "index.json"
    if not index.exists():
        _atomic_json(index, {
            "format": VERDICTS_FORMAT, "verdicts": list(VERDICTS),
            "banks": {name: {"path": reading.bank,
                             "fingerprint": reading.fingerprint,
                             "samples": reading.samples}
                      for name, reading in models.items()},
            "note": "one int8 array per model in <id>.npz; element i is the "
                    "verdict of the sampling's i-th state, coded as its position "
                    "in 'verdicts'. Under region scoring, <model>%s (int32 "
                    "sampling indices), <model>%s (each one's contribution to "
                    "the score, for the paired noise bar) and <model>%s (the "
                    "score) beside it" % (REACH_INDEX, REACH_VALUES,
                                          REACH_SCORE)})
    path = folder / ("%s.npz" % candidate.id)
    partial = folder / ("%s.partial.npz" % candidate.id)
    np.savez_compressed(partial, **arrays)
    os.replace(partial, path)
    return str(path)


def read_candidate(path) -> Document:
    return strip(json.loads(Path(path).read_text()))


def read_meta(path) -> Dict[str, Any]:
    return dict(json.loads(Path(path).read_text()).get("_meta") or {})


def read_verdicts(directory, candidate_id: str) -> Dict[str, np.ndarray]:
    path = Path(directory) / VERDICTS_DIR / ("%s.npz" % candidate_id)
    with np.load(path) as data:
        return {name: np.array(data[name]) for name in data.files}


def _num(value: Optional[float]) -> str:
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if value != value else "%.8g" % value


def _write(path: Path, header: Sequence[str], rows: Sequence[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header),
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def candidate_rows(session: "Session") -> List[Dict[str, Any]]:
    models = list(session.harden.objective.models)
    properties = [p.name for p in session.properties]
    sites = sorted(session.baseline.expressions)
    baseline = session.baseline.score
    rows = []
    for candidate in session.archive:
        mutation = candidate.mutation
        parent = (session.archive.get(candidate.parent)
                  if candidate.parent else None)
        row: Dict[str, Any] = {
            "id": candidate.id, "round": candidate.round,
            "parent": candidate.parent, "generation": candidate.generation,
            "digest": candidate.digest,
            "operator": mutation.operator if mutation else "",
            "target": mutation.target if mutation else "",
            "args": (", ".join("%s=%s" % kv for kv in
                               sorted(mutation.args.items()))
                     if mutation else ""),
            "source": candidate.source, "stage": candidate.stage,
            "note": candidate.note, "same_as": candidate.same_as,
            "valid_check": (candidate.validity.check
                            if candidate.validity else ""),
            "retained": int(candidate.retained),
            "properties_changed": (candidate.comparison.summary()
                                   if candidate.comparison else ""),
            "total": _num(candidate.total),
            "total_parent": _num(parent.total if parent else None),
            "total_vs_baseline": _num(
                None if candidate.total is None or baseline is None
                else candidate.total - baseline.total),
            "seconds": _num(candidate.weakness.seconds
                            if candidate.weakness else None),
            "created_at": "%.3f" % candidate.created_at,
            "file": (os.path.relpath(candidate.path, str(session.out_dir))
                     if candidate.path and session.out_dir else ""),
            "annotations": (json.dumps(candidate.annotations, default=str,
                                       sort_keys=True)
                            if candidate.annotations else ""),
        }
        for name in properties:
            distance = (candidate.comparison.max_distance(name)
                        if candidate.comparison else None)
            row["prop_%s" % name] = _num(distance)
        for name in models:
            after = candidate.score.of(name) if candidate.score else None
            was = parent.score.of(name) if parent and parent.score else None
            start = baseline.of(name) if baseline else None
            row["%s" % name] = _num(after)
            row["%s_delta" % name] = _num(
                None if after is None or was is None else after - was)
            row["%s_vs_baseline" % name] = _num(
                None if after is None or start is None else after - start)
        for site in sites:
            row["expr_%s" % site] = candidate.expressions.get(site, "")
        rows.append(row)
    return rows


def candidates_csv(session: "Session", path: Path) -> str:
    models = list(session.harden.objective.models)
    header = (["id", "round", "parent", "generation", "digest", "operator",
               "target", "args", "source", "stage", "note", "same_as",
               "valid_check", "retained", "properties_changed"]
              + ["prop_%s" % p.name for p in session.properties]
              + [c for name in models
                 for c in (name, "%s_delta" % name, "%s_vs_baseline" % name)]
              + ["total", "total_parent", "total_vs_baseline"]
              + ["expr_%s" % site
                 for site in sorted(session.baseline.expressions)]
              + ["annotations", "seconds", "created_at", "file"])
    return _write(path, header, candidate_rows(session))


def property_rows(session: "Session") -> List[Dict[str, Any]]:
    rows = []
    for candidate in session.archive:
        if candidate.comparison is None or candidate.id == ORIGINAL:
            continue
        for subject in candidate.comparison.touched:
            for name, change in subject.changes.items():
                rows.append({
                    "id": candidate.id, "subject": subject.subject,
                    "status": subject.status, "property": name,
                    "changed": ("" if change.changed is None
                                else int(change.changed)),
                    "shifted": ("" if change.shifted is None
                                else int(change.shifted)),
                    "drift": _num(change.drift),
                    "distance": _num(change.distance),
                    "before": change.before, "after": change.after,
                    "detail": change.detail})
    return rows


def properties_csv(session: "Session", path: Path) -> str:
    header = ["id", "subject", "status", "property", "changed",
              "shifted", "drift", "distance", "before", "after", "detail"]
    return _write(path, header, property_rows(session))


def signature_rows(session: "Session") -> List[Dict[str, Any]]:
    from .properties import signature

    rows = []
    for candidate in session.archive:
        if candidate.profile is None:
            continue
        for subject, classes in signature(candidate.profile).items():
            row = {"id": candidate.id, "subject": subject,
                   "stage": candidate.stage,
                   "total": _num(candidate.total)}
            row.update(classes)
            rows.append(row)
    return rows


def signatures_csv(session: "Session", path: Path) -> str:
    names = [p.name for p in session.properties]
    header = ["id", "subject", "stage", "total"] + names
    return _write(path, header, signature_rows(session))


def rounds_csv(session: "Session", path: Path) -> str:
    models = list(session.harden.objective.models)
    stages = STAGES
    rows = []
    for summary in session.rounds:
        row = {"round": summary.index, "label": summary.label,
               "retained": summary.retained,
               "evaluations": summary.evaluations,
               "best_total": _num(summary.best_total),
               "seconds": _num(summary.seconds)}
        for stage in stages:
            row[stage] = summary.stages.get(stage, 0)
        for name in models:
            row["best_%s" % name] = _num(summary.best_scores.get(name))
        rows.append(row)
    header = (["round", "label"] + list(stages)
              + ["retained", "evaluations", "best_total"]
              + ["best_%s" % name for name in models] + ["seconds"])
    return _write(path, header, rows)


def operators_csv(session: "Session", path: Path) -> str:
    from .operators import OPERATORS

    counts = session.archive.operator_counts()
    stages = STAGES
    rows = []
    for name in sorted(counts):
        found = OPERATORS.get(name)
        rows.append(dict(counts[name], operator=name,
                         family=getattr(found, "family", ""),
                         target=getattr(found, "target", ""),
                         action=getattr(found, "action", "")))
    header = (["operator", "family", "target", "action", "tried"]
              + list(stages) + ["retained"])
    return _write(path, header, rows)


def write_index(session: "Session", path: Path) -> str:
    payload = {
        "schema": INDEX_SCHEMA,
        "inputs": {"config": str(Path(session.config.path).resolve()),
                   "record": str(session.record_path)},
        "family": session.config.name,
        "harden": session.harden.to_dict(),
        "calibration": getattr(session, "calibration", None),
        "budget": session.budget.to_dict(),
        "seconds": round(session.seconds, 3),
        "closed": bool(getattr(session, "closed", True)),
        "domains": {name: domain.to_dict()
                    for name, domain in session.domains.items()},
        "banks": session.evaluator.describe(),
        "properties": [{"name": p.name, "description": p.description}
                       for p in session.properties],
        "scale": session.scale.to_dict(),
        "reference": session.reference.to_dict(),
        "baseline": _meta(session.baseline),
        "rounds": [summary.to_dict() for summary in session.rounds],
        "stages": session.archive.stage_counts(),
        "candidates": [{"id": c.id, "parent": c.parent, "stage": c.stage,
                        "total": c.total, "retained": c.retained,
                        "changed": (c.comparison.summary()
                                    if c.comparison else ""),
                        "mutation": (c.mutation.to_dict() if c.mutation
                                     else None),
                        "file": (os.path.relpath(c.path, str(session.out_dir))
                                 if c.path else "")}
                       for c in session.archive],
    }
    if getattr(session, "resumed", None) is not None:
        payload["resumed"] = session.resumed
    _atomic_json(path, payload)
    return str(path)


def _analysis(session: "Session", directory: Path) -> Dict[str, str]:
    from . import analysis
    models, rows = analysis.rows_of_session(session)
    return analysis.write(directory, models, rows)


def write_all(directory, session: "Session") -> Dict[str, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for candidate in session.archive:
        if session.archive.has_document(candidate):
            candidate.path = str(directory / CANDIDATES_DIR
                                 / ("%s.json" % candidate.id))
            payload = strip(session.archive.document(candidate))
            payload["_meta"] = _meta(candidate)
            _atomic_json(Path(candidate.path), payload)
    return {
        "candidates": candidates_csv(session, directory / "candidates.csv"),
        "properties": properties_csv(session, directory / "properties.csv"),
        "signatures": signatures_csv(session, directory / "signatures.csv"),
        **_analysis(session, directory),
        "rounds": rounds_csv(session, directory / "rounds.csv"),
        "operators": operators_csv(session, directory / "operators.csv"),
        "index": write_index(session, directory / "index.json"),
    }


__all__ = ["CANDIDATE_SCHEMA", "INDEX_SCHEMA", "candidate_rows",
           "property_rows", "read_candidate", "read_meta", "read_verdicts",
           "signature_rows",
           "write_all", "write_candidate", "write_index", "write_verdicts"]
