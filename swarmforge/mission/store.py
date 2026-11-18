from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA = "swarmforge.mission.case.v1"

Case = Dict[str, Any]


def _mission_only(case: Case) -> Dict:
    return {k: v for k, v in case.items() if not k.startswith("_")}


def digest(case: Case) -> str:
    payload = json.dumps(_mission_only(case), sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


@dataclass
class CaseMeta:
    schema: str = SCHEMA
    index: int = 0
    digest: str = ""
    spawn_seed: int = 0
    field_seed: int = 0
    band: Optional[int] = None
    target_success: Optional[float] = None
    rho: Optional[float] = None
    spacing: Optional[float] = None
    ticks: Optional[int] = None
    arm: str = ""
    region: str = ""
    plan: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def case_name(index: int) -> str:
    return "case-%04d.json" % int(index)


def write_case(directory: str, case: Case, meta: CaseMeta) -> str:
    os.makedirs(directory, exist_ok=True)
    document = dict(_mission_only(case))
    document["_meta"] = meta.to_dict()
    path = os.path.join(directory, case_name(meta.index))
    with open(path, "w") as handle:
        json.dump(document, handle, indent=2)
    return path


def read_case(path: str) -> Tuple[Case, CaseMeta]:
    from ..paths import relocate_path

    path = relocate_path(path)
    with open(path) as handle:
        document = json.load(handle)
    raw = document.pop("_meta", {})
    known = set(CaseMeta.__dataclass_fields__)
    return document, CaseMeta(**{k: v for k, v in raw.items() if k in known})


def write_index(directory: str, metas: Sequence[CaseMeta], *,
                header: Optional[Dict] = None) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "index.json")
    payload = {"schema": "swarmforge.mission.index.v1",
               "config": dict(header or {}),
               "cases": [dict(meta.to_dict(), file=case_name(meta.index))
                         for meta in metas]}
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path


@dataclass
class Result:
    record: str
    case: str
    index: int
    band: Optional[int]
    target_success: Optional[float]
    rho: Optional[float]
    spacing: Optional[float]
    spawn_seed: int
    verdict: str
    failed: bool
    verdict_tick: Optional[int]
    crash_kind: str
    ticks_run: int
    seconds: float
    error: str = ""
    reach: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    arm: str = ""
    behaviors: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


RESULT_FIELDS = tuple(f.name for f in fields(Result)
                      if f.name not in ("reach", "behaviors"))


def _behavior_columns(result: Result) -> Dict[str, Any]:
    out = {}
    for group in ("events", "metrics"):
        for key, value in (result.behaviors.get(group) or {}).items():
            out["b_%s" % key] = value
    return out


def _columns(results: Sequence[Result]) -> Tuple[str, ...]:
    out = list(RESULT_FIELDS)
    for result in results:
        for model, reading in result.reach.items():
            for key, value in reading.items():
                column = "%s_%s" % (model, key)
                if (key != "model" and not isinstance(value, dict)
                        and column not in out):
                    out.append(column)
    for result in results:
        for column in _behavior_columns(result):
            if column not in out:
                out.append(column)
    return tuple(out)


def write_results(directory: str, results: Sequence[Result], *,
                  header: Optional[Dict] = None) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "results.csv")
    ordered: List[Result] = sorted(results, key=lambda r: r.index)
    columns = _columns(ordered)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in ordered:
            row = {k: v for k, v in result.to_dict().items()
                   if k not in ("reach", "behaviors")}
            row["failed"] = int(result.failed)
            for model, reading in result.reach.items():
                for key, value in reading.items():
                    if key != "model" and not isinstance(value, dict):
                        row["%s_%s" % (model, key)] = (
                            int(value) if isinstance(value, bool) else value)
            row.update(_behavior_columns(result))
            writer.writerow({k: ("" if row.get(k) is None else row[k])
                             for k in columns})
    with open(os.path.join(directory, "index.json"), "w") as handle:
        json.dump({"schema": "swarmforge.mission.results.v2",
                   "config": dict(header or {}),
                   "results": [r.to_dict() for r in ordered]},
                  handle, indent=2)
    return path


def read_results(directory: str) -> Tuple[Dict[str, Any], List[Result]]:
    with open(os.path.join(directory, "index.json")) as handle:
        payload = json.load(handle)
    known = set(Result.__dataclass_fields__)
    return payload.get("config", {}), [
        Result(**{k: v for k, v in row.items() if k in known})
        for row in payload["results"]]


__all__ = ["CaseMeta", "RESULT_FIELDS", "Result", "SCHEMA", "case_name",
           "digest", "read_case", "read_results", "write_case",
           "write_index", "write_results"]
