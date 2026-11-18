from __future__ import annotations

import ast
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

import numpy as np

from .candidate import ORIGINAL, Candidate
from .document import Document
from .operators import Mutation
from .store import read_candidate, read_meta, read_verdicts, write_candidate
from .validate import Validity
from .weakness import (ModelReading, WeaknessReading, model_reading,
                       score_of)

if TYPE_CHECKING:
    from .session import Session

SCREENED_AGAIN = ("screened", "measured")

SPOT_CHECK = 8


def check_folders(run_dir, out_dir) -> None:
    run_dir = Path(run_dir)
    if not (run_dir / "index.json").is_file():
        raise ValueError("%s holds no finished run: no index.json (a run "
                         "writes it when it closes)" % run_dir)
    if out_dir is not None and Path(out_dir).resolve() == run_dir.resolve():
        raise ValueError("a resumed run writes to a folder of its own: %s "
                         "is the run it carries on" % run_dir)


def carry(session: "Session", run_dir) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    check_folders(run_dir, session.out_dir)
    if len(session.archive) != 1:
        raise ValueError("a run is resumed into a session that has made "
                         "nothing yet")
    index = json.loads((run_dir / "index.json").read_text())
    _same_original(session, index)
    _same_verdicts(session, run_dir)
    previous = {k: float(v) for k, v in
                index["harden"]["objective"].items()}
    rescored = previous != session.harden.objective.to_dict()
    rows = _table(run_dir)
    restaged: Dict[str, str] = {}
    documents: Dict[str, Document] = {}
    out_dir, session.out_dir = session.out_dir, None
    try:
        for entry in index["candidates"]:
            meta = (read_meta(run_dir / entry["file"]) if entry.get("file")
                    else {})
            if entry["id"] == ORIGINAL:
                _annotate(session.baseline, meta)
                session.baseline.retained = bool(meta.get("retained"))
                continue
            if not meta:
                candidate = _unfiled(session, entry, rows.get(entry["id"]))
            else:
                documents[entry["id"]] = read_candidate(run_dir
                                                        / entry["file"])
                if meta["stage"] in SCREENED_AGAIN:
                    candidate = _screen(session, run_dir, meta,
                                        documents[entry["id"]])
                else:
                    candidate = _filed(session, meta,
                                       documents[entry["id"]])
                if candidate.stage != meta["stage"]:
                    restaged[candidate.id] = "%s -> %s" % (meta["stage"],
                                                           candidate.stage)
    finally:
        session.out_dir = out_dir
    _spot_check(session)
    _session_state(session, index)
    if out_dir is not None:
        for candidate in session.archive:
            if candidate.id in documents:
                candidate.path = write_candidate(
                    out_dir, documents[candidate.id], candidate)
    return {"from": str(run_dir.resolve()),
            "candidates": len(session.archive) - 1,
            "stages": session.archive.stage_counts(),
            "pool": sum(1 for c in session.archive if c.retained),
            "restaged": restaged,
            "budget": dict(index.get("budget") or {}),
            "objective": previous, "rescored": rescored,
            "seed": session.rng_seed}


def _same_original(session: "Session", index: Mapping[str, Any]) -> None:
    saved = (index.get("baseline") or {}).get("digest")
    if saved != session.baseline.digest:
        raise ValueError("the run hardened another original (%s; this one is "
                         "%s): resume it with the --original it was given"
                         % (saved, session.baseline.digest))


def _same_verdicts(session: "Session", run_dir: Path) -> None:
    banks = json.loads((run_dir / "verdicts" / "index.json").read_text()
                       )["banks"]
    saved = read_verdicts(run_dir, ORIGINAL)
    for name, reading in session.baseline.weakness.models.items():
        if name not in banks or name not in saved:
            raise ValueError("the run did not read %s: its candidates have "
                             "no verdicts to score on it" % name)
        if banks[name]["fingerprint"] != reading.fingerprint:
            raise ValueError("%s reads sampling %s now and the run read %s: its "
                             "verdicts are of other states"
                             % (name, reading.fingerprint,
                                banks[name]["fingerprint"]))
        moved = int(np.sum(saved[name] != reading.verdicts()))
        if moved:
            raise ValueError("%s judges the original differently now, on %d "
                             "of %d states: the model changed since the run"
                             % (name, moved, saved[name].size))


def _spot_check(session: "Session") -> None:
    ranked = [c for c in session.archive.ranked()
              if c.id != ORIGINAL and c.validity is not None
              and c.validity.record is not None]
    half = SPOT_CHECK // 2
    rest = ranked[half:]
    picks = ranked[:half] + rest[::max(1, len(rest) // half)][:half]
    if not picks:
        return
    readings = session.evaluator.measure(
        [(c.id, session.document(c)) for c in picks],
        [c.validity.record for c in picks])
    for candidate, reading in zip(picks, readings):
        for name, model in reading.models.items():
            carried = candidate.weakness.models.get(name)
            if carried is None or not carried.codes or not model.codes:
                continue
            moved = int(np.sum(model.verdicts() != carried.verdicts()))
            if moved:
                raise ValueError("%s judges %s differently now, on %d of %d "
                                 "states: the model changed since the run"
                                 % (name, candidate.id, moved,
                                    carried.verdicts().size))


def _parent(session: "Session", parent: str) -> Candidate:
    if parent not in session.archive:
        raise ValueError("the run filed a candidate of %s before %s itself"
                         % (parent, parent))
    return session.archive.get(parent)


def _annotate(candidate: Candidate, meta: Mapping[str, Any]) -> None:
    candidate.annotations = dict(meta.get("annotations") or {})
    candidate.round = int(meta.get("round") or 0)
    candidate.created_at = float(meta.get("created_at")
                                 or candidate.created_at)


def _reading(raw: Mapping[str, Any],
             verdicts: Mapping[str, np.ndarray]) -> WeaknessReading:
    models = {}
    for name, found in (raw.get("models") or {}).items():
        common = dict(model=name, samples=int(found.get("samples") or 0),
                      bank=str(found.get("bank") or ""),
                      fingerprint=str(found.get("fingerprint") or ""),
                      focus=str(found.get("focus") or ""),
                      seconds=float(found.get("seconds") or 0.0))
        if found.get("error") or name not in verdicts:
            models[name] = ModelReading(error=str(found.get("error")
                                                  or "no verdicts saved"),
                                        **common)
        else:
            from .store import REACH_INDEX, REACH_SCORE, REACH_VALUES
            weighted = ((verdicts[name + REACH_INDEX],
                         verdicts[name + REACH_VALUES],
                         float(verdicts[name + REACH_SCORE][0]))
                        if name + REACH_INDEX in verdicts else None)
            models[name] = model_reading(verdicts[name], weighted,
                                         str(found.get("scoring") or "region"),
                                         **common)
    return WeaknessReading(models, float(raw.get("seconds") or 0.0),
                           raw.get("started_at"), raw.get("finished_at"))


def _screen(session: "Session", run_dir: Path, meta: Mapping[str, Any],
            document: Document) -> Candidate:
    mutation = (Mutation.from_dict(meta["mutation"]) if meta.get("mutation")
                else None)
    candidate = session.consider(
        document, parent=_parent(session, meta.get("parent") or ORIGINAL),
        mutation=mutation, source=str(meta.get("source") or ""),
        label=meta["id"], fixed=mutation is not None)
    if candidate.digest and candidate.digest != meta.get("digest"):
        raise ValueError("%s reads back as controller %s, the run held %s"
                         % (meta["id"], candidate.digest, meta.get("digest")))
    _annotate(candidate, meta)
    if candidate.stage == meta["stage"]:
        candidate.note = str(meta.get("note") or "")
    if meta["stage"] == "measured":
        candidate.weakness = _reading(meta.get("weakness") or {},
                                      read_verdicts(run_dir, meta["id"]))
        candidate.score = score_of(candidate.weakness,
                                   session.harden.objective)
        if candidate.stage == "screened":
            candidate.stage = "measured"
            candidate.note = str(meta.get("note") or "")
            candidate.retained = bool(meta.get("retained"))
    return candidate


def _validity(raw: Optional[Mapping[str, Any]]) -> Optional[Validity]:
    if not raw:
        return None
    return Validity(bool(raw.get("valid")), str(raw.get("check") or ""),
                    str(raw.get("reason") or ""))


def _filed(session: "Session", meta: Mapping[str, Any],
           document: Optional[Document]) -> Candidate:
    parent = meta.get("parent") or ""
    candidate = Candidate(
        id=meta["id"], parent=parent,
        parent_digest=(_parent(session, parent).digest if parent
                       else str(meta.get("parent_digest") or "")),
        generation=int(meta.get("generation") or 0),
        mutation=(Mutation.from_dict(meta["mutation"])
                  if meta.get("mutation") else None),
        source=str(meta.get("source") or ""),
        digest=str(meta.get("digest") or ""), stage=meta["stage"],
        note=str(meta.get("note") or ""),
        same_as=str(meta.get("same_as") or ""),
        validity=_validity(meta.get("validity")),
        expressions=dict(meta.get("expressions") or {}))
    _annotate(candidate, meta)
    session.archive.add(candidate, document)
    return candidate


def _unfiled(session: "Session", entry: Mapping[str, Any],
             row: Optional[Mapping[str, str]]) -> Candidate:
    row = row or {}
    parent = _parent(session, entry.get("parent") or ORIGINAL)
    mutation = (Mutation.from_dict(entry["mutation"])
                if entry.get("mutation") else _recipe(row))
    candidate = Candidate(
        id=entry["id"], parent=parent.id, parent_digest=parent.digest,
        generation=parent.generation + 1, mutation=mutation,
        stage=entry["stage"], note=row.get("note", ""),
        round=int(row.get("round") or 0),
        created_at=float(row.get("created_at") or 0.0))
    session.archive.add(candidate, None)
    return candidate


def _table(run_dir: Path) -> Dict[str, Dict[str, str]]:
    path = run_dir / "candidates.csv"
    if not path.is_file():
        return {}
    with open(path, newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def _recipe(row: Mapping[str, str]) -> Optional[Mutation]:
    if not row.get("operator"):
        return None
    target = row.get("target") or ""
    args: Dict[str, Any] = {}
    for part in re.split(r",\s*(?=[A-Za-z_]\w*=)", row.get("args") or ""):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            args[key.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            args[key.strip()] = value.strip()
    return Mutation(row["operator"], target if target not in ("", "-")
                    else None, None, args)


def _session_state(session: "Session", index: Mapping[str, Any]) -> None:
    from .session import Round

    numbers = [int(found.group(1)) for found in
               (re.fullmatch(r"cand-(\d+)", c.id) for c in session.archive)
               if found]
    session._next = max(numbers, default=0)
    for candidate in session.archive:
        if candidate.mutation is not None and candidate.parent:
            parent = session.archive.get(candidate.parent)
            session._screened[(parent.digest, candidate.mutation.key())] = \
                candidate
    spent = index.get("budget") or {}
    session.budget.used = int(spent.get("used") or 0)
    session.budget.probed = float(spent.get("probed") or 0.0)
    session.rounds = [
        Round(index=int(r["round"]), label=str(r.get("label") or ""),
              stages=dict(r.get("stages") or {}),
              retained=int(r.get("retained") or 0),
              best_total=r.get("best_total"),
              best_scores=dict(r.get("best_scores") or {}),
              evaluations=int(r.get("evaluations") or 0),
              seconds=float(r.get("seconds") or 0.0))
        for r in index.get("rounds") or ()]
    session._round_from = len(session.archive.candidates)
    session._round_used = session.budget.used
    session._round_started = time.time()
    session.rng_seed = "%d+%d" % (int(session.harden.seed),
                                  len(session.archive))
    session.rng = random.Random(session.rng_seed)


__all__ = ["SCREENED_AGAIN", "carry", "check_folders"]
