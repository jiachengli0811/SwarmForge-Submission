from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from .candidate import STAGES, Candidate
from .document import Document


class Archive:
    def __init__(self):
        self.candidates: List[Candidate] = []
        self._by_id: Dict[str, Candidate] = {}
        self._by_digest: Dict[str, Candidate] = {}
        self._documents: Dict[str, Document] = {}

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)

    def __contains__(self, candidate_id: str) -> bool:
        return candidate_id in self._by_id

    def add(self, candidate: Candidate,
            document: Optional[Document] = None) -> Candidate:
        if candidate.id in self._by_id:
            raise ValueError("candidate %r is already archived" % candidate.id)
        self.candidates.append(candidate)
        self._by_id[candidate.id] = candidate
        if document is not None:
            self._documents[candidate.id] = document
        if candidate.digest and candidate.stage != "duplicate" \
                and candidate.digest not in self._by_digest:
            self._by_digest[candidate.digest] = candidate
        return candidate

    def get(self, candidate_id: str) -> Candidate:
        return self._by_id[candidate_id]

    def holding(self, digest: str) -> Optional[Candidate]:
        return self._by_digest.get(digest)

    def document(self, candidate: Candidate) -> Document:
        found = self._documents.get(candidate.id)
        if found is None:
            raise KeyError("candidate %r has no record" % candidate.id)
        return found

    def has_document(self, candidate: Candidate) -> bool:
        return candidate.id in self._documents


    def at(self, stage: str) -> List[Candidate]:
        return [c for c in self.candidates if c.stage == stage]

    def measured(self) -> List[Candidate]:
        return self.at("measured")

    def retained(self) -> List[Candidate]:
        return [c for c in self.candidates if c.retained]

    def ranked(self) -> List[Candidate]:
        return sorted(self.measured(),
                      key=lambda c: (c.total if c.total is not None
                                     else float("inf"), c.id))

    def best(self) -> Optional[Candidate]:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def children(self, candidate: Candidate) -> List[Candidate]:
        return [c for c in self.candidates if c.parent == candidate.id]

    def lineage(self, candidate: Candidate) -> List[Candidate]:
        chain = [candidate]
        while chain[-1].parent:
            chain.append(self.get(chain[-1].parent))
        return chain[::-1]

    def stage_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for candidate in self.candidates:
            out[candidate.stage] = out.get(candidate.stage, 0) + 1
        return out

    def operator_counts(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for candidate in self.candidates:
            if candidate.mutation is None:
                continue
            row = out.setdefault(candidate.mutation.operator,
                                 dict({"tried": 0, "retained": 0},
                                      **{stage: 0 for stage in STAGES}))
            row["tried"] += 1
            row[candidate.stage] = row.get(candidate.stage, 0) + 1
            row["retained"] += int(candidate.retained)
        return out

    def best_scores(self) -> Mapping[str, float]:
        out: Dict[str, float] = {}
        for candidate in self.measured():
            if candidate.score is None:
                continue
            for name, value in candidate.score.scores.items():
                out[name] = min(value, out.get(name, value))
        return out


__all__ = ["Archive"]
