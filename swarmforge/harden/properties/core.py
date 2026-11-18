from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

from ..domain import Domain

PROFILE_FORMAT = 6


class Undecided(Exception):
    pass


@dataclass(frozen=True)
class Change:
    changed: Optional[bool]
    distance: Optional[float] = None
    detail: str = ""
    before: str = ""
    after: str = ""
    shifted: Optional[bool] = None
    drift: Optional[float] = None
    classes: Optional[Tuple[str, str]] = None

    @property
    def verdict(self) -> str:
        if self.changed is None:
            return "undecided"
        if self.shifted:
            return "shifted"
        return "changed" if self.changed else "same"

    def to_dict(self) -> Dict[str, Any]:
        return {"changed": self.changed,
                "distance": (None if self.distance is None
                             else round(self.distance, 6)),
                "shifted": self.shifted,
                "drift": None if self.drift is None else round(self.drift, 6),
                "classes": None if self.classes is None else list(
                    self.classes),
                "detail": self.detail, "before": self.before,
                "after": self.after}


class Property:
    name = ""
    description = ""

    def extract(self, analysis: Any) -> Any:
        raise NotImplementedError

    def compare(self, before: Any, after: Any) -> Change:
        same = before == after
        return Change(not same, 0.0 if same else 1.0,
                      "" if same else "changed",
                      self.text(before), self.text(after))

    def text(self, value: Any) -> str:
        return str(value)


@dataclass
class SubjectProfile:
    subject: str
    key: str
    text: str
    axes: Tuple[str, ...] = ()
    spans: Tuple[Tuple[float, float], ...] = ()
    values: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "key": self.key,
                "text": self.text, "axes": list(self.axes),
                "spans": [list(span) for span in self.spans],
                "values": dict(self.values), "errors": dict(self.errors)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SubjectProfile":
        return cls(str(raw["subject"]), str(raw["key"]),
                   str(raw.get("text", "")),
                   tuple(raw.get("axes") or ()),
                   tuple(tuple(span) for span in raw.get("spans") or ()),
                   dict(raw.get("values") or {}),
                   dict(raw.get("errors") or {}))


@dataclass
class PropertyProfile:
    label: str
    subjects: Dict[str, SubjectProfile] = field(default_factory=dict)
    properties: Tuple[str, ...] = ()

    def value(self, subject: str, prop: str) -> Any:
        return self.subjects[subject].values.get(prop)

    def to_dict(self) -> Dict[str, Any]:
        return {"format": PROFILE_FORMAT, "label": self.label,
                "properties": list(self.properties),
                "subjects": {name: profile.to_dict()
                             for name, profile in self.subjects.items()}}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PropertyProfile":
        if int(raw.get("format", 0)) != PROFILE_FORMAT:
            raise ValueError("a profile of format %r, not %d"
                             % (raw.get("format"), PROFILE_FORMAT))
        return cls(str(raw.get("label", "")),
                   {name: SubjectProfile.from_dict(subject)
                    for name, subject in (raw.get("subjects") or {}).items()},
                   tuple(raw.get("properties") or ()))


def signature(profile: PropertyProfile,
              properties: Optional[Sequence[str]] = None
              ) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for name, subject in profile.subjects.items():
        classes = {prop: value["class"] for prop, value in
                   subject.values.items()
                   if isinstance(value, Mapping) and "class" in value
                   and (properties is None or prop in properties)}
        if classes:
            out[name] = classes
    return out


class Profiler:
    def __init__(self, domains: Mapping[str, Domain],
                 properties: Sequence[Property], scale: Any):
        self.domains = dict(domains)
        self.properties: Tuple[Property, ...] = tuple(properties)
        self.scale = scale
        names = [p.name for p in self.properties]
        if len(set(names)) != len(names) or not all(names):
            raise ValueError("properties need distinct names, got %s" % names)
        self._cache: Dict[str, SubjectProfile] = {}

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(p.name for p in self.properties)

    def property(self, name: str) -> Property:
        for prop in self.properties:
            if prop.name == name:
                return prop
        raise KeyError("no property %r (have: %s)"
                       % (name, ", ".join(self.names)))

    def subjects(self, record: Any) -> List[Any]:
        from .response import LawSubject
        return [LawSubject(term, self.domains.get(term.name), self.scale)
                for term in record.terms]

    def profile_subject(self, subject: Any) -> SubjectProfile:
        key = subject.key
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        out = SubjectProfile(subject.name, key, subject.text())
        try:
            analysis = subject.analysis()
        except (ValueError, TypeError, ArithmeticError, KeyError) as error:
            out.errors = {p.name: str(error) for p in self.properties}
        else:
            out.axes = tuple(analysis.axes)
            out.spans = tuple(tuple(map(float, s)) for s in analysis.spans)
            for prop in self.properties:
                try:
                    out.values[prop.name] = prop.extract(analysis)
                except Undecided as why:
                    out.errors[prop.name] = str(why)
        self._cache[key] = out
        return out

    def profile(self, record: Any, label: str = "") -> PropertyProfile:
        return PropertyProfile(
            label, {s.name: self.profile_subject(s)
                    for s in self.subjects(record)}, self.names)


UNCHANGED, CHANGED, ADDED, REMOVED = "unchanged", "changed", "added", "removed"


@dataclass
class SubjectComparison:
    subject: str
    status: str
    before: str = ""
    after: str = ""
    changes: Dict[str, Change] = field(default_factory=dict)

    @property
    def changed_properties(self) -> List[str]:
        return [name for name, change in self.changes.items()
                if change.changed]

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject,
                "status": self.status, "before": self.before,
                "after": self.after,
                "changes": {k: v.to_dict() for k, v in self.changes.items()}}


@dataclass
class PropertyComparison:
    reference: str
    variant: str
    subjects: Dict[str, SubjectComparison] = field(default_factory=dict)
    properties: Tuple[str, ...] = ()

    @property
    def touched(self) -> List[SubjectComparison]:
        return [s for s in self.subjects.values() if s.status != UNCHANGED]

    def changes(self) -> Iterable[Tuple[str, str, Change]]:
        for subject in self.touched:
            for name, change in subject.changes.items():
                yield subject.subject, name, change

    def changed(self) -> List[Tuple[str, str]]:
        return [(subject, name) for subject, name, change in self.changes()
                if change.changed]

    def qualitative(self) -> List[Tuple[str, str, str, str]]:
        order = {name: i for i, name in enumerate(self.properties)}
        out = [(subject, name) + tuple(change.classes or (change.before,
                                                           change.after))
               for subject, name, change in self.changes() if change.shifted]
        return sorted(out, key=lambda row: (order.get(row[1], 99), row[0]))

    def by_property(self) -> Dict[str, List[Tuple[str, Change]]]:
        out: Dict[str, List[Tuple[str, Change]]] = {p: [] for p in
                                                     self.properties}
        for subject, name, change in self.changes():
            out.setdefault(name, []).append((subject, change))
        return out

    def max_distance(self, prop: str) -> Optional[float]:
        changes = [c for _s, c in self.by_property().get(prop, ())]
        if not changes:
            return 0.0
        known = [c.distance for c in changes if c.distance is not None]
        return max(known) if known else None

    def summary(self) -> str:
        return "; ".join("%s%s(%s)" % (name, "!" if change.shifted else "",
                                        subject)
                         for subject, name, change in self.changes()
                         if change.changed)

    def to_dict(self) -> Dict[str, Any]:
        return {"reference": self.reference, "variant": self.variant,
                "properties": list(self.properties),
                "changed": ["%s(%s)" % (n, s) for s, n in self.changed()],
                "qualitative": [list(row) for row in self.qualitative()],
                "max_distance": {p: self.max_distance(p)
                                 for p in self.properties},
                "subjects": [s.to_dict() for s in self.touched],
                "unchanged": [s.subject for s in self.subjects.values()
                              if s.status == UNCHANGED]}


def compare(reference: PropertyProfile, variant: PropertyProfile,
            properties: Sequence[Property]) -> PropertyComparison:
    out = PropertyComparison(reference.label, variant.label,
                             properties=tuple(p.name for p in properties))
    for name in list(reference.subjects) + [n for n in variant.subjects
                                            if n not in reference.subjects]:
        before = reference.subjects.get(name)
        after = variant.subjects.get(name)
        if after is None:
            out.subjects[name] = SubjectComparison(name, REMOVED,
                                                   before.text, "")
            continue
        if before is None:
            out.subjects[name] = SubjectComparison(name, ADDED, "",
                                                   after.text)
            continue
        if before.key == after.key:
            out.subjects[name] = SubjectComparison(name, UNCHANGED,
                                                   before.text, after.text)
            continue
        subject = SubjectComparison(name, CHANGED, before.text, after.text)
        for prop in properties:
            problem = (before.errors.get(prop.name)
                       or after.errors.get(prop.name))
            if problem or prop.name not in before.values \
                    or prop.name not in after.values:
                subject.changes[prop.name] = Change(
                    None, None, "undecided: %s" % (problem or "not profiled"))
                continue
            subject.changes[prop.name] = prop.compare(
                before.values[prop.name], after.values[prop.name])
        out.subjects[name] = subject
    return out


__all__ = ["ADDED", "CHANGED", "Change", "PROFILE_FORMAT", "Profiler",
           "Property", "PropertyComparison", "PropertyProfile", "REMOVED",
           "SubjectComparison", "SubjectProfile", "UNCHANGED", "Undecided",
           "compare", "signature"]
