from .core import (ADDED, CHANGED, REMOVED, UNCHANGED, Change, Profiler,
                   Property, PropertyComparison, PropertyProfile,
                   SubjectComparison, SubjectProfile, Undecided, compare,
                   signature)
from .facets import (FACETS, SEVERITY, Facet, Magnitude, Sign, Switching,
                     Variables, family)
from .response import (CommandScale, LawAnalysis, LawResponse, LawSubject,
                       Switch)

PROPERTIES = FACETS

REGISTRY = {prop.name: prop for prop in FACETS}

__all__ = ["ADDED", "CHANGED", "Change", "CommandScale", "FACETS", "Facet",
           "LawAnalysis", "LawResponse", "LawSubject", "Magnitude",
           "PROPERTIES",
           "Profiler", "Property", "PropertyComparison", "PropertyProfile",
           "REGISTRY", "REMOVED", "SEVERITY", "Sign", "SubjectComparison",
           "SubjectProfile", "Switch", "Switching",
           "UNCHANGED", "Undecided", "Variables", "compare",
           "family", "signature"]
