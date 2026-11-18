from __future__ import annotations

from enum import IntEnum


class Verdict(IntEnum):
    SAFE = 0
    VIOLATES = 1
    DOOMED = 2
    SINGULAR = 3
    INFEASIBLE = 4


CHARGED = (Verdict.VIOLATES, Verdict.SINGULAR)


__all__ = ["CHARGED", "Verdict"]
