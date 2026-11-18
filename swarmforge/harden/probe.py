from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..solve.bank import Bank
from ..solve.config import FamilyConfig
from ..solve.models import build_model
from ..solve.record import Record
from ..solve.state import State
from ..solve.verdict import CHARGED, Verdict
from .candidate import Candidate

_CHARGED = np.array([int(v) for v in CHARGED], dtype=np.int8)


@dataclass
class ModelProbe:
    model: str
    indices: np.ndarray
    was: np.ndarray
    weights: np.ndarray
    states: State = field(repr=False)
    bank_size: int = 0


@dataclass
class ProbeSet:
    parent: str
    models: Dict[str, ModelProbe]
    weights: Mapping[str, float]

    @property
    def size(self) -> int:
        return sum(int(p.indices.size) for p in self.models.values())

    def to_dict(self) -> Dict[str, Any]:
        return {"parent": self.parent,
                "states": {m: {"charged": int(p.was.sum()),
                               "safe": int((~p.was).sum()),
                               "bank": p.bank_size}
                           for m, p in self.models.items()}}


@dataclass(frozen=True)
class ProbeReading:
    repaired: Mapping[str, float]
    broken: Mapping[str, float]
    gain: float

    def to_dict(self) -> Dict[str, Any]:
        return {"gain": round(self.gain, 8),
                "repaired": {m: round(v, 1) for m, v in self.repaired.items()},
                "broken": {m: round(v, 1) for m, v in self.broken.items()}}


def draw(parent: Candidate, banks: Mapping[str, Bank],
         weights: Mapping[str, float], charged: int, safe: int,
         rng: np.random.Generator,
         state_weights: Optional[Mapping[str, np.ndarray]] = None
         ) -> ProbeSet:
    if parent.weakness is None:
        raise ValueError("%s has not been measured: it has no weakness "
                         "states to probe" % parent.id)
    models: Dict[str, ModelProbe] = {}
    for name in weights:
        reading = parent.weakness.models.get(name)
        if reading is None or not reading.codes:
            raise ValueError("%s has no states of %s" % (parent.id, name))
        verdicts = reading.verdicts()
        failing = np.flatnonzero(np.isin(verdicts, _CHARGED))
        holding = np.flatnonzero(verdicts == int(Verdict.SAFE))
        picked_failing = rng.choice(failing, min(charged, failing.size),
                                    replace=False)
        picked_holding = rng.choice(holding, min(safe, holding.size),
                                    replace=False)
        indices = np.concatenate([np.sort(picked_failing),
                                  np.sort(picked_holding)])
        was = np.zeros(indices.size, dtype=bool)
        was[:picked_failing.size] = True
        stands = np.concatenate([
            np.full(picked_failing.size,
                    failing.size / max(picked_failing.size, 1)),
            np.full(picked_holding.size,
                    holding.size / max(picked_holding.size, 1))])
        if state_weights and name in state_weights:
            stands = stands * (state_weights[name][indices] * verdicts.size)
        models[name] = ModelProbe(name, indices, was, stands,
                                  banks[name].state.take(indices),
                                  int(verdicts.size))
    return ProbeSet(parent.id, models, dict(weights))


def read(probes: ProbeSet, record: Record,
         config: FamilyConfig) -> ProbeReading:
    repaired: Dict[str, float] = {}
    broken: Dict[str, float] = {}
    gain = 0.0
    for name, probe in probes.models.items():
        outcome = build_model(name, config, record).evaluate(probe.states)
        now = np.isin(outcome.verdict, _CHARGED)
        repaired[name] = float(np.sum(probe.weights[probe.was & ~now]))
        broken[name] = float(np.sum(probe.weights[~probe.was & now]))
        gain += (probes.weights[name] * (repaired[name] - broken[name])
                 / max(probe.bank_size, 1))
    return ProbeReading(repaired, broken, gain)


__all__ = ["ModelProbe", "ProbeReading", "ProbeSet", "draw", "read"]
