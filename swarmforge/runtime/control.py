from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..solve.config import FamilyConfig
from ..solve.laws import Controller, Members, View
from ..solve.record import Record
from .scene import Scene


class _Bound(Controller):
    def __init__(self, record: Record, config: FamilyConfig):
        super().__init__(record, config)
        self.scene: Optional[Scene] = None
        self.agent = None

    def view(self, agent, scene: Scene) -> View:
        self.scene, self.agent = scene, agent
        here = np.array([[scene.position(agent)]], dtype=float)
        targets = scene.resolve("target", agent)
        target = (np.array([[scene.position(targets[0])]], dtype=float)
                  if targets else here.copy())
        return View(here, np.array([[scene.velocity(agent)]], dtype=float),
                    np.zeros((1, 0, 2)), np.zeros((1, 0)), target)

    def _members(self, source: str, i: int, view: View, role: str,
                 cache: Dict[str, Members]) -> Members:
        if source in cache:
            return cache[source]
        if source == "target":
            out = Members(view.targets[:, i:i + 1], np.zeros((1, 1, 2)),
                          np.zeros((1, 1)))
        else:
            objects = self.scene.resolve(source, self.agent)
            out = Members(
                np.array([[self.scene.position(o) for o in objects]],
                         dtype=float).reshape(1, len(objects), 2),
                np.array([[self.scene.velocity(o) for o in objects]],
                         dtype=float).reshape(1, len(objects), 2),
                np.array([[self.scene.size(o) for o in objects]],
                         dtype=float).reshape(1, len(objects)))
        cache[source] = out
        return out


class SceneLaw:
    def __init__(self, record: Record, config: FamilyConfig):
        self.record = record
        self._law = _Bound(record, config)

    def params(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.record.params.items()
                if v is not None}

    def command(self, agent, scene: Scene) -> Tuple[float, float]:
        out = self._law.command(0, self._law.view(agent, scene),
                                scene.role(agent))
        return float(out[0, 0]), float(out[0, 1])


__all__ = ["SceneLaw"]
