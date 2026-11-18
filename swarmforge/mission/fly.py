from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from ..runtime.adapter import real_env_adapter
from ..solve.config import FamilyConfig
from ..solve.record import Record
from .snapshot import Snapshot


@dataclass
class Flight:
    snapshots: Tuple[Snapshot, ...] = ()
    ticks_run: int = 0
    completed: bool = True
    error: Optional[str] = None
    decided: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


def _snapshot(adapter, tick: int, goal: Tuple[float, float]) -> Snapshot:
    agents = adapter.agents()
    bodies = [o for o in adapter.objects() if o.kind == "obstacle"]
    aim, route = adapter.aim()
    return Snapshot(
        tick=tick,
        positions=tuple(a.position for a in agents),
        velocities=tuple(a.velocity for a in agents),
        obstacles=tuple((o.position, float(o.meta.get("size", 0.0)))
                        for o in bodies),
        goal=goal, movers=adapter.movers(), aim=aim, route=route)


def fly(document: Dict[str, Any], config: FamilyConfig, record: Record, *,
        spawn_seed: int = 0, ticks: Optional[int] = None, every: int = 4,
        decides: Optional[Callable[[Snapshot], bool]] = None,
        probe: Optional[Any] = None) -> Flight:
    budget = int(ticks if ticks is not None else document.get("ticks", 600))
    every = max(1, int(every))
    goal = tuple(float(value) for value in document["goal"])
    if len(goal) != 2:
        raise ValueError("mission goal must contain two coordinates")
    adapter = real_env_adapter(config, record, document)
    if probe is not None:
        probe.attach(adapter)
    try:
        adapter.reset(seed=int(spawn_seed))
    except Exception as error:
        adapter.close()
        return Flight((), 0, False, "reset failed: %s" % error)
    if probe is not None:
        probe.bound(adapter)

    snapshots = []
    tick = 0
    try:
        snapshots.append(_snapshot(adapter, 0, goal))
        for tick in range(1, budget + 1):
            if probe is not None:
                probe.before(tick)
            adapter.step()
            keep = tick % every == 0 or tick == budget
            if keep or decides is not None:
                snapshot = _snapshot(adapter, tick, goal)
                if decides is not None and decides(snapshot):
                    snapshots.append(snapshot)
                    return Flight(tuple(snapshots), tick, True, None,
                                  decided=True)
                if keep:
                    snapshots.append(snapshot)
    except Exception as error:
        return Flight(tuple(snapshots), tick, False,
                      "step failed: %s: %s" % (type(error).__name__, error))
    finally:
        adapter.close()
    return Flight(tuple(snapshots), budget, True, None)


__all__ = ["Flight", "fly"]
