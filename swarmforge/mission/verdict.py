from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

from ..solve.config import FamilyConfig
from .snapshot import Snapshot


class Verdict(Enum):
    CRASH = "crash"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"


BENIGN = (Verdict.SUCCESS,)
NOT_EVIDENCE = (Verdict.ERROR,)


@dataclass(frozen=True)
class Oracle:
    agent_radius: float
    agent_boundary: float
    obstacle_gap: float
    arrive_fraction: float
    goal_radius: float
    settle_ticks: int = 1

    @classmethod
    def for_config(cls, config: FamilyConfig) -> "Oracle":
        mission = config.mission
        if mission is None:
            raise ValueError("%s declares no mission block" % config.path)
        physics = config.physics
        step = float(physics.plant.max_step)
        settle = (int(math.ceil(mission.settle_factor * physics.agent_boundary
                                / step)) if step > 0.0 else 8)
        return cls(agent_radius=float(physics.agent_radius),
                   agent_boundary=float(physics.agent_boundary),
                   obstacle_gap=float(physics.obstacle_min_gap),
                   arrive_fraction=float(mission.arrive_fraction),
                   goal_radius=float(mission.goal_radius),
                   settle_ticks=max(1, settle))

    def to_dict(self) -> Dict:
        return {"agent_radius": self.agent_radius,
                "agent_boundary": self.agent_boundary,
                "obstacle_gap": self.obstacle_gap,
                "arrive_fraction": self.arrive_fraction,
                "goal_radius": self.goal_radius,
                "settle_ticks": self.settle_ticks}


@dataclass
class Outcome:
    verdict: Verdict
    tick: Optional[int] = None
    crash_kind: str = ""
    detail: Dict = _field(default_factory=dict)

    @property
    def errored(self) -> bool:
        return self.verdict in NOT_EVIDENCE

    @property
    def failed(self) -> bool:
        return self.verdict not in BENIGN and not self.errored

    def to_dict(self) -> Dict:
        return {"verdict": self.verdict.value, "tick": self.tick,
                "crash_kind": self.crash_kind, "detail": dict(self.detail)}


def _crash_kind(snapshot: Snapshot, oracle: Oracle) -> str:
    positions = snapshot.positions
    hit_agent = any(
        math.dist(positions[i], positions[j]) < oracle.agent_boundary
        for i in range(len(positions)) for j in range(i + 1, len(positions)))
    hit_obstacle = any(
        math.dist(position, centre) - radius - oracle.agent_radius
        < oracle.obstacle_gap
        for position in positions
        for centre, radius in snapshot.obstacles)
    if hit_agent and hit_obstacle:
        return "both"
    return "agent" if hit_agent else ("obstacle" if hit_obstacle else "")


def contact_kind(snapshot: Snapshot, oracle: Oracle) -> str:
    if snapshot.tick < oracle.settle_ticks:
        return ""
    return _crash_kind(snapshot, oracle)


def _arrived(snapshot: Snapshot, goal: Tuple[float, float],
             radius: float) -> int:
    return sum(1 for p in snapshot.positions if math.dist(p, goal) <= radius)


def _need_arrived(snapshot: Snapshot, oracle: Oracle) -> int:
    return max(1, int(math.ceil(oracle.arrive_fraction
                                * len(snapshot.positions))))


def home(snapshot: Snapshot, goal: Tuple[float, float],
         oracle: Oracle) -> bool:
    return (_arrived(snapshot, goal, oracle.goal_radius)
            >= _need_arrived(snapshot, oracle))


def decisive(snapshot: Snapshot, goal: Tuple[float, float],
             oracle: Oracle) -> bool:
    return bool(contact_kind(snapshot, oracle)) or home(snapshot, goal,
                                                        oracle)


def _centroid_range(snapshot: Snapshot, goal: Tuple[float, float]) -> float:
    if not snapshot.positions:
        return 0.0
    cx = sum(p[0] for p in snapshot.positions) / len(snapshot.positions)
    cy = sum(p[1] for p in snapshot.positions) / len(snapshot.positions)
    return math.dist((cx, cy), goal)


def judge(snapshots: Sequence[Snapshot], *, goal: Tuple[float, float],
          oracle: Oracle, completed: bool = True) -> Outcome:
    if not snapshots:
        return Outcome(Verdict.ERROR, None, "", {"note": "no snapshots"})
    n_agents = len(snapshots[0].positions)
    need_arrived = _need_arrived(snapshots[0], oracle)
    for snapshot in snapshots:
        kind = contact_kind(snapshot, oracle)
        if kind:
            return Outcome(Verdict.CRASH, snapshot.tick, kind,
                           {"agents": n_agents,
                            "settle_ticks": oracle.settle_ticks})
        if _arrived(snapshot, goal, oracle.goal_radius) >= need_arrived:
            return Outcome(Verdict.SUCCESS, snapshot.tick, "",
                           {"arrived": need_arrived})
    if not completed:
        return Outcome(Verdict.ERROR, snapshots[-1].tick, "",
                       {"note": "flight did not complete"})
    return Outcome(Verdict.TIMEOUT, snapshots[-1].tick, "",
                   {"range_left": round(_centroid_range(snapshots[-1], goal),
                                        6)})


__all__ = ["BENIGN", "NOT_EVIDENCE", "Oracle", "Outcome", "Verdict",
           "contact_kind", "decisive", "home", "judge"]
