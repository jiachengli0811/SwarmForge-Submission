from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

KINDS = {"obstacle_collision": "obstacle", "agent_collision": "agent",
         "timeout": "timeout"}


@dataclass(frozen=True)
class Calibrate:
    missions: int
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {"missions": self.missions, "seed": self.seed}


def read_calibrate(raw: Any) -> Optional[Calibrate]:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("harden.calibrate must be a mapping {missions, seed}")
    missing = [k for k in ("missions", "seed") if raw.get(k) is None]
    if missing:
        raise ValueError("harden.calibrate needs %s -- none has a default"
                         % ", ".join(missing))
    unknown = sorted(set(raw) - {"missions", "seed"})
    if unknown:
        raise ValueError("harden.calibrate has no option %s"
                         % ", ".join(unknown))
    out = Calibrate(int(raw["missions"]), int(raw["seed"]))
    if out.missions < 1:
        raise ValueError("harden.calibrate.missions must be at least one")
    return out


def rates(flights: List[Dict[str, Any]]) -> Dict[str, float]:
    ok = [f for f in flights if not f["error"]]
    if not ok:
        raise ValueError("no calibration flight completed")
    n = float(len(ok))
    return {"obstacle_collision": sum(f["obstacle"] is not None for f in ok) / n,
            "agent_collision": sum(f["agent"] is not None for f in ok) / n,
            "timeout": sum(f["agent"] is None and f["obstacle"] is None
                           and f["arrived"] is None for f in ok) / n}


def weights(configured: Mapping[str, float], scores: Mapping[str, float],
            rate: Mapping[str, float], missions: int, cells: int
            ) -> Dict[str, float]:
    out = dict(configured)
    for name in configured:
        if name in KINDS:
            out[name] = float(configured[name]) * (
                (rate[name] + 1.0 / missions)
                / (float(scores[name]) + 1.0 / cells))
    return out


def calibrate(config, record_path: Path, harden, scores: Mapping[str, float],
              out_dir: Optional[Path] = None
              ) -> Tuple[Any, Dict[str, Any]]:
    import dataclasses

    from .trend import fly_through
    from .weakness import Objective

    plan = harden.calibrate
    if not harden.scoring.region:
        raise ValueError("harden.calibrate is region scoring's (a score's "
                         "resolution is one cell): this run scores by %s"
                         % harden.scoring.name)
    folder = (Path(out_dir) / "calibration" if out_dir is not None
              else Path(tempfile.mkdtemp(prefix="calibration-")))
    flown = fly_through(Path(config.path), {"original": Path(record_path)},
                        missions=plan.missions, seed=plan.seed,
                        out_dir=folder, workers=harden.workers)["original"]
    rate = rates(flown)
    configured = dict(harden.objective.weights)
    weighed = weights(configured, scores, rate, plan.missions,
                      harden.scoring.cells)
    info = {"missions": plan.missions, "seed": plan.seed,
            "swarm_size": int(config.swarm_size),
            "flown": sum(1 for f in flown if not f["error"]),
            "rates": rate, "scores": {k: float(scores[k]) for k in configured},
            "configured": configured, "weights": weighed}
    return dataclasses.replace(harden, objective=Objective(weighed)), info


__all__ = ["Calibrate", "KINDS", "calibrate", "rates", "read_calibrate",
           "weights"]
