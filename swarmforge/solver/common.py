from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from swarmforge.solve import read_config
from swarmforge.solve.models import build_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "runs" / "solver"

FAMILIES = ("boids", "gradient", "vasarhelyi", "adaptive-swarm")


def config_path(family: str) -> Path:
    return ROOT / "configs_new" / ("%s.yaml" % family)


def horizon_one() -> List[Tuple[str, str]]:
    out = []
    for family in FAMILIES:
        config = read_config(config_path(family))
        for name, spec in config.models.items():
            if spec.horizon == 1:
                out.append((family, name))
    return out


def model_of(family: str, name: str):
    return build_model(name, read_config(config_path(family)))


def parse_cases(text: str) -> List[Tuple[str, str]]:
    cases = horizon_one()
    if text == "all":
        return cases
    wanted = [tuple(item.split("/")) for item in text.split(",")]
    missing = [w for w in wanted if w not in cases]
    if missing:
        raise SystemExit("not a horizon-1 model: %s" % missing)
    return wanted
