from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import yaml

DEFAULT_DRONES = 4
DEFAULT_HORIZON = 1

INTEGRATIONS = ("velocity", "displacement")
UPDATE_ORDERS = ("sequential", "synchronous")
OBSTACLE_SHAPES = ("disc", "square")
PEER_SHAPES = ("point", "square")
CLEARANCES = ("self", "other", "surface")
TARGET_KINDS = ("shared_goal", "leader_formation")
MOTIONS = ("constant_velocity",)
FOCUSES = ("contested", "uniform")


def body_frame(offset, radii, axes=None):
    offset = np.asarray(offset, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if axes is None or not np.any(axes):
        return offset, np.stack([radii, radii], axis=-1), None
    axes = np.asarray(axes, dtype=float)
    length = np.sqrt(np.sum(axes * axes, axis=-1))
    wall = length > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        along = np.where(wall[..., None], axes / length[..., None],
                         np.array([1.0, 0.0]))
    across = np.stack([-along[..., 1], along[..., 0]], axis=-1)
    local = np.stack([np.sum(offset * along, axis=-1),
                      np.sum(offset * across, axis=-1)], axis=-1)
    half = np.stack([np.where(wall, length, radii),
                     np.broadcast_to(radii, length.shape)], axis=-1)
    return local, half, (along, across)


def surface_distance(offset, radii, axes=None):
    local, half, _ = body_frame(offset, radii, axes)
    outside = np.maximum(np.abs(local) - half, 0.0)
    return np.sqrt(np.sum(outside * outside, axis=-1))


@dataclass(frozen=True)
class Span:
    lo: float
    hi: float

    @classmethod
    def read(cls, raw: Any, label: str, *, nonnegative: bool = False) -> "Span":
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError("%s must be a [lo, hi] pair, got %r" % (label, raw))
        lo, hi = float(raw[0]), float(raw[1])
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            raise ValueError("%s must satisfy lo <= hi, got %r" % (label, raw))
        if nonnegative and lo < 0.0:
            raise ValueError("%s must be nonnegative, got %r" % (label, raw))
        return cls(lo, hi)


@dataclass(frozen=True)
class Box:
    x: Span
    y: Span

    @classmethod
    def read(cls, raw: Any, label: str) -> "Box":
        block = _mapping(raw, label)
        return cls(Span.read(_need(block, "x", label), label + ".x"),
                   Span.read(_need(block, "y", label), label + ".y"))


@dataclass(frozen=True)
class Plant:
    integration: str
    update_order: str
    max_accel: Optional[float]
    max_step: float


@dataclass(frozen=True)
class Geometry:
    obstacles: str
    peers: str
    peer_half_size: float
    clearance: str
    peer_size: float = 0.0


@dataclass(frozen=True)
class Targets:
    kind: str
    formation_spacing: float = 0.0
    formation_robots: int = 0
    waypoint_spacing: float = 0.0
    waypoint_advance_within: float = 0.0


@dataclass(frozen=True)
class Physics:
    dt: float
    agent_radius: float
    agent_min_gap: float
    obstacle_min_gap: float
    plant: Plant
    geometry: Geometry
    targets: Targets
    record_params: Mapping[str, float]

    @property
    def agent_boundary(self) -> float:
        return 2.0 * self.agent_radius + self.agent_min_gap

    def obstacle_boundary(self, radius):
        return self.agent_radius + radius + self.obstacle_min_gap

    @property
    def square_bodies(self) -> bool:
        return self.geometry.obstacles == "square"

    def body_gap(self, points, centres, radii, axes=None):
        offset = np.asarray(points, dtype=float) - np.asarray(centres,
                                                              dtype=float)
        radii = np.asarray(radii, dtype=float)
        if self.square_bodies:
            surface = surface_distance(offset, radii, axes)
        else:
            surface = np.sqrt(np.sum(offset * offset, axis=-1)) - radii
        return surface - self.agent_radius - self.obstacle_min_gap

    def contact_reach(self, radius):
        corner = math.sqrt(2.0) if self.square_bodies else 1.0
        return self.agent_radius + corner * radius + self.obstacle_min_gap


@dataclass(frozen=True)
class ObstacleRanges:
    radius: Span
    min_spacing: float = 0.0
    speed: Optional[Span] = None
    motion: str = "constant_velocity"
    wall_share: float = 0.0
    wall_extent: Optional[Span] = None
    wall_length: Optional[Span] = None
    wall_thickness: Optional[Span] = None
    standoff: float = 0.0


@dataclass(frozen=True)
class Window:
    drones: float
    static_bodies: Optional[float]
    dynamic_bodies: Optional[float]
    target: float
    walls: Optional[float] = None


@dataclass(frozen=True)
class StateRanges:
    drone_region: Box
    drone_speed: Optional[Span]
    static: Optional[ObstacleRanges]
    dynamic: Optional[ObstacleRanges]
    window: Window
    waypoint_distance: Optional[Span] = None
    waypoint_bearing_deg: Optional[Span] = None
    route_heading_deg: Optional[Span] = None
    route_final_share: float = 0.0


@dataclass(frozen=True)
class Sampling:
    samples: int
    seed: int
    chunk: int
    max_draws: int
    focus: str = "contested"
    bank_dir: Optional[Path] = None


@dataclass(frozen=True)
class ModelSpec:
    drones: int = DEFAULT_DRONES
    static_obstacles: int = 0
    dynamic_obstacles: int = 0
    horizon: int = DEFAULT_HORIZON
    pace: Optional[float] = None
    sense_range: Optional[float] = None
    max_angle_deg: Optional[float] = None
    min_speed: Optional[float] = None
    target_range: Optional[float] = None
    path_width: Optional[float] = None
    margin: Optional[float] = None


@dataclass(frozen=True)
class Attackers:
    count: int
    max_speed: float
    size: Optional[float]
    min_gap: float


@dataclass(frozen=True)
class MovingGoal:
    speed: float
    lead: Optional[float] = None


@dataclass(frozen=True)
class Mission:
    ticks: int
    goal_radius: float
    arrive_fraction: float
    obstacle_radius: float
    rect_prob: float
    rect_radius: Optional[float]
    spawn_separation: float
    attackers: Attackers
    settle_factor: float = 3.0
    moving_goal: Optional[MovingGoal] = None


@dataclass(frozen=True)
class TestRangeSpec:
    bands: int
    rho_easy: float
    rho_hard: float
    curve: Tuple[Tuple[float, float], ...]
    spacing_easy: float
    spacing_hard: float
    weights: Tuple[int, ...] = ()


@dataclass(frozen=True)
class FamilyConfig:
    path: Path
    name: str
    record_path: Path
    units: Mapping[str, str]
    world: Box
    swarm_size: int
    physics: Physics
    state: StateRanges
    sampling: Sampling
    models: Mapping[str, ModelSpec]
    mission: Optional[Mission] = None
    test_range: Optional[TestRangeSpec] = None

    def model(self, name: str) -> ModelSpec:
        if name not in self.models:
            raise ValueError("%s declares no model %r under `models:` (it "
                             "declares: %s)" % (self.path, name,
                                                ", ".join(self.models) or "none"))
        return self.models[name]


def _mapping(raw: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("%s must be a mapping" % label)
    return raw


def _need(block: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in block or block[key] is None:
        raise ValueError("config is missing %s.%s" % (label, key))
    return block[key]


def _number(block: Mapping[str, Any], key: str, label: str, *,
            positive: bool = False, nonnegative: bool = False) -> float:
    value = float(_need(block, key, label))
    if not math.isfinite(value):
        raise ValueError("%s.%s must be finite" % (label, key))
    if positive and value <= 0.0:
        raise ValueError("%s.%s must be positive" % (label, key))
    if nonnegative and value < 0.0:
        raise ValueError("%s.%s must be nonnegative" % (label, key))
    return value


def _choice(block: Mapping[str, Any], key: str, label: str, allowed) -> str:
    value = str(_need(block, key, label))
    if value not in allowed:
        raise ValueError("%s.%s is %r (have: %s)"
                         % (label, key, value, ", ".join(allowed)))
    return value


def _count(raw: Any, label: str, *, minimum: int = 0) -> int:
    value = float(raw)
    if value != int(value) or int(value) < minimum:
        raise ValueError("%s must be a whole number >= %d, got %r"
                         % (label, minimum, raw))
    return int(value)


def _obstacles(raw: Any, label: str, dynamic: bool) -> ObstacleRanges:
    block = _mapping(raw, label)
    if "region" in block:
        raise ValueError("%s.region is no longer read: a body is drawn inside "
                         "state.window of a contact, not over a region"
                         % label)
    speed = None
    motion = "constant_velocity"
    if dynamic:
        speed = Span.read(_need(block, "speed", label), label + ".speed",
                          nonnegative=True)
        motion = _choice(block, "motion", label, MOTIONS)
    radius = Span.read(_need(block, "radius", label), label + ".radius",
                       nonnegative=True)
    if radius.lo <= 0.0:
        raise ValueError("%s.radius must be positive" % label)
    wall_share = float(block.get("wall_share") or 0.0)
    if not 0.0 <= wall_share <= 1.0:
        raise ValueError("%s.wall_share must be a share in [0, 1]" % label)
    if wall_share and dynamic:
        raise ValueError("%s.wall_share: a moving body is never a wall"
                         % label)
    shape = {}
    if wall_share:
        for key in ("wall_extent", "wall_length", "wall_thickness"):
            span = Span.read(_need(block, key, label), "%s.%s" % (label, key),
                             nonnegative=True)
            if span.lo <= 0.0:
                raise ValueError("%s.%s must be positive" % (label, key))
            shape[key] = span
    standoff = float(block.get("standoff") or 0.0)
    if standoff < 0.0:
        raise ValueError("%s.standoff must be nonnegative" % label)
    if standoff and not dynamic:
        raise ValueError("%s.standoff: a static body keeps no stand-off"
                         % label)
    return ObstacleRanges(
        radius=radius,
        min_spacing=float(block.get("min_spacing") or 0.0),
        speed=speed, motion=motion, wall_share=wall_share,
        standoff=standoff, **shape)


def read_config(path) -> FamilyConfig:
    from ..paths import relocate_path

    path = relocate_path(path)
    document = yaml.safe_load(path.read_text())
    root = _mapping(document, str(path))

    name = str(_need(root, "name", "config"))
    record_path = relocate_path(path.parent / str(_need(root, "record", "config"))).resolve()
    if not record_path.is_file():
        raise ValueError("%s: record %s does not exist" % (path, record_path))
    units = MappingProxyType({str(k): str(v) for k, v in
                              _mapping(root.get("units") or {}, "units").items()})
    world = Box.read(_need(root, "world", "config"), "world")
    swarm_size = _count(_need(_mapping(_need(root, "swarm", "config"), "swarm"),
                              "size", "swarm"), "swarm.size", minimum=1)

    phys = _mapping(_need(root, "physics", "config"), "physics")
    plant_raw = _mapping(_need(phys, "plant", "physics"), "physics.plant")
    max_accel = plant_raw.get("max_accel")
    plant = Plant(
        integration=_choice(plant_raw, "integration", "physics.plant",
                            INTEGRATIONS),
        update_order=_choice(plant_raw, "update_order", "physics.plant",
                             UPDATE_ORDERS),
        max_accel=None if max_accel is None else float(max_accel),
        max_step=_number(plant_raw, "max_step", "physics.plant", positive=True))
    if plant.max_accel is not None and plant.max_accel <= 0.0:
        raise ValueError("physics.plant.max_accel must be positive or null")
    if plant.max_accel is not None and plant.integration != "velocity":
        raise ValueError("physics.plant.max_accel limits a velocity change; a "
                         "displacement plant carries no velocity to limit")

    geo_raw = _mapping(_need(phys, "geometry", "physics"), "physics.geometry")
    peers = _choice(geo_raw, "peers", "physics.geometry", PEER_SHAPES)
    clearance = _choice(geo_raw, "clearance", "physics.geometry", CLEARANCES)
    geometry = Geometry(
        obstacles=_choice(geo_raw, "obstacles", "physics.geometry",
                          OBSTACLE_SHAPES),
        peers=peers,
        peer_half_size=(_number(geo_raw, "peer_half_size", "physics.geometry",
                                positive=True) if peers == "square" else 0.0),
        clearance=clearance,
        peer_size=(_number(geo_raw, "peer_size", "physics.geometry",
                           nonnegative=True) if clearance == "other" else 0.0))

    tgt_raw = _mapping(_need(phys, "targets", "physics"), "physics.targets")
    kind = _choice(tgt_raw, "kind", "physics.targets", TARGET_KINDS)
    if kind == "leader_formation":
        targets = Targets(
            kind=kind,
            formation_spacing=_number(tgt_raw, "formation_spacing",
                                      "physics.targets", positive=True),
            formation_robots=_count(_need(tgt_raw, "formation_robots",
                                          "physics.targets"),
                                    "physics.targets.formation_robots",
                                    minimum=1),
            waypoint_spacing=_number(tgt_raw, "waypoint_spacing",
                                     "physics.targets", nonnegative=True),
            waypoint_advance_within=_number(
                tgt_raw, "waypoint_advance_within", "physics.targets",
                nonnegative=True))
    else:
        targets = Targets(kind=kind)

    physics = Physics(
        dt=_number(phys, "dt", "physics", positive=True),
        agent_radius=_number(phys, "agent_radius", "physics", nonnegative=True),
        agent_min_gap=_number(phys, "agent_min_gap", "physics",
                              nonnegative=True),
        obstacle_min_gap=_number(phys, "obstacle_min_gap", "physics",
                                 nonnegative=True),
        plant=plant, geometry=geometry, targets=targets,
        record_params=MappingProxyType({
            str(k): float(v) for k, v in
            _mapping(phys.get("record_params") or {},
                     "physics.record_params").items()}))

    st = _mapping(_need(root, "state", "config"), "state")
    drones = _mapping(_need(st, "drones", "state"), "state.drones")
    drone_speed = None
    if plant.integration == "velocity":
        drone_speed = Span.read(_need(drones, "speed", "state.drones"),
                                "state.drones.speed", nonnegative=True)
    elif "speed" in drones:
        drone_speed = Span.read(drones["speed"], "state.drones.speed",
                                nonnegative=True)
        top = plant.max_step / physics.dt
        if drone_speed.hi > top * (1.0 + 1e-9):
            raise ValueError("state.drones.speed: a displacement plant's last "
                             "step is at most physics.plant.max_step / "
                             "physics.dt = %g, got %g" % (top, drone_speed.hi))
    if "goal" in st:
        raise ValueError("state.goal is no longer read: the goal is drawn "
                         "inside state.window.target of a contact")
    has_static = bool(st.get("static_obstacles"))
    has_dynamic = bool(st.get("dynamic_obstacles"))
    win = _mapping(_need(st, "window", "state"), "state.window")
    window = Window(
        drones=_number(win, "drones", "state.window", nonnegative=True),
        static_bodies=(_number(win, "static_bodies", "state.window",
                               nonnegative=True) if has_static else None),
        dynamic_bodies=(_number(win, "dynamic_bodies", "state.window",
                                nonnegative=True) if has_dynamic else None),
        target=_number(win, "target", "state.window", nonnegative=True),
        walls=(_number(win, "walls", "state.window", nonnegative=True)
               if has_static and (_mapping(st["static_obstacles"],
                                           "state.static_obstacles")
                                  .get("wall_share") or 0.0) else None))
    state = StateRanges(
        drone_region=Box.read(_need(drones, "region", "state.drones"),
                              "state.drones.region"),
        drone_speed=drone_speed,
        static=(_obstacles(st["static_obstacles"], "state.static_obstacles",
                           False) if has_static else None),
        dynamic=(_obstacles(st["dynamic_obstacles"], "state.dynamic_obstacles",
                            True) if has_dynamic else None),
        window=window)
    if kind == "leader_formation":
        waypoint = _mapping(_need(st, "waypoint", "state"), "state.waypoint")
        route = _mapping(_need(st, "route", "state"), "state.route")
        state = _replace(
            state,
            waypoint_distance=Span.read(
                _need(waypoint, "distance", "state.waypoint"),
                "state.waypoint.distance", nonnegative=True),
            waypoint_bearing_deg=Span.read(
                _need(waypoint, "bearing_deg", "state.waypoint"),
                "state.waypoint.bearing_deg"),
            route_heading_deg=Span.read(
                _need(route, "heading_deg", "state.route"),
                "state.route.heading_deg"),
            route_final_share=_final_share(route))

    smp = _mapping(_need(root, "sampling", "config"), "sampling")
    sampling = Sampling(
        samples=_count(_need(smp, "samples", "sampling"), "sampling.samples",
                       minimum=1),
        seed=_count(_need(smp, "seed", "sampling"), "sampling.seed"),
        chunk=_count(_need(smp, "chunk", "sampling"), "sampling.chunk",
                     minimum=1),
        max_draws=_count(_need(smp, "max_draws", "sampling"),
                         "sampling.max_draws", minimum=1),
        focus=(_choice(smp, "focus", "sampling", FOCUSES)
               if smp.get("focus") is not None else "contested"),
        bank_dir=relocate_path(path.parent / str(smp.get("bank_dir") or "banks")).resolve())

    models: Dict[str, ModelSpec] = {}
    for model_name, raw in _mapping(_need(root, "models", "config"),
                                    "models").items():
        block = _mapping(raw or {}, "models.%s" % model_name)
        label = "models.%s" % model_name
        spec = ModelSpec(
            drones=_count(block.get("drones", DEFAULT_DRONES),
                          label + ".drones", minimum=1),
            static_obstacles=_count(_need(block, "static_obstacles", label),
                                    label + ".static_obstacles"),
            dynamic_obstacles=_count(_need(block, "dynamic_obstacles", label),
                                     label + ".dynamic_obstacles"),
            horizon=_count(block.get("horizon", DEFAULT_HORIZON),
                           label + ".horizon", minimum=1),
            pace=_optional(block, "pace", label, positive=True),
            sense_range=_optional(block, "sense_range", label, positive=True),
            max_angle_deg=_optional(block, "max_angle_deg", label,
                                    positive=True),
            min_speed=_optional(block, "min_speed", label, nonnegative=True),
            target_range=_optional(block, "target_range", label,
                                   positive=True),
            path_width=_optional(block, "path_width", label, positive=True),
            margin=_optional(block, "margin", label, positive=True))
        _check_spec(spec, label, state, targets)
        models[str(model_name)] = spec

    return FamilyConfig(
        path=path, name=name, record_path=record_path, units=units,
        world=world, swarm_size=swarm_size, physics=physics, state=state,
        sampling=sampling, models=MappingProxyType(models),
        mission=(_mission(root["mission"]) if root.get("mission") else None),
        test_range=(_test_range(root["test_range"])
                    if root.get("test_range") else None))


def _optional(block: Mapping[str, Any], key: str, label: str,
              **checks) -> Optional[float]:
    if block.get(key) is None:
        return None
    return _number(block, key, label, **checks)


def _share(block: Mapping[str, Any], key: str, label: str) -> float:
    value = _number(block, key, label)
    if not 0.0 < value <= 1.0:
        raise ValueError("%s.%s must be a share in (0, 1]" % (label, key))
    return value


def _mission(raw: Any) -> Mission:
    block = _mapping(raw, "mission")
    bodies = _mapping(_need(block, "obstacles", "mission"), "mission.obstacles")
    movers = _mapping(_need(block, "attackers", "mission"),
                      "mission.attackers")
    rect_prob = _number(bodies, "rect_prob", "mission.obstacles")
    if not 0.0 <= rect_prob <= 1.0:
        raise ValueError("mission.obstacles.rect_prob must be in [0, 1]")
    goal = block.get("moving_goal")
    moving = None
    if goal:
        goal = _mapping(goal, "mission.moving_goal")
        moving = MovingGoal(
            speed=_number(goal, "speed", "mission.moving_goal", positive=True),
            lead=_optional(goal, "lead", "mission.moving_goal",
                           nonnegative=True))
    return Mission(
        ticks=_count(_need(block, "ticks", "mission"), "mission.ticks",
                     minimum=1),
        goal_radius=_number(block, "goal_radius", "mission", positive=True),
        arrive_fraction=_share(block, "arrive_fraction", "mission"),
        obstacle_radius=_number(bodies, "radius", "mission.obstacles",
                                positive=True),
        rect_prob=rect_prob,
        rect_radius=_optional(bodies, "rect_radius", "mission.obstacles",
                              positive=True),
        spawn_separation=_number(block, "spawn_separation", "mission",
                                 nonnegative=True),
        attackers=Attackers(
            count=_count(_need(movers, "count", "mission.attackers"),
                         "mission.attackers.count"),
            max_speed=_number(movers, "max_speed", "mission.attackers",
                              nonnegative=True),
            size=_optional(movers, "size", "mission.attackers",
                           positive=True),
            min_gap=_number(movers, "min_gap", "mission.attackers",
                            nonnegative=True)),
        settle_factor=float(block.get("settle_factor", 3.0)),
        moving_goal=moving)


def _test_range(raw: Any) -> TestRangeSpec:
    block = _mapping(raw, "test_range")
    rho = _mapping(_need(block, "rho", "test_range"), "test_range.rho")
    spacing = _mapping(_need(block, "min_spacing", "test_range"),
                       "test_range.min_spacing")
    curve = tuple(sorted((float(x), float(p))
                         for x, p in _need(rho, "curve", "test_range.rho")))
    spec = TestRangeSpec(
        bands=_count(block.get("bands", 5), "test_range.bands", minimum=1),
        rho_easy=float(_need(rho, "easy", "test_range.rho")),
        rho_hard=float(_need(rho, "hard", "test_range.rho")),
        curve=curve,
        spacing_easy=_number(spacing, "easy", "test_range.min_spacing",
                             positive=True),
        spacing_hard=_number(spacing, "hard", "test_range.min_spacing",
                             positive=True),
        weights=tuple(_count(w, "test_range.weights", minimum=0)
                      for w in (block.get("weights") or ())))
    if spec.weights and len(spec.weights) != spec.bands:
        raise ValueError("test_range.weights needs one entry per band (%d)"
                         % spec.bands)
    if spec.weights and not any(spec.weights):
        raise ValueError("test_range.weights needs a band above 0")
    if len(curve) < 2:
        raise ValueError("test_range.rho.curve needs two points")
    if any(p1 > p0 + 1e-12 for (_, p0), (_, p1) in zip(curve, curve[1:])):
        raise ValueError("test_range.rho.curve must not rise with density")
    if not 0.0 <= spec.rho_easy < spec.rho_hard:
        raise ValueError("test_range.rho needs 0 <= easy < hard")
    if not (curve[0][0] <= spec.rho_easy and spec.rho_hard <= curve[-1][0]):
        raise ValueError("test_range.rho.curve must cover easy..hard")
    return spec


def _check_spec(spec: ModelSpec, label: str, state: StateRanges,
                targets: Targets) -> None:
    if spec.static_obstacles and state.static is None:
        raise ValueError("%s asks for %d static obstacles but the config has no "
                         "state.static_obstacles ranges"
                         % (label, spec.static_obstacles))
    if spec.dynamic_obstacles and state.dynamic is None:
        raise ValueError("%s asks for %d dynamic obstacles but the config has "
                         "no state.dynamic_obstacles ranges"
                         % (label, spec.dynamic_obstacles))
    if targets.kind == "leader_formation" and \
            spec.drones > targets.formation_robots:
        raise ValueError(
            "%s models %d drones but the formation holds %d; the extra drones "
            "would have no slot to fly to"
            % (label, spec.drones, targets.formation_robots))


def _final_share(route: Mapping[str, Any]) -> float:
    value = float(route.get("final_share") or 0.0)
    if not 0.0 <= value <= 1.0:
        raise ValueError("state.route.final_share must be a share in [0, 1]")
    return value


def _replace(state: StateRanges, **changes) -> StateRanges:
    import dataclasses
    return dataclasses.replace(state, **changes)


__all__ = ["Attackers", "Box", "FamilyConfig", "Geometry", "Mission",
           "ModelSpec", "MovingGoal", "ObstacleRanges", "Physics", "Plant",
           "Sampling", "Span", "StateRanges", "Targets", "TestRangeSpec",
           "Window", "read_config"]
