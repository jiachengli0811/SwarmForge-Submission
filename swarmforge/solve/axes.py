from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import FamilyConfig
from .dynamics import formation
from .laws import norm
from .state import State


@dataclass(frozen=True)
class Axis:
    name: str
    label: str
    description: str
    velocity: bool = False
    roles: bool = False


AGENT_AXES = (
    Axis("gap", "gap to contact (steps)",
         "centre distance minus the agent boundary"),
    Axis("closing_speed", "closing speed (vmax)",
         "how fast the two close along the line between them; negative is "
         "moving apart", velocity=True),
    Axis("miss_distance", "miss distance if both coast (steps)",
         "closest approach, minus the agent boundary, if both kept their "
         "velocities; negative is a collision course", velocity=True),
    Axis("time_to_contact", "time to contact (ticks)",
         "gap over closing speed; NaN when they are not closing",
         velocity=True),
    Axis("crossing_angle", "crossing angle (deg)",
         "angle between the two velocities: 0 following, 180 head-on",
         velocity=True),
    Axis("speed", "pair speed (vmax)", "the two drones' mean speed",
         velocity=True),
    Axis("goal_pulls_together", "goal pulls them together (-2..2)",
         "how much the two drones' target directions close them: the "
         "closing speed if each flew straight at its target at unit speed"),
    Axis("goal_distance", "goal distance (steps)",
         "from the pair's midpoint to the mean of the two targets"),
    Axis("third_drone_distance", "nearest third drone (b)",
         "the closest other drone to either of the two"),
    Axis("third_drone_squeeze", "third drone pushes them together (cos)",
         "+1: that drone sits right behind one of the two, on the far side "
         "from its partner"),
    Axis("body_clearance", "nearest body clearance (steps)",
         "surface clearance of the closest body to either of the two"),
    Axis("body_squeeze", "body pushes them together (cos)",
         "+1: that body sits right behind one of the two"),
    Axis("leader_in_pair", "leader in the pair (0/1)",
         "one of the two is the formation leader", roles=True),
)

OBSTACLE_AXES = (
    Axis("clearance", "clearance (steps)",
         "centre distance minus body radius minus agent radius"),
    Axis("approach_speed", "approach speed (vmax)",
         "how fast drone and body close along the line between them"),
    Axis("miss_distance", "miss distance if both coast (steps)",
         "closest approach, minus the contact distance, if both kept their "
         "velocities; negative is a collision course"),
    Axis("time_to_contact", "time to contact (ticks)",
         "clearance over approach speed; NaN when they are not closing"),
    Axis("incidence_angle", "incidence angle (deg)",
         "angle between the drone's velocity and the direction to the body: "
         "0 head-on", velocity=True),
    Axis("speed", "drone speed (vmax)", "the drone's speed", velocity=True),
    Axis("body_moving", "moving body (0/1)", "the body is a dynamic one"),
    Axis("body_speed", "body speed (vmax)", "the body's speed"),
    Axis("body_coming", "body coming at the drone (cos)",
         "+1: the body flies straight at the drone; NaN for a still body"),
    Axis("goal_behind_body", "goal behind the body (cos)",
         "+1: the drone's target lies straight through the body"),
    Axis("goal_distance", "goal distance (steps)",
         "from the drone to its target"),
    Axis("other_drone_distance", "nearest other drone (b)",
         "the closest other drone to the contact drone"),
    Axis("other_drone_squeeze", "other drone pushes it into the body (cos)",
         "+1: that drone sits right behind the contact drone"),
    Axis("second_body_clearance", "second body clearance (steps)",
         "the next-closest body's clearance to the contact drone: a gap "
         "between two bodies"),
    Axis("contact_is_leader", "contact drone is the leader (0/1)",
         "the contact drone is the formation leader", roles=True),
)


TIMEOUT_AXES = (
    Axis("goal_distance", "goal distance (steps)",
         "from the swarm's centroid to the edge of its arrival zone (the goal "
         "disc; a formation's setpoint advance radius until the route's "
         "end); negative inside"),
    Axis("arrived_share", "share arrived (0..1)",
         "the drones already inside the arrival zone"),
    Axis("route_end", "setpoint is the goal (0/1)",
         "a formation's setpoint is the route's last: the goal, where the "
         "leader has to enter the goal disc", roles=True),
    Axis("goal_speed", "speed toward the goal (vmax)",
         "the swarm's mean velocity along the direction to its target; "
         "negative is moving away", velocity=True),
    Axis("speed", "swarm speed (vmax)", "the drones' mean speed",
         velocity=True),
    Axis("polarization", "polarization (0..1)",
         "how much the drones fly one way: the length of their mean unit "
         "velocity", velocity=True),
    Axis("spread", "spread (arrival radii)",
         "root mean square distance of the drones from their centroid, over "
         "the arrival zone's radius: past about 1 the swarm cannot fit it"),
    Axis("spacing", "spacing (b)",
         "the drones' mean distance to their nearest neighbour"),
    Axis("block_distance", "body ahead (steps)",
         "surface distance from the centroid to the nearest body standing "
         "across the swarm's way to its target; NaN when none does"),
    Axis("block_offset", "body ahead off the line (radii)",
         "how far that body's centre sits off the line to the target, in "
         "its contact radius: 0 dead ahead, 1 grazing"),
    Axis("gate_width", "gap ahead (b)",
         "surface gap between the nearest two bodies on either side of the "
         "line to the target ahead of the swarm; NaN when no two straddle it"),
    Axis("mover_distance", "nearest mover (steps)",
         "surface clearance from the centroid to the nearest moving body; "
         "NaN when there is none"),
    Axis("mover_coming", "mover coming at the swarm (cos)",
         "+1: that body flies straight at the centroid; NaN when still or "
         "absent"),
)


REGION_AXES = {
    "obstacle_collision": ("clearance", "approach_speed", "miss_distance",
                           "incidence_angle", "goal_behind_body",
                           "goal_distance", "second_body_clearance",
                           "other_drone_distance", "other_drone_squeeze",
                           "body_moving", "body_coming"),
    "agent_collision": ("gap", "closing_speed", "miss_distance",
                        "crossing_angle", "goal_pulls_together",
                        "goal_distance", "third_drone_distance",
                        "third_drone_squeeze", "body_clearance",
                        "body_squeeze"),
    "timeout": ("goal_distance", "arrived_share", "route_end", "goal_speed",
                "spread", "block_distance", "block_offset", "gate_width",
                "mover_distance", "mover_coming"),
}


def region_axes(model_name: str, config: FamilyConfig) -> Tuple[str, ...]:
    have = {a.name for a in axes_for(model_name, config)}
    return tuple(a for a in REGION_AXES[model_name] if a in have)


def axes_for(model_name: str, config: FamilyConfig):
    if model_name not in ("obstacle_collision", "agent_collision", "timeout"):
        raise ValueError("no projection axes for %r yet" % model_name)
    if model_name == "timeout":
        axes = TIMEOUT_AXES
    else:
        axes = AGENT_AXES if model_name == "agent_collision" else OBSTACLE_AXES
    velocity = config.physics.plant.integration == "velocity"
    roles = config.physics.targets.kind == "leader_formation"
    return tuple(a for a in axes
                 if (velocity or not a.velocity) and (roles or not a.roles))


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(norm(x), 1e-12)[..., None]


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(_unit(a) * _unit(b), axis=-1)


def targets_of(state: State, config: FamilyConfig) -> np.ndarray:
    m = state.drones
    targets = np.repeat(state.target[:, None, :], m, axis=1)
    if config.physics.targets.kind == "leader_formation" and m > 1:
        t = config.physics.targets
        slots = formation(t.formation_robots, state.target,
                          state.target - state.positions[:, 0],
                          t.formation_spacing)
        for i in range(1, m):
            targets[:, i] = slots[i - 1]
    return targets


def _closing(r: np.ndarray, w: np.ndarray, boundary) -> Tuple[np.ndarray, ...]:
    closing = -np.sum(w * _unit(r), axis=-1)
    ww = np.sum(w * w, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(ww > 0.0, -np.sum(r * w, axis=-1) / ww, 0.0)
    t = np.clip(t, 0.0, None)
    miss = norm(r + w * t[:, None]) - boundary
    return closing, miss


def agent_contact(state: State) -> Tuple[np.ndarray, np.ndarray]:
    first, second = np.triu_indices(state.drones, 1)
    d = norm(state.positions[:, first] - state.positions[:, second])
    k = np.argmin(d, axis=1)
    return first[k], second[k]


def obstacle_contact(state: State, config: FamilyConfig
                     ) -> Tuple[np.ndarray, np.ndarray]:
    k = state.obstacles.shape[1]
    clear = _clearances(state, config)
    flat = np.argmin(clear.reshape(state.size, -1), axis=1)
    return flat // k, flat % k


def _clearances(state: State, config: FamilyConfig) -> np.ndarray:
    return config.physics.body_gap(state.positions[:, :, None, :],
                                   state.obstacles[:, None, :, :],
                                   state.obstacle_radii[:, None, :],
                                   state.obstacle_axes[:, None, :, :])


def agent_axes(state: State, config: FamilyConfig
               ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
    physics = config.physics
    step = physics.plant.max_step
    vmax = step / physics.dt
    b = physics.agent_boundary
    rows = np.arange(state.size)
    p, v = state.positions, state.velocities
    i, j = agent_contact(state)
    pi, pj = p[rows, i], p[rows, j]
    r = pj - pi
    u = _unit(r)
    dist = norm(r)
    out: Dict[str, np.ndarray] = {"gap": (dist - b) / step}

    closing, miss = _closing(r, v[rows, j] - v[rows, i], b)
    out["closing_speed"] = closing / vmax
    out["miss_distance"] = miss / step
    with np.errstate(divide="ignore", invalid="ignore"):
        out["time_to_contact"] = np.where(
            closing > 0.0, (dist - b) / (closing * physics.dt), np.nan)
    si, sj = norm(v[rows, i]), norm(v[rows, j])
    moving = (si > 0.0) & (sj > 0.0)
    out["crossing_angle"] = np.where(
        moving, np.degrees(np.arccos(np.clip(_cos(v[rows, i], v[rows, j]),
                                             -1.0, 1.0))), np.nan)
    out["speed"] = 0.5 * (si + sj) / vmax

    targets = targets_of(state, config)
    ti, tj = targets[rows, i], targets[rows, j]
    out["goal_pulls_together"] = -np.sum((_unit(tj - pj) - _unit(ti - pi)) * u,
                                         axis=-1)
    out["goal_distance"] = norm(0.5 * (ti + tj) - 0.5 * (pi + pj)) / step

    def squeeze(where: np.ndarray, sizes: np.ndarray, others: np.ndarray):
        best = np.full(state.size, np.inf)
        cos = np.full(state.size, np.nan)
        for e in range(where.shape[1]):
            for member, partner in ((pi, pj), (pj, pi)):
                gap = norm(member - where[:, e]) - sizes[:, e]
                behind = _cos(member - where[:, e], partner - member)
                better = others[:, e] & (gap < best)
                best = np.where(better, gap, best)
                cos = np.where(better, behind, cos)
        return np.where(np.isfinite(best), best, np.nan), cos

    m = state.drones
    others = np.ones((state.size, m), dtype=bool)
    others[rows, i] = others[rows, j] = False
    third, third_cos = squeeze(p, np.zeros((state.size, m)), others)
    out["third_drone_distance"] = third / b
    out["third_drone_squeeze"] = third_cos
    k = state.obstacles.shape[1]
    body, body_cos = squeeze(
        state.obstacles,
        state.obstacle_radii + physics.agent_radius + physics.obstacle_min_gap,
        np.ones((state.size, k), dtype=bool))
    out["body_clearance"] = body / step
    out["body_squeeze"] = body_cos
    out["leader_in_pair"] = ((i == 0) | (j == 0)).astype(float)
    return (i, j), out


def obstacle_axes(state: State, config: FamilyConfig
                  ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
    physics = config.physics
    step = physics.plant.max_step
    vmax = step / physics.dt
    rows = np.arange(state.size)
    p, v = state.positions, state.velocities
    i, k = obstacle_contact(state, config)
    clear = _clearances(state, config)
    pi, ck = p[rows, i], state.obstacles[rows, k]
    vb = state.obstacle_velocities[rows, k]
    n = _unit(ck - pi)
    contact = physics.obstacle_boundary(state.obstacle_radii[rows, k])
    out: Dict[str, np.ndarray] = {"clearance": clear[rows, i, k] / step}

    closing, miss = _closing(ck - pi, vb - v[rows, i], contact)
    out["approach_speed"] = closing / vmax
    out["miss_distance"] = miss / step
    with np.errstate(divide="ignore", invalid="ignore"):
        out["time_to_contact"] = np.where(
            closing > 0.0, clear[rows, i, k] / (closing * physics.dt), np.nan)
    speed = norm(v[rows, i])
    out["incidence_angle"] = np.where(
        speed > 0.0, np.degrees(np.arccos(np.clip(_cos(v[rows, i], n),
                                                  -1.0, 1.0))), np.nan)
    out["speed"] = speed / vmax
    out["body_moving"] = state.dynamic[k].astype(float)
    body_speed = norm(vb)
    out["body_speed"] = body_speed / vmax
    out["body_coming"] = np.where(body_speed > 0.0, _cos(vb, -n), np.nan)

    targets = targets_of(state, config)
    ti = targets[rows, i]
    out["goal_behind_body"] = _cos(ti - pi, n)
    out["goal_distance"] = norm(ti - pi) / step

    best = np.full(state.size, np.inf)
    cos = np.full(state.size, np.nan)
    for e in range(state.drones):
        d = norm(pi - p[:, e])
        behind = _cos(pi - p[:, e], n)
        better = (i != e) & (d < best)
        best = np.where(better, d, best)
        cos = np.where(better, behind, cos)
    out["other_drone_distance"] = (np.where(np.isfinite(best), best, np.nan)
                                   / physics.agent_boundary)
    out["other_drone_squeeze"] = cos
    if state.obstacles.shape[1] > 1:
        own = np.where(np.arange(state.obstacles.shape[1])[None] == k[:, None],
                       np.inf, clear[rows, i])
        out["second_body_clearance"] = own.min(axis=1) / step
    else:
        out["second_body_clearance"] = np.full(state.size, np.nan)
    out["contact_is_leader"] = (i == 0).astype(float)
    return (i, k), out


def arrival_radius(config: FamilyConfig) -> float:
    targets = config.physics.targets
    if targets.kind == "leader_formation":
        return float(targets.waypoint_advance_within)
    if config.mission is None:
        raise ValueError("%s: the timeout axes read the mission's goal disc, "
                         "and there is no mission block" % config.path)
    return float(config.mission.goal_radius)


def timeout_axes(state: State, config: FamilyConfig
                 ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
    physics = config.physics
    step = physics.plant.max_step
    vmax = step / physics.dt
    b = physics.agent_boundary
    radius = arrival_radius(config)
    rows = np.arange(state.size)
    p, v, g = state.positions, state.velocities, state.target
    m = state.drones
    centroid = p.mean(axis=1)
    to_goal = g - centroid
    along = _unit(to_goal)
    lateral = np.stack([-along[:, 1], along[:, 0]], axis=-1)
    distance = norm(to_goal)
    d = norm(p - g[:, None])
    route_end = np.zeros(state.size, dtype=bool)
    zone = np.full(state.size, radius)
    if config.physics.targets.kind == "leader_formation" and config.mission:
        route_end = norm(state.route) <= 1e-12
        zone = np.where(route_end, float(config.mission.goal_radius), radius)
    out: Dict[str, np.ndarray] = {
        "goal_distance": (distance - zone) / step,
        "arrived_share": np.mean(d <= zone[:, None], axis=1),
        "route_end": route_end.astype(float)}
    goal_dirs = _unit(g[:, None] - p)
    out["goal_speed"] = np.mean(np.sum(v * goal_dirs, axis=-1), axis=1) / vmax
    speeds = norm(v)
    out["speed"] = speeds.mean(axis=1) / vmax
    moving = speeds > 0.0
    with np.errstate(invalid="ignore"):
        heading = np.where(moving[..., None], v / np.maximum(
            speeds, 1e-300)[..., None], 0.0)
    out["polarization"] = np.where(moving.any(axis=1),
                                   norm(heading.sum(axis=1))
                                   / np.maximum(moving.sum(axis=1), 1), np.nan)
    out["spread"] = np.sqrt(np.mean(np.sum((p - centroid[:, None]) ** 2,
                                           axis=-1), axis=1)) / radius
    if m > 1:
        apart = norm(p[:, :, None] - p[:, None])
        apart[:, np.arange(m), np.arange(m)] = np.inf
        out["spacing"] = apart.min(axis=2).mean(axis=1) / b
    else:
        out["spacing"] = np.full(state.size, np.nan)

    k = state.obstacles.shape[1]
    block_distance = np.full(state.size, np.nan)
    block_offset = np.full(state.size, np.nan)
    block_body = np.full(state.size, -1)
    gate = np.full(state.size, np.nan)
    if k:
        rel = state.obstacles - centroid[:, None]
        s = np.sum(rel * along[:, None], axis=-1)
        y = np.sum(rel * lateral[:, None], axis=-1)
        contact = physics.obstacle_boundary(state.obstacle_radii)
        half = np.sqrt(np.mean(np.sum((p - centroid[:, None]) ** 2, axis=-1),
                               axis=1))[:, None]
        ahead = (s > 0.0) & (s < distance[:, None] + contact)
        across = np.abs(y) < contact + half
        surface = norm(rel) - contact
        candidate = np.where(ahead & across, surface, np.inf)
        best = np.argmin(candidate, axis=1)
        found = np.isfinite(candidate[rows, best])
        block_distance = np.where(found, candidate[rows, best] / step, np.nan)
        block_offset = np.where(found, np.abs(y[rows, best])
                                / contact[rows, best], np.nan)
        block_body = np.where(found, best, -1)
        left = np.where(ahead & (y >= 0.0), surface, np.inf)
        right = np.where(ahead & (y < 0.0), surface, np.inf)
        li, ri = np.argmin(left, axis=1), np.argmin(right, axis=1)
        both = np.isfinite(left[rows, li]) & np.isfinite(right[rows, ri])
        span = (norm(state.obstacles[rows, li] - state.obstacles[rows, ri])
                - state.obstacle_radii[rows, li]
                - state.obstacle_radii[rows, ri])
        gate = np.where(both, span / b, np.nan)
    out["block_distance"] = block_distance
    out["block_offset"] = block_offset
    out["gate_width"] = gate

    mover_distance = np.full(state.size, np.nan)
    mover_coming = np.full(state.size, np.nan)
    moving_bodies = np.flatnonzero(state.dynamic)
    if moving_bodies.size:
        centres = state.obstacles[:, moving_bodies]
        clear = (norm(centres - centroid[:, None])
                 - state.obstacle_radii[:, moving_bodies]
                 - physics.agent_radius - physics.obstacle_min_gap)
        near = np.argmin(clear, axis=1)
        mover_distance = clear[rows, near] / step
        vb = state.obstacle_velocities[rows, moving_bodies[near]]
        coming = _cos(vb, centroid - centres[rows, near])
        mover_coming = np.where(norm(vb) > 0.0, coming, np.nan)
    out["mover_distance"] = mover_distance
    out["mover_coming"] = mover_coming
    nearest = np.argmin(d, axis=1)
    return (nearest, block_body), out


def project(model_name: str, state: State, config: FamilyConfig
            ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
    if model_name == "agent_collision":
        contact, values = agent_axes(state, config)
    elif model_name == "obstacle_collision":
        contact, values = obstacle_axes(state, config)
    elif model_name == "timeout":
        contact, values = timeout_axes(state, config)
    else:
        raise ValueError("no axes for model %r" % model_name)
    return contact, {a.name: values[a.name]
                     for a in axes_for(model_name, config)}


def _rotate(x: np.ndarray, axis: np.ndarray) -> np.ndarray:
    c, s = axis[..., 0], axis[..., 1]
    return np.stack([c * x[..., 0] + s * x[..., 1],
                     -s * x[..., 0] + c * x[..., 1]], axis=-1)


def contact_frame(model_name: str, state: State, config: FamilyConfig
                  ) -> Dict[str, np.ndarray]:
    rows = np.arange(state.size)
    m, k = state.drones, state.obstacles.shape[1]
    if model_name == "agent_collision":
        (a, other), _ = agent_axes(state, config)
        axis = _unit(state.positions[rows, other] - state.positions[rows, a])
        firsts = [a, other]
        body_order = np.tile(np.arange(k), (state.size, 1))
    else:
        (a, body), _ = obstacle_axes(state, config)
        axis = _unit(state.obstacles[rows, body] - state.positions[rows, a])
        firsts = [a]
        rest = np.argsort(np.arange(k)[None] != body[:, None], axis=1,
                          kind="stable")
        body_order = rest
    order = np.empty((state.size, m), dtype=int)
    for row in range(state.size):
        head = [int(f[row]) for f in firsts]
        order[row] = head + [d for d in range(m) if d not in head]
    origin = state.positions[rows, a][:, None]
    take = lambda x, idx: np.take_along_axis(x, idx[..., None], axis=1)
    return {
        "positions": _rotate(take(state.positions, order) - origin,
                             axis[:, None]),
        "velocities": _rotate(take(state.velocities, order), axis[:, None]),
        "obstacles": _rotate(take(state.obstacles, body_order) - origin,
                             axis[:, None]),
        "obstacle_velocities": _rotate(take(state.obstacle_velocities,
                                            body_order), axis[:, None]),
        "obstacle_radii": np.take_along_axis(state.obstacle_radii, body_order,
                                             axis=1),
        "dynamic": state.dynamic[body_order],
        "targets": _rotate(take(targets_of(state, config), order)
                           - origin, axis[:, None]),
        "order": order,
        "body_order": body_order,
        "axis": axis,
        "origin": origin[:, 0],
    }


__all__ = ["AGENT_AXES", "Axis", "OBSTACLE_AXES", "REGION_AXES", "TIMEOUT_AXES",
           "agent_axes", "agent_contact", "arrival_radius", "axes_for",
           "contact_frame", "obstacle_axes", "obstacle_contact", "project",
           "region_axes", "targets_of", "timeout_axes"]
