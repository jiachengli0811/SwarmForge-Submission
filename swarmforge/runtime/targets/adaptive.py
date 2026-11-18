from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

from ..environment import EnvRuntime, Environment, Frame
from ..scene import Scene, vnorm

TICK_SECONDS = 0.01

_CANDIDATES = ('target/adaptive_swarm/scripts/layered_planner',)


def find_source(root: Optional[str] = None) -> str:
    root = root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    for c in _CANDIDATES:
        p = os.path.join(root, c)
        if os.path.isfile(os.path.join(p, "layered_planner_sim.py")):
            return p
    raise RuntimeError("adaptive_swarm source not found under %s (looked in %s)"
                       % (root, ", ".join(_CANDIDATES)))


_SCRIPT_MARKER = "# Layered Motion Planning"

_CACHE: Dict[str, Tuple] = {}


def load_module(source: Optional[str] = None):
    src = find_source(source)
    if src in _CACHE:
        return _CACHE[src]
    import matplotlib
    matplotlib.use("Agg", force=True)
    if src not in sys.path:
        sys.path.insert(0, src)
    import types
    import potential_fields as pf
    import rrt
    import tools
    path = os.path.join(src, "layered_planner_sim.py")
    with open(path) as f:
        text = f.read()
    cut = text.find(_SCRIPT_MARKER)
    if cut < 0:
        raise RuntimeError("layered_planner_sim.py changed shape: %r not found"
                           % _SCRIPT_MARKER)
    sim = types.ModuleType("layered_planner_sim")
    sim.__file__ = path
    exec(compile(text[:cut], path, "exec"), sim.__dict__)
    sys.modules["layered_planner_sim"] = sim
    _CACHE[src] = (sim, pf, rrt, tools)
    return _CACHE[src]


class _Surface:
    __slots__ = ("position", "size", "inside")

    def __init__(self, position, inside: bool):
        self.position = position
        self.size = 0.0
        self.inside = inside


def _closest_on_segment(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    if L == 0.0:
        return (ax, ay)
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / L
    t = max(0.0, min(1.0, t))
    return (ax + t * dx, ay + t * dy)


def _point_in_poly(p, poly) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i][0], poly[i][1]
        x2, y2 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]
        if (y1 > p[1]) != (y2 > p[1]):
            xx = x1 + (p[1] - y1) * (x2 - x1) / (y2 - y1)
            if p[0] < xx:
                inside = not inside
    return inside


def surface_of(poly, p) -> _Surface:
    best, bd = None, float("inf")
    n = len(poly)
    for i in range(n):
        q = _closest_on_segment(p, poly[i], poly[(i + 1) % n])
        d = math.hypot(q[0] - p[0], q[1] - p[1])
        if d < bd:
            best, bd = q, d
    if _point_in_poly(p, poly):
        cx = sum(v[0] for v in poly) / n
        cy = sum(v[1] for v in poly) / n
        return _Surface((cx, cy), True)
    return _Surface(best, False)


class _Point:
    __slots__ = ("position", "size")

    def __init__(self, position):
        self.position = position
        self.size = 0.0


class AdaptiveScene(Scene):
    dims = 2

    def __init__(self, params: Dict[str, float], nearest_only: bool = False):
        self._params = params
        self.map_obstacles: List = []
        self.peer_obstacles: List = []
        self.nearest_only = nearest_only
        self.penetrations = 0

    def _surfaces(self, polys, agent) -> List[_Surface]:
        p = (float(agent.sp[0]), float(agent.sp[1]))
        out = [surface_of(poly, p) for poly in polys]
        self.penetrations += sum(1 for s in out if s.inside)
        if self.nearest_only and out:
            out = [min(out, key=lambda s: math.hypot(s.position[0] - p[0],
                                                     s.position[1] - p[1]))]
        return out

    def resolve(self, source: str, agent) -> List:
        if source == "obstacles":
            return self._surfaces(self.map_obstacles, agent)
        if source == "agents":
            return self._surfaces(self.peer_obstacles, agent)
        if source == "bodies":
            return self._surfaces(list(self.map_obstacles)
                                  + list(self.peer_obstacles), agent)
        if source == "target":
            return [_Point(tuple(float(x) for x in agent.sp_global))]
        if source == "self":
            return [agent]
        raise ValueError(
            "no source %r in this scene (have: obstacles, agents, bodies, "
            "target, self). Returning nothing here is how a term the record "
            "declares came to be silently inert." % source)

    def role(self, agent) -> str:
        return "leader" if getattr(agent, "leader", False) else "follower"

    def param(self, name: Optional[str]) -> Optional[float]:
        return self._params.get(name) if name else None

    def position(self, obj):
        sp = getattr(obj, "sp", None)
        return tuple(float(x) for x in (obj.position if sp is None else sp))

    def clearance(self, agent, other, dist: float) -> float:
        return dist


def patch(sim, controller, scene: AdaptiveScene):
    import numpy as np
    original = sim.Robot.local_planner

    def local_planner(self, obstacles, params):
        cmd = controller.command(self, scene)
        self.V = params.drone_vel * np.array(cmd)
        self.vel_array.append(vnorm(cmd))
        self.sp = self.sp + np.array(cmd)
        self.route = np.vstack([self.route, self.sp])

    sim.Robot.local_planner = local_planner
    return lambda: setattr(sim.Robot, "local_planner", original)


class World:
    def __init__(self, controller, env: Environment, frame: Frame,
                 source: Optional[str] = None, nearest_only: bool = False,
                 **kw):
        import numpy as np
        import random
        sim, pf, rrt, tools = load_module(source)
        self.np, self.sim, self.pf, self.rrt, self.tools = np, sim, pf, rrt, tools

        params = sim.Params()
        params.animate_rrt = 0
        params.visualize = 0
        params.postprocessing = 0
        for k, v in kw.items():
            setattr(params, k, v)
        self.params = params

        self.env = env
        num_robots = params.num_robots = env.agents
        seed = env.seed
        np.random.seed(seed)
        random.seed(seed)
        default_r = 0.1
        obstacles = [np.array(frame.polygon(b.as_polygon(default_r)),
                              dtype=float) for b in env.static]
        self.env_rt = EnvRuntime(env, frame, default_r,
                                 agent_radius=params.interrobots_dist / 2.0)
        self.shapes = []
        for d, m in zip(env.dynamic, self.env_rt.movers):
            poly = np.array(frame.polygon(d.body.as_polygon(default_r)),
                            dtype=float)
            self.shapes.append(poly - poly.mean(axis=0))
            obstacles.append(poly)
        xy_start = np.array(frame.point(env.start_centre()))
        xy_goal = np.array(frame.point(env.goal))
        self.frame = frame
        self.obstacles = obstacles
        self.map_obstacles = obstacles
        self.xy_start, self.xy_goal = xy_start, xy_goal

        P_long = rrt.rrt_path(obstacles, xy_start, xy_goal, params)
        self.P = rrt.ShortenPath(P_long, obstacles, smoothiters=50)
        self.traj_global = tools.waypts2setpts(self.P, params)

        self.robots = [sim.Robot(i + 1) for i in range(num_robots)]
        leader = self.robots[0]
        leader.leader = True
        leader.route = np.array([self.traj_global[0, :]])
        leader.sp = np.array(leader.route[-1, :], dtype=float)
        self.followers_sp = tools.formation(
            num_robots, leader_des=leader.sp, v=np.array([0, -1]),
            l=params.interrobots_dist)
        for i in range(len(self.followers_sp)):
            self.robots[i + 1].sp = np.array(self.followers_sp[i], dtype=float)
            self.robots[i + 1].route = np.array([self.followers_sp[i]])

        self.scene = AdaptiveScene(controller.params(),
                                   nearest_only=nearest_only)
        self._restore = patch(sim, controller, self.scene)
        self.sp_ind = 0
        self.tick = 0
        self.reached = False
        self.centroid = np.mean([r.sp for r in self.robots], axis=0)


    @property
    def leader(self):
        return self.robots[0]

    def close(self) -> None:
        if self._restore is not None:
            self._restore()
            self._restore = None


    def step(self) -> bool:
        np, tools = self.np, self.tools
        params, leader = self.params, self.leader
        if np.linalg.norm(leader.sp - self.xy_goal) < params.goal_tolerance:
            self.reached = True
            return False
        if self.env_rt is not None:
            self.env_rt.step(len(leader.route), [r.sp for r in self.robots],
                             [getattr(r, "V", (0.0, 0.0)) for r in self.robots],
                             TICK_SECONDS)
            n_static = len(self.env.static)
            for k, mover in enumerate(self.env_rt.movers):
                self.obstacles[n_static + k] = \
                    self.shapes[k] + np.array(mover.position)
            self.map_obstacles = self.obstacles
        leader.sp_global = self.traj_global[self.sp_ind, :]
        self.scene.map_obstacles, self.scene.peer_obstacles = \
            self.map_obstacles, []
        leader.local_planner(self.obstacles, params)

        followers_sp_global = tools.formation(
            params.num_robots, leader.sp_global,
            v=tools.normalize(leader.sp_global - leader.sp),
            l=params.interrobots_dist)
        for i in range(len(followers_sp_global)):
            self.robots[i + 1].sp_global = followers_sp_global[i]
        for p in range(len(self.followers_sp)):
            peers_sp = [x for i, x in
                        enumerate(self.followers_sp + [leader.sp]) if i != p]
            peers = tools.poses2polygons(peers_sp)
            self.scene.map_obstacles, self.scene.peer_obstacles = \
                self.map_obstacles, peers
            self.robots[p + 1].local_planner(
                np.array(list(self.obstacles) + peers), params)
            self.followers_sp[p] = self.robots[p + 1].sp

        self.centroid = np.mean([r.sp for r in self.robots], axis=0)
        if self.sp_ind < self.traj_global.shape[0] - 1 and \
                np.linalg.norm(leader.sp_global - self.centroid) < params.max_sp_dist:
            self.sp_ind += 1
        self.tick += 1
        return True
