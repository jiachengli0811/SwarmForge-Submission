from __future__ import annotations

import math
import os
import random
import sys
import types
from typing import Dict, List, Optional

from ..environment import EnvRuntime, Environment, Frame
from ..scene import Scene

_CANDIDATES = ('target/pyglet-boids',)


def _stub_pyglet() -> None:
    if "pyglet.gl" in sys.modules:
        return
    pg = types.ModuleType("pyglet")
    gl = types.ModuleType("pyglet.gl")
    for n in ("glPushMatrix glPopMatrix glBegin glEnd glColor3f glColor4f "
              "glVertex2f glTranslatef glRotatef glEnable glBlendFunc "
              "glLoadIdentity glClearColor".split()):
        setattr(gl, n, lambda *a, **k: None)
    for n in ("GL_LINE_LOOP GL_LINES GL_TRIANGLES GL_TRIANGLE_FAN GL_BLEND "
              "GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA GL_COLOR_BUFFER_BIT".split()):
        setattr(gl, n, 0)
    pg.gl = gl
    sys.modules["pyglet"] = pg
    sys.modules["pyglet.gl"] = gl


def find_source(root: Optional[str] = None) -> str:
    root = root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    for c in _CANDIDATES:
        p = os.path.join(root, c)
        if os.path.isdir(os.path.join(p, "boids")):
            return p
    raise RuntimeError("pyglet-boids source not found under %s (looked in %s)"
                       % (root, ", ".join(_CANDIDATES)))


def load_module(source: Optional[str] = None):
    _stub_pyglet()
    src = find_source(source)
    if src not in sys.path:
        sys.path.insert(0, src)
    from boids import boid as boid_mod, attractor as att_mod, obstacle as obs_mod
    return boid_mod, att_mod, obs_mod


class BoidsScene(Scene):
    dims = 2

    def __init__(self, world: "World", params: Dict[str, float]):
        self.world = world
        self._params = params

    def resolve(self, source: str, agent) -> List:
        if source == "agents":
            return [b for b in self.world.boids if b is not agent]
        if source == "obstacles":
            return self.world.obstacles
        if source == "target":
            return self.world.attractors
        if source == "self":
            return [agent]
        raise ValueError(
            "no source %r in this scene (have: agents, obstacles, target, "
            "self). Returning nothing here is how a term the record declares "
            "came to be silently inert." % source)

    def param(self, name: Optional[str]) -> Optional[float]:
        return self._params.get(name) if name else None


DEFAULT_OBSTACLE_SIZE = 30.0

MAX_COVER_DISCS = 64


def _oriented_box(poly):
    best = None
    n = len(poly)
    for i in range(n):
        ax = poly[(i + 1) % n][0] - poly[i][0]
        ay = poly[(i + 1) % n][1] - poly[i][1]
        m = math.hypot(ax, ay)
        if m < 1e-9:
            continue
        ux, uy = ax / m, ay / m
        us = [p[0] * ux + p[1] * uy for p in poly]
        vs = [-p[0] * uy + p[1] * ux for p in poly]
        box = (max(vs) - min(vs), (ux, uy), (min(us) + max(us)) / 2.0,
               (max(us) - min(us)) / 2.0, (min(vs) + max(vs)) / 2.0)
        if best is None or box[0] < best[0]:
            best = box
    if best is None:
        return tuple(poly[0]), (1.0, 0.0), 0.0, 0.0
    w, u, uc, half_len, vc = best
    return ((uc * u[0] - vc * u[1], uc * u[1] + vc * u[0]),
            u, half_len, w / 2.0)


def cover_discs(body, default_size: float = DEFAULT_OBSTACLE_SIZE):
    if not body.polygon:
        return [(body.centre(), body.size(default_size))]
    poly = [tuple(p) for p in body.polygon]
    centre, u, half_len, half_w = _oriented_box(poly)
    r = half_w if half_w > 1e-9 else default_size
    if half_len <= r:
        return [(body.centre(), body.size(default_size))]
    span = half_len - r
    n = min(MAX_COVER_DISCS, int(math.ceil(2.0 * span / r)) + 1)
    step = 2.0 * span / (n - 1)
    x0, y0 = centre[0] - u[0] * span, centre[1] - u[1] * span
    return [((x0 + u[0] * step * i, y0 + u[1] * step * i), r) for i in range(n)]


class World:
    def __init__(self, env: Environment, frame: Frame,
                 source: Optional[str] = None):
        self.boid_mod, att_mod, obs_mod = load_module(source)
        Boid, Attractor, Obstacle = self.boid_mod.Boid, att_mod.Attractor, \
            obs_mod.Obstacle
        width, height = env.bounds
        n_boids = env.agents
        rng = random.Random(env.seed)
        self.width, self.height = width, height

        x0, y0, x1, y1 = env.start_region
        self.boids = [Boid(position=[rng.uniform(x0, x1), rng.uniform(y0, y1)],
                           bounds=[width, height],
                           velocity=[rng.uniform(-50.0, 50.0),
                                     rng.uniform(-50.0, 50.0)],
                           color=[rng.random(), rng.random(), rng.random()])
                      for _ in range(n_boids)]
        self.obstacles = [Obstacle(position=[p[0], p[1]], size=r)
                          for b in env.static
                          for p, r in cover_discs(b, DEFAULT_OBSTACLE_SIZE)]
        self.attractors = [Attractor(position=list(env.goal))]
        self.env_rt = EnvRuntime(env, frame, DEFAULT_OBSTACLE_SIZE,
                                 agent_radius=self.boids[0].size if self.boids
                                 else 10.0)
        self.movers, self.mover_parts = [], []
        for d, m in zip(env.dynamic, self.env_rt.movers):
            cx, cy = d.body.centre()
            for p, r in cover_discs(d.body, DEFAULT_OBSTACLE_SIZE):
                self.movers.append(Obstacle(position=[p[0], p[1]], size=r))
                self.mover_parts.append((m, (p[0] - cx, p[1] - cy)))
        self.obstacles.extend(self.movers)
        self.tick = 0

    def step(self, dt: float) -> None:
        if self.env_rt is not None:
            self.env_rt.step(self.tick, [b.position for b in self.boids],
                             [b.velocity for b in self.boids], dt)
            for obj, (m, off) in zip(self.movers, self.mover_parts):
                obj.position[0] = m.position[0] + off[0]
                obj.position[1] = m.position[1] + off[1]
            self.tick += 1
        for b in self.boids:
            b.update(dt, self.boids, self.attractors, self.obstacles)


def patch(world: World, controller):
    mod = world.boid_mod
    original = mod.Boid.update
    scene = BoidsScene(world, controller.params())
    slop = mod._BOUNDARY_SLOP
    wrap = getattr(mod, "_WRAP_AT_BOUNDARY", True)

    def update(self, dt, all_boids, attractors, obstacles):
        scene.world.boids = all_boids
        scene.world.attractors = attractors
        scene.world.obstacles = obstacles
        self.velocity = list(controller.command(self, scene))
        for i in range(0, len(self.position)):
            self.position[i] += dt * self.velocity[i]
            if not wrap:
                continue
            if self.position[i] >= self.wrap_bounds[i]:
                self.position[i] = (self.position[i] % self.wrap_bounds[i]) - slop
            elif self.position[i] < -slop:
                self.position[i] = self.position[i] + self.wrap_bounds[i] + slop

    mod.Boid.update = update
    return lambda: setattr(mod.Boid, "update", original)
