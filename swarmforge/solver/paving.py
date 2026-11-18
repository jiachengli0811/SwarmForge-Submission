from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from swarmforge.solve.state import State

from .engine import UNDECIDED, Boxes, Engine
from .interval import I

POSITION, VELOCITY = "position", "velocity"

CHUNK = 20000


@dataclass
class Slice:
    base: State
    kind: str
    drone: int
    box: Tuple[float, float, float, float]

    def boxes(self, x0, x1, y0, y1) -> Boxes:
        n = int(np.size(x0))
        base = self.base

        def tile(a):
            return np.broadcast_to(a, (n,) + a.shape[1:])
        plo, phi = tile(base.positions).copy(), tile(base.positions).copy()
        vlo, vhi = tile(base.velocities).copy(), tile(base.velocities).copy()
        lo, hi = (plo, phi) if self.kind == POSITION else (vlo, vhi)
        lo[:, self.drone, 0], hi[:, self.drone, 0] = x0, x1
        lo[:, self.drone, 1], hi[:, self.drone, 1] = y0, y1
        return Boxes(I(plo, phi), I(vlo, vhi), tile(base.obstacles),
                     tile(base.obstacle_velocities), tile(base.obstacle_radii),
                     tile(base.obstacle_axes), base.dynamic, tile(base.target),
                     tile(base.route))

    def states(self, x, y) -> State:
        n = int(np.size(x))
        base = self.base
        out = State(**{name: (getattr(base, name) if name == "dynamic" else
                              np.repeat(getattr(base, name), n, axis=0))
                       for name in ("positions", "velocities", "obstacles",
                                    "obstacle_velocities", "obstacle_radii",
                                    "dynamic", "target", "route",
                                    "obstacle_axes")})
        moved = out.positions if self.kind == POSITION else out.velocities
        moved[:, self.drone, 0] = x
        moved[:, self.drone, 1] = y
        return out


@dataclass
class Paving:
    box: Tuple[float, float, float, float]
    max_depth: int
    depth: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    code: np.ndarray
    asked: int

    def area(self) -> np.ndarray:
        return 4.0 ** (-self.depth.astype(float))

    def share(self, code: int) -> float:
        return float(np.sum(self.area()[self.code == code]))

    def raster(self) -> np.ndarray:
        r = 2 ** self.max_depth
        grid = np.full((r, r), UNDECIDED, dtype=np.int8)
        for d in np.unique(self.depth):
            pick = self.depth == d
            s = 2 ** (self.max_depth - int(d))
            ix, iy, code = self.ix[pick] * s, self.iy[pick] * s, self.code[pick]
            for a in range(s):
                for b in range(s):
                    grid[iy + a, ix + b] = code
        return grid


def pave(engine: Engine, piece: Slice, start_depth: int = 3,
         max_depth: int = 9) -> Paving:
    x0, x1, y0, y1 = piece.box
    n = 2 ** start_depth
    ix, iy = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n)))
    out = {"depth": [], "ix": [], "iy": [], "code": []}
    asked = 0
    for depth in range(start_depth, max_depth + 1):
        if ix.size == 0:
            break
        wx, wy = (x1 - x0) / 2 ** depth, (y1 - y0) / 2 ** depth
        code = np.empty(ix.size, dtype=np.int8)
        for lo in range(0, ix.size, CHUNK):
            part = slice(lo, lo + CHUNK)
            code[part] = engine.classify(piece.boxes(
                x0 + ix[part] * wx, x0 + (ix[part] + 1) * wx,
                y0 + iy[part] * wy, y0 + (iy[part] + 1) * wy))
        asked += ix.size
        done = (code != UNDECIDED) | (depth == max_depth)
        out["depth"].append(np.full(int(done.sum()), depth, dtype=np.int8))
        out["ix"].append(ix[done])
        out["iy"].append(iy[done])
        out["code"].append(code[done])
        ix, iy = ix[~done], iy[~done]
        ix = np.concatenate([2 * ix, 2 * ix + 1, 2 * ix, 2 * ix + 1])
        iy = np.concatenate([2 * iy, 2 * iy, 2 * iy + 1, 2 * iy + 1])
    return Paving(box=piece.box, max_depth=max_depth,
                  **{k: np.concatenate(v) for k, v in out.items()}, asked=asked)


__all__ = ["POSITION", "Paving", "Slice", "VELOCITY", "pave"]
