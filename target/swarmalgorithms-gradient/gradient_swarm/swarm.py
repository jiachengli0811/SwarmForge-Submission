# -*- coding: utf-8 -*-
"""The swarm and its iteration -- ``TestGradient_Swarm.m``'s own loop.

THE UPDATE IS SEQUENTIAL
------------------------
This is the one detail of the original that a casual port loses.  ``drones``
is an array of ``handle`` objects and the loop assigns straight into it::

    while( i <= Nd )
        drones(i).pos = drones(i).pos + GradientDescentUpdate(drones(i).pos, i, drones, obst, end_loc);
        i = i + 1;
    end

so drone ``i``'s new position is already in the array when drone ``i+1``
evaluates the field.  The sweep is Gauss-Seidel, not Jacobi: within one
iteration a drone sees its lower-numbered peers where they moved TO and its
higher-numbered peers where they moved FROM.  Collecting every step first and
applying them together would be a different algorithm -- a symmetric one --
and would change which drone gives way when two meet.  :meth:`Swarm.update`
keeps the original order.

NO TIMESTEP
-----------
``GradientDescentUpdate`` returns a displacement and the loop adds it.  There
is no ``dt`` in the algorithm, and one iteration is one iteration.  ``dt``
appears in :meth:`Swarm.update` only to turn the applied step into the
``vel`` a viewer points a marker with and a metric calls a speed; nothing in
the control law reads it.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

from . import params as P
from .descent import gradient_descent_update
from .objects import Drone, Obstacle

Vec = Tuple[float, ...]


class Swarm:
    """The drones, the obstacles and the goal, iterated together."""

    def __init__(self, end_loc: Sequence[float] = P.END_LOC):
        self.drones: List[Drone] = []
        self.obstacles: List[Obstacle] = []
        self.end_loc: Vec = tuple(float(x) for x in end_loc)
        self.algorithm = "gradient_descent"
        self.iteration = 0

    # -- construction --------------------------------------------------------

    def add_drone(self, radius: float, pos: Sequence[float],
                  type: int = 1) -> Drone:
        drone = Drone(radius, pos, type)
        self.drones.append(drone)
        return drone

    def add_obstacle(self, pos: Sequence[float], radius: float,
                     reach: Optional[float] = None) -> Obstacle:
        obstacle = Obstacle(pos, radius, reach)
        self.obstacles.append(obstacle)
        return obstacle

    @property
    def nb_agents(self) -> int:
        return len(self.drones)

    # -- reading the state ---------------------------------------------------

    def get_pos(self) -> List[Vec]:
        return [d.pos for d in self.drones]

    def centroid(self) -> Vec:
        n = len(self.drones) or 1
        dim = len(self.drones[0].pos) if self.drones else 2
        return tuple(sum(d.pos[k] for d in self.drones) / n
                     for k in range(dim))

    # -- commanding it -------------------------------------------------------

    def update_command(self, drone: Drone, index: int) -> Vec:
        """The step for ONE drone, at its current position.

        The whole of the control law, and the only thing a runtime adapter
        replaces.  ``index`` rather than ``drone`` identifies the body to the
        field, because that is how ``getTotalPotential`` excludes a drone from
        its own peer sum.
        """
        return gradient_descent_update(drone.pos, index, self.drones,
                                       self.obstacles, self.end_loc)

    def update(self, dt: float = 1.0) -> None:
        """One iteration of the original's loop, in the original's order."""
        for i, drone in enumerate(self.drones):
            step = self.update_command(drone, i)
            # --- original statement (TestGradient_Swarm.m) ------------------
            drone.pos = tuple(x + s for x, s in zip(drone.pos, step))
            # --- derived, for the viewer and the metrics --------------------
            drone.command = step
            drone.vel = tuple(s / dt for s in step) if dt else step
        self.iteration += 1

    def done(self) -> bool:
        """``count > 20 || it >= 200``.

        Only the second half is live: ``count`` is set to 0 before the loop and
        never touched again, so the iteration cap is the whole of the exit
        condition.  There is no convergence test and no goal test -- a run ends
        when it runs out of iterations.
        """
        return self.iteration >= P.MAX_ITERATIONS


# --------------------------------------------------------------------------- #
# the scene TestGradient_Swarm.m builds
# --------------------------------------------------------------------------- #

def build_scenario(nb_drones: int = P.NB_DRONES,
                   nb_obstacles: int = P.NB_OBSTACLES,
                   radius: float = P.RADIUS,
                   xc: Sequence[float] = P.XC,
                   end_loc: Sequence[float] = P.END_LOC,
                   dims_o: Sequence[Sequence[float]] = P.DIMS_O,
                   spawn_delta: float = P.SPAWN_DELTA,
                   ind_centre: int = P.IND_CENTRE,
                   seed: Optional[int] = 0) -> Swarm:
    """The script's setup: obstacles in ``dims_o``, drones around ``xc``.

    ``ind_centre`` is 0-BASED here, where the script's ``ind_c`` counts from
    1.  At the default ``-1`` neither selects anything, which is the case that
    matters; a caller asking for a real lead drone should count from 0.

    ``rand`` cannot be reproduced -- MATLAB's Mersenne stream and Python's are
    different sequences -- so what is ported is the DRAW: uniform over the same
    box, obstacles before drones, in the same order.  ``seed`` makes a Python
    run repeatable; ``None`` leaves it to the system source.
    """
    rng = random.Random(seed)
    swarm = Swarm(end_loc)

    # obstacles first (the script's own order), uniform over dims_o
    for _ in range(nb_obstacles):
        pos = tuple(rng.random() * (hi - lo) + lo for lo, hi in dims_o)
        swarm.add_obstacle(pos, radius)

    # then the drones: the lead at xc, the followers uniform in xc +/- 10
    for i in range(nb_drones):
        if i == ind_centre:
            swarm.add_drone(radius, xc, 2)
        else:
            pos = tuple(rng.random() * ((c + spawn_delta) - (c - spawn_delta))
                        + (c - spawn_delta) for c in xc)
            swarm.add_drone(radius, pos, 1)

    return swarm
