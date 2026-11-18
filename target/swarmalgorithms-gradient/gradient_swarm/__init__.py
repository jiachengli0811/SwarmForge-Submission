# -*- coding: utf-8 -*-
"""SwarmAlgorithms' gradient-descent swarm, in Python.

A port of ``TestGradient_Swarm.m`` and everything it calls:
``GradientDescentUpdate.m``, ``getTotalPotential.m``,
``FinalLocationPotential.m``, ``Drone.m`` and ``Obstacle.m``.

Every drone descends the same scalar field -- a quadratic bowl on the goal,
plus a bounded repulsive barrier around every obstacle and every other drone
-- by sampling the field's slope with finite differences and line-searching
four fixed step sizes.  There is no velocity state, no timestep and no
alignment or cohesion term: the swarm holds together only because every member
is falling toward the same point.

    from gradient_swarm import build_scenario

    swarm = build_scenario(seed=0)
    while not swarm.done():
        swarm.update()
    print(swarm.centroid())

MODULE MAP
----------
``objects``    -- ``Drone`` / ``Obstacle`` and their potentials
``potential``  -- the total field one drone descends
``descent``    -- the finite-difference gradient, the cap, the line search
``swarm``      -- the sequential sweep and the script's own scene
``law``        -- the same step in closed form, for a program analyser
``params``     -- the script's top-of-file constants
"""

from .descent import (EPSILON, MAX_DESCENT, STEP_SIZES,             # noqa: F401
                      finite_difference_gradient, gradient_descent_update)
from .law import gradient_step, step_terms                          # noqa: F401
from .objects import Drone, Obstacle, barrier                       # noqa: F401
from .potential import (GOAL_GAIN, final_location_potential,        # noqa: F401
                        get_total_potential)
from .swarm import Swarm, build_scenario                            # noqa: F401
from . import law, params                                           # noqa: F401

__all__ = ["Swarm", "Drone", "Obstacle", "build_scenario",
           "gradient_descent_update", "finite_difference_gradient",
           "get_total_potential", "final_location_potential", "barrier",
           "gradient_step", "step_terms", "law", "params",
           "STEP_SIZES", "EPSILON", "MAX_DESCENT", "GOAL_GAIN"]
