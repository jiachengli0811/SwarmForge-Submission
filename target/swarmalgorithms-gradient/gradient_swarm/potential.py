# -*- coding: utf-8 -*-
"""``getTotalPotential.m`` and ``FinalLocationPotential.m``.

The scalar field one drone descends.  Three sources add into it: every
obstacle, every OTHER drone, and the goal.

    J(x) = SUM_obstacles barrier(|x - o|, R_o)
         + SUM_peers     barrier(|x - p|, R_p)
         + 100 * |x - goal|^2

The goal term is unbounded and quadratic while both barriers are bounded by
1e3, so far from home the field is a bowl and the swarm simply falls toward
``end_loc``; the barriers only matter once a body is within ``R`` of another.

THE BRANCH THAT NEVER RUNS
--------------------------
``getTotalPotential`` splits on the drone's own type::

    if( drone_array(drone_index).type <= 2 )    % peers + goal
    else                                        % goal only
    end

and a lead drone was meant to take the second branch -- answering to the goal
alone while the followers answer to it AND to each other.  Every drone is type
1 (see :mod:`gradient_swarm.objects`), so the first branch always runs.  Both
are ported; only one is reachable without assigning ``drone.type`` by hand.
"""

from __future__ import annotations

import math
from typing import Sequence

#: ``FinalLocationPotential``'s gain.  Two hundred times the barrier's peak
#: over the arena's diagonal, which is why the goal dominates everything until
#: a drone is almost on top of another body.
GOAL_GAIN = 100.0


def final_location_potential(drone_pos: Sequence[float],
                             final_pos: Sequence[float]) -> float:
    """``FinalLocationPotential`` -- ``100 * r^2``, an isotropic bowl on the
    goal."""
    r2 = sum((x - y) ** 2 for x, y in zip(drone_pos, final_pos))
    r = math.sqrt(r2)
    # `100*r*r` rather than `100*r2`: the original takes the square root and
    # squares it back, and on a denormal that is not quite the identity
    return GOAL_GAIN * r * r


def get_total_potential(pos: Sequence[float], drone_index: int,
                        drone_array: Sequence, obst_array: Sequence,
                        end_loc: Sequence[float]) -> float:
    """``getTotalPotential`` -- the field at ``pos`` for drone ``drone_index``.

    ``pos`` is passed separately from ``drone_array[drone_index].pos`` on
    purpose: the finite-difference gradient in :mod:`gradient_swarm.descent`
    probes the field at points the drone is not at, while the drone itself
    stays excluded from the peer sum by its INDEX.  A probe point is therefore
    never repelled by the body doing the probing.
    """
    J = 0.0

    # the obstacles first, exactly as the original orders it -- the sum is
    # floating point, so the order is part of the answer
    for obstacle in obst_array:
        J += obstacle.get_potential(pos)

    if drone_array[drone_index].type <= 2:      # a follower: peers and goal
        for i, drone in enumerate(drone_array):
            if i != drone_index:
                J += drone.get_potential(pos)
        J += final_location_potential(pos, end_loc)
    else:                                       # a lead: the goal alone
        J += final_location_potential(pos, end_loc)

    return J
