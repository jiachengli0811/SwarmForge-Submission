# -*- coding: utf-8 -*-
"""``GradientDescentUpdate.m`` -- one drone's step down the field.

Three stages, in the original's order:

1. **the gradient**, by FORWARD differences with ``eps = 1e-4``: one extra
   field evaluation per axis plus one at the point itself;
2. **a speed cap**, on the gradient's magnitude only -- the direction is the
   true unit gradient, taken before the cap, so capping shortens the step
   without turning it;
3. **a line search** over four fixed step sizes, keeping the one that lands
   lowest.

The result is a DISPLACEMENT, not a velocity and not a force: the caller adds
it straight to the position.  There is no timestep anywhere in this algorithm.

DIMENSION
---------
``GradientDescentUpdate.m`` branches on ``dim == 2`` and writes the 3-D case
out again with a third unit vector.  The two branches are the same loop, so
this port writes the loop once and covers both -- and any other ``dim``, which
``Test_Swarm_3d.m`` exercises at 3.

WHAT THE LINE SEARCH ACTUALLY DOES
----------------------------------
``ind_p`` is initialised to the FIRST step (1e-1) and only replaced by one
that strictly lowers the potential::

    Jtmp = J; ind_p = 1;
    ... if( J0 < Jtmp ) Jtmp = J0; ind_p = p; end

so when no step improves on standing still, the drone takes the LARGEST step
anyway rather than none.  That is what keeps a swarm from freezing in a local
minimum between two barriers, and it is also why the potential is not
monotone along a run.  The search is a refinement of the step SIZE only; it
never changes the direction, which is the whole of the coupling structure and
is what :mod:`gradient_swarm.law` restates in closed form.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .potential import get_total_potential

Vec = Tuple[float, ...]

#: ``arr`` -- the candidate step sizes, in the order the search tries them
STEP_SIZES = (1e-1, 1e-2, 1e-3, 1e-4)

#: the forward-difference offset.  ``arr``'s last entry is the same number;
#: they are unrelated and the original spells this one inline four times.
EPSILON = 1e-4

#: ``if( mag >= 10 ) mag = 10;`` -- the cap on how far one step may reach
MAX_DESCENT = 10.0


def finite_difference_gradient(pos: Sequence[float], drone_index: int,
                               drones: Sequence, obst: Sequence,
                               end_loc: Sequence[float]) -> Tuple[List[float], float]:
    """``(grad, J)`` at ``pos``, by forward differences.

    Forward, not central: the original evaluates ``J(x + eps*e_k)`` and ``J(x)``
    and divides by ``eps``, so the gradient carries a first-order truncation
    error of ``eps/2 * d2J/dx2``.  With ``eps = 1e-4`` and the goal term's
    curvature of 200 that is a bias of about 0.01 on a gradient whose magnitude
    is in the thousands -- invisible after the cap, but it is the reason
    :mod:`gradient_swarm.law`'s exact gradient and this one are compared with a
    tolerance rather than for equality.
    """
    J = get_total_potential(pos, drone_index, drones, obst, end_loc)
    grad = []
    for k in range(len(pos)):
        probe = list(pos)
        probe[k] += EPSILON
        Jk = get_total_potential(probe, drone_index, drones, obst, end_loc)
        grad.append((Jk - J) / EPSILON)
    return grad, J


def gradient_descent_update(pos: Sequence[float], drone_index: int,
                            drones: Sequence, obst: Sequence,
                            end_loc: Sequence[float]) -> Vec:
    """``dx`` -- the displacement to add to ``pos`` this iteration."""
    grad, J = finite_difference_gradient(pos, drone_index, drones, obst, end_loc)

    mag = math.sqrt(sum(g * g for g in grad))

    if mag == 0.0:
        # The original writes `u = grad./mag` unconditionally and then returns
        # `u.*0` on this branch, which in MATLAB is 0/0 -> NaN, times zero ->
        # NaN: a drone that reached a stationary point would have its position
        # poisoned rather than held.  The intent of `u.*0` is a zero step, and
        # that is what is returned.  It is reachable only at an exact
        # stationary point -- standing on the goal with no body within R.
        return tuple(0.0 for _ in pos)

    # the UNIT gradient, taken before the cap: capping is a speed limit, not a
    # change of heading
    u = [g / mag for g in grad]
    if mag >= MAX_DESCENT:
        mag = MAX_DESCENT

    Jtmp = J
    ind_p = 0                               # MATLAB's `ind_p = 1`, 0-based
    for p, step in enumerate(STEP_SIZES):
        dp = [x - step * ui * mag for x, ui in zip(pos, u)]
        J0 = get_total_potential(dp, drone_index, drones, obst, end_loc)
        if J0 < Jtmp:
            Jtmp = J0
            ind_p = p

    step = STEP_SIZES[ind_p]
    return tuple(-step * ui * mag for ui in u)
