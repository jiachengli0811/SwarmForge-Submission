# -*- coding: utf-8 -*-
"""The gradient-descent law for ONE drone, stated over objects.

The same step as :func:`gradient_swarm.descent.gradient_descent_update`, in the
shape a program analyser can read: per-drone, per-axis, over object
collections, with the field's gradient in CLOSED FORM instead of sampled.
``tests/test_law_parity.py`` asserts the two agree to the finite-difference
truncation error, so this is a restatement and not a fork.

WHY IT EXISTS
-------------
``descent.py`` is a faithful port of the MATLAB, and the MATLAB never writes
the control law down.  It writes a scalar field and then *measures* the field's
slope with four extra evaluations of it.  To an extractor reading that code
there is no coupling structure at all: no direction, no per-object term, no
activation -- just a difference of two sums divided by 1e-4.  Differentiating
the field by hand puts the structure back.

THE CLOSED FORM
---------------
The field is ``J(x) = SUM barrier(|x-o|, R_o) + 100*|x-goal|^2`` and
``barrier(r,R) = 1e3*exp(-(r/R)^2)`` inside ``R``.  Term by term::

    -d/dx [ 1e3*exp(-(r/R)^2) ]  =  (2e3*r/R^2) * exp(-(r/R)^2) * unit(x - o)
    -d/dx [ 100*r^2 ]            =  200*r * unit(goal - x)

so the descent direction is a sum of an AWAY term per body inside its radius
and a TOWARD term on the goal -- three terms, one activation, one clamp.  The
two barriers are the same function, and the sources differ only in which
collection they come from; they are kept apart because a record names them
apart and an edit reaches one without the other.

THE ONE RADIUS
--------------
``TestGradient_Swarm.m`` builds every drone AND every obstacle with the same
``radius = 2``, so the algorithm as the script runs it has exactly one barrier
width, and :data:`BARRIER_RADIUS` is it.  The faithful port keeps the width on
each body, because that is where ``Drone.m`` and ``Obstacle.m`` store it; here
it is a named constant, and the difference is invisible for any scene the
original builds.

It is a constant because the record schema cannot hold the alternative, and
not holding it is not a detail.  A term is recorded against ONE distance
convention: either the record's ``dist`` is a centre distance, or it is a
surface distance and the body's radius has already been taken off.  This law
needs both at once -- the Gaussian ``exp(-(r/R)^2)`` is centred on the body and
wants the centre distance, while the gate ``r < R`` is a statement about the
surface.  Written with a per-body radius the term comes back declaring
``centre_to_surface``, every ``dist`` in the magnitude is reinterpreted as a
surface distance, and the width recovers as ``dist - clearance == 0``: the
record evaluates to complex infinity at the first tick.  One constant width
resolves it, and everything stays a centre distance.

The cost is real and it is only outside the original's own scenes: on an
obstacle field whose bodies are NOT radius 2 -- which is what an environment
document builds -- this law repels with the algorithm's width where the
original would use each body's own.  A large obstacle therefore pushes over a
shorter range here than it does in ``descent.py``.
``tests/test_law_parity.py`` pins both halves of that: exact agreement at one
radius, and the divergence measured rather than left to be discovered.

WHERE THE STEP SIZE WENT
------------------------
``descent.py`` writes the step as a scalar on the outside::

    dx = -arr(1) * u * min(|grad|, 10)

and here it is distributed into each term's gain, with the cap restated as a
displacement: ``MAX_STEP = 1e-1 * 10``.  The two are equal for any positive
step, because scaling a vector and its clamp by the same factor commutes::

    k * limit_magnitude(v, M)  ==  limit_magnitude(k*v, k*M)

It is written the distributed way because the outer multiply is what the
top-level expression would otherwise BE -- an extractor reading
``step * (A + B + C)`` recovers one term whose magnitude is ``step`` and whose
direction is the whole controller, which is not a description of anything.
Distributed, the top level is ``A + B + C`` and the three terms are separable,
which is the entire reason this module exists.

WHAT IS NOT HERE
----------------
The line search.  It chooses among four step SIZES by evaluating the field,
which needs the field and not its gradient, and it never changes the
direction.  :data:`STEP` is ``arr(1)``, the one the original uses whenever no
smaller step improves on standing still -- the common case away from a
barrier.  See ``descent.py``'s docstring.

THE PLANE
---------
Two dimensions.  The 3-D branch of ``GradientDescentUpdate.m`` is the same
arithmetic with a third axis, and an environment document is planar.
"""

from __future__ import annotations

import math

# Drone.m + Obstacle.m + FinalLocationPotential.m, as flat constants.  Flat
# because a record names its parameters, and `obj.radius` is not a name.
BARRIER_GAIN = 1e3     # peak of the repulsive barrier, at r = 0
BARRIER_RADIUS = 2.0   # `radius` -- the one width, see the module docstring
GOAL_GAIN = 100.0      # gain of the quadratic goal bowl
MAX_DESCENT = 10.0     # cap on the gradient magnitude one step may use
STEP = 1e-1            # arr(1), the step the line search starts and defaults to
# The cap, restated as a displacement: STEP * MAX_DESCENT.  Spelled as the
# LITERAL and not as that product, which is the tidier way to write it and is
# wrong: a bound's argument has to be a named constant for the extractor to
# record a parameter for it.  Written as an expression the record comes back
# with `clamp_speed()` -- a clamp with no ceiling, which does not clamp -- and
# nothing about the summary says the cap went missing.
MAX_STEP = 1.0


def magnitude(x, y):
    return math.sqrt((x ** 2) + (y ** 2))


def limit_magnitude(vector, max_magnitude, min_magnitude=0.0):
    mag = magnitude(*vector)
    if mag > max_magnitude:
        normalizing_factor = max_magnitude / mag
    elif mag < min_magnitude:
        normalizing_factor = min_magnitude / mag
    else:
        return vector
    return [value * normalizing_factor for value in vector]


def step_terms(self, neighbours, obstacles, targets):
    """The three terms, summed, stepped, and NOT yet capped.

    ``STEP * -grad J``: the displacement this drone would take if the cap never
    fired.  Split out from :func:`gradient_step` so the cap has one statement
    of its own, and so a test can divide the step back out and compare the
    result against the finite differences ``descent.py`` measures.

    ``self``, each of ``neighbours`` and each obstacle carry a ``position``;
    ``targets`` holds the goal.  ``neighbours`` is every OTHER drone -- this
    algorithm has no sensing limit, so membership is the whole swarm, and what
    makes a distant peer contribute nothing is :data:`BARRIER_RADIUS` inside
    the term rather than a neighbourhood outside it.
    """
    vel_obs = [0.0, 0.0]
    vel_rep = [0.0, 0.0]

    # Obstacle repulsion: the gradient of the barrier, pointing outward.
    # `dist < BARRIER_RADIUS` is the whole gate -- `r < R` in the MATLAB.
    # There is no guard against `dist == 0` because none is needed: the gain
    # divides by the barrier width, never by the distance, and the direction
    # vector is zero there anyway.  A guard would also be the second half of a
    # conjunction, and the extractor keeps ONE conjunct as the term's
    # activation -- so a defensive `and dist > 0` is how the barrier's range
    # came to be dropped from the record while the term still looked
    # well-formed.
    for obs in obstacles:
        dx = self.position[0] - obs.position[0]
        dy = self.position[1] - obs.position[1]
        dist = magnitude(dx, dy)
        if dist < BARRIER_RADIUS:
            R = BARRIER_RADIUS
            f = STEP * 2 * BARRIER_GAIN * math.exp(-((dist / R) ** 2)) / (R ** 2)
            vel_obs[0] = vel_obs[0] + f * dx
            vel_obs[1] = vel_obs[1] + f * dy

    # peer repulsion: the same barrier, over the other drones (same gate)
    for agent2 in neighbours:
        dx = self.position[0] - agent2.position[0]
        dy = self.position[1] - agent2.position[1]
        dist = magnitude(dx, dy)
        if dist < BARRIER_RADIUS:
            R = BARRIER_RADIUS
            g = STEP * 2 * BARRIER_GAIN * math.exp(-((dist / R) ** 2)) / (R ** 2)
            vel_rep[0] = vel_rep[0] + g * dx
            vel_rep[1] = vel_rep[1] + g * dy

    # the goal bowl: the only term that reaches across the arena
    vel_goal = [0.0, 0.0]
    for goal in targets:
        gx = goal.position[0] - self.position[0]
        gy = goal.position[1] - self.position[1]
        n = magnitude(gx, gy)
        if n > 0:
            vel_goal[0] = vel_goal[0] + STEP * 2 * GOAL_GAIN * n * gx / n
            vel_goal[1] = vel_goal[1] + STEP * 2 * GOAL_GAIN * n * gy / n

    return [
        vel_obs[0] + vel_rep[0] + vel_goal[0],
        vel_obs[1] + vel_rep[1] + vel_goal[1]]


def gradient_step(self, neighbours, obstacles, targets):
    """The displacement of one drone for one iteration.

    ``dx = -arr(1) * u * min(|grad|, 10)``, with the step distributed into the
    terms and the cap restated as :data:`MAX_STEP` (see the module docstring).
    ``u`` is the unit gradient taken BEFORE the cap over there, so capping
    shortens the step without turning it -- which is what a magnitude clamp on
    the summed terms does, at every magnitude and not only below the cap.
    """
    return limit_magnitude(step_terms(self, neighbours, obstacles, targets),
                           MAX_STEP)
