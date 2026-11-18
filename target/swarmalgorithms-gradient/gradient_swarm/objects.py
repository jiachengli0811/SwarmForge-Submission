# -*- coding: utf-8 -*-
"""``Drone.m`` and ``Obstacle.m`` -- the two bodies that carry a potential.

Both classes are ``handle`` classes over there, so a drone written through one
reference is the same drone seen through another; a Python object is that
already, which is what makes the sequential update in :mod:`gradient_swarm.swarm`
port directly.

THE POTENTIAL
-------------
Both bodies raise the same barrier -- ``1e3 * exp(-(r/R)^2)`` inside ``R``,
nothing outside it -- so a drone and an obstacle repel identically and only
their radius can differ.  ``R`` is a *range*, not a drawn size: the barrier is
finite at ``r = 0`` (1e3), so a body is soft and a drone deep enough inside one
can be pushed through it rather than around.  That is the original's behaviour
and the reason a run is judged on clearance rather than on overlap alone.

THE TYPE FIELD
--------------
``Drone.m``'s constructor takes a ``type`` argument and then ignores it::

    function obj = Drone(radius, pos, type)
        obj.radius = radius;
        obj.pos = pos;
        obj.type = 1;          % <- the argument is not read
        obj.dim = length(pos);

So every drone is type 1 whatever the caller asked for, and every ``type <= 2``
test -- here and in :func:`~gradient_swarm.potential.get_total_potential` --
takes the follower branch.  :meth:`Drone.get_centre_potential` is therefore
dead code in the original.  It is ported anyway, reachable by assigning
``drone.type = 3`` after construction, because it is the only statement of what
the author meant a lead drone to do: hold station at ``R + 0.3`` instead of
merely not colliding.

``TestGradient_Swarm.m`` also passes ``ind_c = -1``, so it never asks for a
lead drone in the first place.  Both facts point the same way and neither
depends on the other.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

Vec = Tuple[float, ...]

#: the peak of the barrier, at ``r = 0``
BARRIER_GAIN = 1e3

#: ``GetCenterPotential``'s stand-off beyond ``R``, and its gain
CENTRE_OFFSET = 0.3
CENTRE_GAIN = 10.0


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    """``del = a - b; r = sqrt(dot(del,del))`` -- the three lines every
    ``GetPotential`` in this codebase opens with."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def barrier(r: float, radius: float) -> float:
    """``1e3 * exp(-(r/R)^2)`` inside ``R``, ``0`` outside.

    The shared body of ``Drone.GetOutsidePotential`` and
    ``Obstacle.GetObstaclePotential``, which are character-for-character the
    same function in the MATLAB.
    """
    if r < radius:
        return BARRIER_GAIN * math.exp(-((r / radius) ** 2))
    return 0.0


class _Body:
    """Position, radius, dimension -- the three fields both classes declare."""

    __slots__ = ("pos", "radius", "dim")

    def __init__(self, pos: Sequence[float], radius: float):
        self.pos: Vec = tuple(float(x) for x in pos)
        self.radius = float(radius)
        self.dim = len(self.pos)

    # -- what a runtime Scene reads ------------------------------------------
    # The MATLAB names are `pos` and `radius`; `position` and `size` are the
    # names swarmmodeling's Scene protocol uses.  Aliases rather than renames,
    # so the port reads like the original and the adapter needs no shim.

    @property
    def position(self) -> Vec:
        return self.pos

    @position.setter
    def position(self, value) -> None:
        self.pos = tuple(float(x) for x in value)

    @property
    def size(self) -> float:
        return self.radius


class Obstacle(_Body):
    """``Obstacle.m``: a static body with a repulsive barrier.

    THE BODY AND THE REACH
    ----------------------
    ``Obstacle.m`` has one ``radius`` doing two jobs, because the script's
    scene has one number: it is the barrier's RANGE (``r < R``) and, to anyone
    drawing the scene, the body's size. They are the same value there and
    ``reach`` defaults to ``radius`` so they stay the same here.

    They are separable because the original conflates them at a cost. The
    barrier fires at ``r < radius`` measured centre to centre, so where the
    body really is ``radius`` wide the repulsion does not begin until the drone
    is already inside it -- an obstacle with no stand-off at all, which is why
    a swarm crossing an obstacle field clips its bodies. A scene that wants a
    stand-off gives the barrier a longer ``reach`` than the body it guards, and
    nothing else changes.

    ``radius`` stays the body -- it is what a viewer draws and what an oracle
    measures a collision with -- and ``reach`` is what the potential reads.
    """

    __slots__ = ("reach",)

    def __init__(self, pos: Sequence[float], radius: float,
                 reach: Optional[float] = None):
        super().__init__(pos, radius)
        self.reach = float(radius if reach is None else reach)

    def get_potential(self, pos_drone: Sequence[float]) -> float:
        """``GetPotential`` -- what this obstacle costs a drone at
        ``pos_drone``."""
        return barrier(_distance(pos_drone, self.pos), self.reach)


class Drone(_Body):
    """``Drone.m``: a body that both feels the field and contributes to it.

    ``command`` and ``vel`` are not in the MATLAB.  ``command`` is the step
    :mod:`gradient_swarm.swarm` last applied, kept so a caller can see it;
    ``vel`` is that step divided by the nominal tick, which exists only so a
    viewer can point the marker and a metric can speak of speed.  The control
    law reads neither -- this algorithm is a position update with no velocity
    state at all.
    """

    __slots__ = ("type", "command", "vel")

    def __init__(self, radius: float, pos: Sequence[float], type: int = 1):
        super().__init__(pos, radius)
        # verbatim: the constructor discards `type` (see the module docstring)
        self.type = 1
        self.command: Vec = tuple(0.0 for _ in self.pos)
        self.vel: Vec = tuple(0.0 for _ in self.pos)

    @property
    def velocity(self) -> Vec:
        """The Scene protocol's name for :attr:`vel` (derived, see above)."""
        return self.vel

    def get_potential(self, pos_drone: Sequence[float]) -> float:
        """``GetPotential`` -- what this drone costs a drone at ``pos_drone``."""
        r = _distance(pos_drone, self.pos)
        if self.type <= 2:
            return self.get_outside_potential(r)
        return self.get_centre_potential(r)

    def get_outside_potential(self, r: float) -> float:
        """``GetOutsidePotential`` -- the follower's barrier: keep out of
        ``R``, and be indifferent beyond it.

        Nothing here pulls a swarm together.  Cohesion in this algorithm comes
        entirely from every drone chasing the same goal.
        """
        return barrier(r, self.radius)

    def get_centre_potential(self, r: float) -> float:
        """``GetCenterPotential`` -- the lead's well: the same barrier inside
        ``R``, and a quadratic outside it minimised at ``R + 0.3``.

        Dead code in the original; see the module docstring.
        """
        if r < self.radius:
            return barrier(r, self.radius)
        return CENTRE_GAIN * (r - (self.radius + CENTRE_OFFSET)) ** 2
