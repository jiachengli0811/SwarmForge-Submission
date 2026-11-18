# -*- coding: utf-8 -*-
"""``TestGradient_Swarm.m``'s own top-of-file constants.

Flat module-level names, because a record names its parameters and
``p.radius`` is not a name -- the dot makes it two.  Everything is in the
original's own units: the arena is the 100x100 box ``dims`` declares and a
length is a side of it.
"""

from __future__ import annotations

#: ``Nd = 1+50`` -- one nominal lead plus fifty followers
NB_DRONES = 51

#: ``ind_c = -1`` -- the index of the centre/lead drone.  Negative means NO
#: drone is the lead, so every body in the script is a follower.  See
#: :class:`~gradient_swarm.objects.Drone` for the second, stronger reason the
#: centre branch never runs.
IND_CENTRE = -1

#: ``radius`` -- the cushion radius shared by every drone and every obstacle.
#: It is the range of the repulsive potential, not a drawn body size.
RADIUS = 2.0

#: ``dims`` -- lower/upper bounds per axis.  The script only ever uses the
#: first two rows (it draws with ``axis([...])``); nothing clamps a drone to
#: them, so they are a viewport, not a wall.
DIMS = ((0.0, 100.0), (0.0, 100.0), (0.0, 100.0))

#: ``xc`` -- where the lead drone sits, and the centre of the spawn box
XC = (80.0, 80.0)

#: ``end_loc`` -- the goal every drone is pulled toward
END_LOC = (5.0, 5.0)

#: ``DEL = [xc - 10, xc + 10]`` -- followers spawn uniformly in this box
SPAWN_DELTA = 10.0

#: ``No`` -- how many obstacles
NB_OBSTACLES = 10

#: ``dims_o`` -- the box the obstacles are drawn in.  It sits on the GOAL end
#: of the arena, so the field the swarm has to cross is between it and home.
DIMS_O = ((0.0, 20.0), (0.0, 20.0))

#: ``it >= 200`` -- the iteration cap.  The script's other exit, ``count > 20``,
#: is unreachable: ``count`` is initialised to 0 and never incremented.
MAX_ITERATIONS = 200
