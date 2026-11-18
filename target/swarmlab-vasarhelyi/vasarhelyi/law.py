# -*- coding: utf-8 -*-
"""The Vasarhelyi law for ONE agent, stated over objects.

The same control law as :func:`vasarhelyi.control.compute_vel_agent`, in the
shape a program analyser can read: per-agent, per-axis, over object
collections, in one configuration.  ``tests/test_law_parity.py`` asserts the
two agree to floating-point noise, so this is a restatement and not a fork.

WHY IT EXISTS
-------------
``control.py`` is a faithful port of the MATLAB, which means it is written the
way MATLAB writes: ``3 x nb_agents`` arrays, neighbours as integer indices
into them, obstacles as columns of ``p_swarm.cylinders``, and five mode flags
selecting between arena / spheres / cylinders and migration / goal /
self-propulsion.  Each of those is invisible to an extractor reading a control
law:

* whole-array arithmetic hides the per-axis structure a vector direction is
  recovered from;
* an integer index into a matrix is not a coupled body, so no source set can
  be identified;
* five boolean flags are 32 different controllers, and a record describes one.

None of that is a defect in the port -- it is what the original looks like.
This module is the same arithmetic with those three things removed.

THE CONFIGURATION
-----------------
The one the runtime adapter flies for an environment document: cylindric
obstacles on, arena and spheres off, goal-homing rather than migration along a
fixed bearing, and the speed clamp.  ``max_a`` is NOT here -- an acceleration
clamp is a slew limit on the command, which the record schema's global bounds
do not express; the simulation still applies the target's own.

THE PLANE
---------
Two dimensions, north and east.  The world an environment document builds is
planar by construction (see the adapter), so the ``down`` row of the 3-D law
is identically zero and dropping it changes no arithmetic.
"""

from __future__ import annotations

import math

# param_swarm.m + param_vasarhelyi.m, as flat constants.  Flat because a
# record names its parameters, and `p_swarm.r0_rep` is not a name -- the dot
# makes it two.
R0_REP = 25.0        # radius of repulsion (= d_ref)
P_REP = 0.03         # repulsion gain
R0_FRIC = 85.3       # stopping point offset of alignment
C_FRIC = 0.05        # coefficient of velocity alignment
V_FRIC = 0.63        # velocity slack of alignment
P_FRIC = 3.2         # gain of braking curve
A_FRIC = 4.16        # acceleration of braking curve
R0_SHILL = 0.3       # stopping point offset of walls
V_SHILL = 13.6       # velocity of virtual shill agents
P_SHILL = 3.55       # gain of braking curve for walls
A_SHILL = 3.02       # acceleration of braking curve for walls
V_REF = 6.0          # reference speed
MAX_V = 7.0          # speed clamp


def get_v_max(v_fric, r, a, p):
    """The braking curve of Vasarhelyi's Eq. (4) -- see ``control.py``."""
    if r < 0:
        v_fricmax = 0.0
    elif r * p > 0 and r * p < a / p:
        v_fricmax = r * p
    else:
        v_fricmax = math.sqrt(max(0.0, 2 * a * r - a ** 2 / p ** 2))
    if v_fricmax < v_fric:
        v_fricmax = v_fric
    return v_fricmax


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


def compute_vel_agent(self, neighbours, obstacles, targets, dt):
    """The commanded velocity of one agent.

    ``self`` and each of ``neighbours`` carry ``position`` and ``velocity``;
    each obstacle carries ``position`` and ``size`` (its radius); ``targets``
    holds the goal.  ``neighbours`` arrives already narrowed to the metric and
    topological neighbourhood -- ``p_swarm.r`` and ``p_swarm.max_neig`` are
    membership, not law.
    """
    vel_rep = [0.0, 0.0]
    vel_coh = [0.0, 0.0]
    vel_fric = [0.0, 0.0]
    vel_obs = [0.0, 0.0]

    for agent2 in neighbours:
        dx = agent2.position[0] - self.position[0]
        dy = agent2.position[1] - self.position[1]
        dist = magnitude(dx, dy)
        dvx = agent2.velocity[0] - self.velocity[0]
        dvy = agent2.velocity[1] - self.velocity[1]
        v_rel_norm = magnitude(dvx, dvy)

        # repulsion and attraction, as two accumulators rather than one signed
        # one: they are two laws sharing a gain, and a record names them apart
        if dist < R0_REP:
            f = P_REP * (R0_REP - dist) / dist
            vel_rep[0] = vel_rep[0] - f * dx
            vel_rep[1] = vel_rep[1] - f * dy
        if dist >= R0_REP:
            g = P_REP * (dist - R0_REP) / dist
            vel_coh[0] = vel_coh[0] + g * dx
            vel_coh[1] = vel_coh[1] + g * dy

        # velocity alignement
        v_fric_max = get_v_max(V_FRIC, dist - R0_FRIC, A_FRIC, P_FRIC)
        if v_rel_norm > v_fric_max:
            h = C_FRIC * (v_rel_norm - v_fric_max) / v_rel_norm
            vel_fric[0] = vel_fric[0] - h * dvx
            vel_fric[1] = vel_fric[1] - h * dvy

    # cylindric effect: brake toward a virtual agent fleeing the obstacle
    for obs in obstacles:
        ox = self.position[0] - obs.position[0]
        oy = self.position[1] - obs.position[1]
        dist_ab = magnitude(ox, oy) - obs.size
        vx = ox / (dist_ab + obs.size) * V_SHILL - self.velocity[0]
        vy = oy / (dist_ab + obs.size) * V_SHILL - self.velocity[1]
        vel_ab = magnitude(vx, vy)
        v_obs_max = get_v_max(0.0, dist_ab - R0_SHILL, A_SHILL, P_SHILL)
        if vel_ab > v_obs_max:
            k = (vel_ab - v_obs_max) / vel_ab
            vel_obs[0] = vel_obs[0] + k * vx
            vel_obs[1] = vel_obs[1] + k * vy

    # the goal term (p_swarm.is_active_goal)
    vel_goal = [0.0, 0.0]
    for goal in targets:
        gx = goal.position[0] - self.position[0]
        gy = goal.position[1] - self.position[1]
        n = magnitude(gx, gy)
        vel_goal[0] = vel_goal[0] + V_REF * gx / n
        vel_goal[1] = vel_goal[1] + V_REF * gy / n

    command = [
        vel_rep[0] + vel_coh[0] + vel_fric[0] + vel_obs[0] + vel_goal[0],
        vel_rep[1] + vel_coh[1] + vel_fric[1] + vel_obs[1] + vel_goal[1]]
    return limit_magnitude(command, MAX_V)
