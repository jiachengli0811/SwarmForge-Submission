# -*- coding: utf-8 -*-
"""VASARHELYI SWARM ALGORITHM.

A port of ``@Swarm/compute_vel_vasarhelyi.m``.  It allows the navigation of a
swarm of agents in presence of obstacles and walls.

Ref:      Vasarhelyi, Science Robotics, 2018

Modif:    a cohesion term has been added to make the agents get closer when
          they are farther than ``r0_rep``.

Everything is NED and metres, as in the original: ``pos`` and ``vel`` are
``3 x nb_agents`` with rows ``(north, east, down)``, and a column is one agent.

WHAT THE PORT CHANGED, AND WHY
------------------------------
Two things, both mechanical:

* the per-agent body of the original ``for agent = 1:nb_agents`` loop is
  :func:`compute_vel_agent` -- one function, one drone, the whole law
  including the velocity and acceleration clamps.  The MATLAB applies those
  clamps to the whole ``3 x N`` command matrix after the loop, but they are
  per-column operations, so moving them inside changes no arithmetic and
  gives the control law a single entry point (which is what the runtime
  adapter swaps, and what the extractor reads).
* ``get_v_max``'s ``sqrt`` is taken of ``max(0, ...)``.  The original branch
  reaches that expression at ``r == 0`` exactly, where ``2*a*r - a^2/p^2`` is
  negative and MATLAB returns a *complex* number; the clamp is the intended
  reading and is what the swarm's own solved expressions already assume.

The friction term's sign is the original's.  ``v_rel_u`` is built as
``-v_rel ./ v_rel_norm``, i.e. it points from the neighbour's velocity toward
this agent's, so the term as written pushes the two velocities apart rather
than matching them -- the reverse of Vasarhelyi 2018's ``v_ij`` convention.
That is what ``compute_vel_vasarhelyi.m`` does, so it is what this does; it is
flagged here rather than silently corrected, because a port that "fixes" the
target is no longer the target.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

#: the value ``min_dist_obs`` starts at, so a run with no obstacle in reach
#: still reports a number rather than an infinity
MIN_DIST_OBS_INIT = 20.0


def get_v_max(v_fric: float, r: float, a: float, p: float) -> float:
    """Calculate V fric max -- the braking curve of Vasarhelyi's Eq. (4).

    Below the stopping point the envelope is zero; just above it the ramp is
    linear at gain ``p``; beyond ``a/p^2`` it is the square-root branch of a
    constant-deceleration stop.  ``v_fric`` is the floor: an envelope never
    drops below the velocity slack.
    """
    if r < 0:
        v_fricmax = 0.0
    elif r * p > 0 and r * p < a / p:
        v_fricmax = r * p
    else:
        v_fricmax = math.sqrt(max(0.0, 2 * a * r - a ** 2 / p ** 2))

    if v_fricmax < v_fric:
        v_fricmax = v_fric
    return v_fricmax


def compute_vel_agent(agent: int, pos: np.ndarray, vel: np.ndarray, p_swarm,
                      r_agent: float, dt: float,
                      rng: Optional[np.random.Generator] = None
                      ) -> Tuple[np.ndarray, float, int, float, np.ndarray]:
    """The commanded velocity of one agent: the body of the original loop.

    Inputs:
      agent: index of the drone being commanded
      pos, vel: ``3 x nb_agents`` NED position and velocity of the swarm
      p_swarm: swarm parameters
      r_agent: safety radius of agents
      dt: time step

    Outputs:
      ``(vel_command, nb_agent_collisions, nb_obs_collisions, min_dist_obs,
      neighbours)`` -- the last is this agent's row of the adjacency matrix
      ``M``, which the original assembles and the callers of this port read
      for the neighbourhood metrics.
    """
    nb_agents = pos.shape[1]

    vel_rep = np.zeros(3)       # Repulsion velocity
    vel_fric = np.zeros(3)      # Velocity matching velocity
    vel_wall = np.zeros(3)      # Arena repulsion velocity
    vel_obs = np.zeros(3)       # Obstacle repulsion velocity

    nb_agent_collisions = 0.0
    nb_obs_collisions = 0
    min_dist_obs = MIN_DIST_OBS_INIT
    neighbours = np.zeros(nb_agents, dtype=bool)

    # ---- Find neighbors ---------------------------------------------------

    # Compute agent-agent distance matrix
    p_rel = pos - pos[:, [agent]]
    dist = np.sqrt((p_rel ** 2).sum(axis=0))

    # Define neighbours list.  `dist ~= 0` drops this agent -- and, as in the
    # original, any agent that happens to sit exactly on top of it.
    neig_list = np.flatnonzero(dist != 0)

    # Count collisions
    nb_agent_collisions += int((dist < 2 * r_agent).sum()) - 1

    # Initialize number of neighbours
    nb_neig = nb_agents - 1

    # Constraint on neighborhood given by the euclidean distance
    if p_swarm.r is not None:
        neig_list = neig_list[dist[neig_list] < p_swarm.r]
        nb_neig = len(neig_list)

    # Constraint on neighborhood given by the topological distance
    if p_swarm.max_neig is not None:
        if nb_neig > p_swarm.max_neig:
            idx = np.argsort(dist[neig_list], kind="stable")
            neig_list = neig_list[idx[:p_swarm.max_neig]]
            nb_neig = p_swarm.max_neig

    # Adjacency matrix (asymmetric in case of limited fov)
    neighbours[neig_list] = True

    # ---- Compute different contributions ----------------------------------

    if nb_neig != 0:
        v_rel = vel - vel[:, [agent]]
        v_rel_norm = np.sqrt((v_rel ** 2).sum(axis=0))

        # Compute vel and pos unit vector between two agents.  The self column
        # divides by zero here exactly as it does in MATLAB; it is never read.
        with np.errstate(divide="ignore", invalid="ignore"):
            p_rel_u = -p_rel / dist
            v_rel_u = -v_rel / v_rel_norm

        for agent2 in neig_list:

            # Repulsion and attraction
            if dist[agent2] < p_swarm.r0_rep:      # repulsion
                vel_rep += (p_swarm.p_rep * (p_swarm.r0_rep - dist[agent2])
                            * p_rel_u[:, agent2])
            else:                                  # attraction
                vel_rep += (p_swarm.p_rep * (dist[agent2] - p_swarm.r0_rep)
                            * -p_rel_u[:, agent2])

            # Velocity alignement
            v_fric_max = get_v_max(p_swarm.v_fric,
                                   dist[agent2] - p_swarm.r0_fric,
                                   p_swarm.a_fric, p_swarm.p_fric)

            if v_rel_norm[agent2] > v_fric_max:
                vel_fric += (p_swarm.C_fric * (v_rel_norm[agent2] - v_fric_max)
                             * v_rel_u[:, agent2])

    # ---- Wall and obstacle avoidance --------------------------------------

    # Add arena repulsion effect
    if p_swarm.is_active_arena:
        unit = np.eye(3)
        # On each axis we have the two repulsions
        for axis in range(3):
            # On each axis there is two forces (each side of the arena)
            for direction in range(2):
                dist_ab = abs(pos[axis, agent] - p_swarm.x_arena[axis, direction])

                # Compute velocity of wall shill agent toward center of arena
                v_wall_virtual = unit[:, axis] * p_swarm.v_shill

                if direction == 1:
                    v_wall_virtual = -v_wall_virtual

                # Compute relative velocity (Wall - Agent)
                vel_ab = math.sqrt(((vel[:, agent] - v_wall_virtual) ** 2).sum())

                v_wall_max = get_v_max(0.0, dist_ab - p_swarm.r0_shill,
                                       p_swarm.a_shill, p_swarm.p_shill)

                if vel_ab > v_wall_max:
                    vel_wall += ((vel_ab - v_wall_max)
                                 * (v_wall_virtual - vel[:, agent]) / vel_ab)

    # Compute spheric effect
    if p_swarm.is_active_spheres:

        for obs in range(p_swarm.n_spheres):
            # Get obstacle center and radius
            c_obs = p_swarm.spheres[0:3, obs]
            r_obs = p_swarm.spheres[3, obs]

            # Compute distance agent(a)-obstacle(b)
            dist_ab = math.sqrt(((pos[:, agent] - c_obs) ** 2).sum()) - r_obs
            nb_obs_collisions += int(dist_ab < r_agent)

            # Set the virtual speed of the obstacle direction out of the
            # obstacle
            v_obs_virtual = ((pos[:, agent] - c_obs) / (dist_ab + r_obs)
                             * p_swarm.v_shill)

            # Compute relative velocity agent-obstacle
            vel_ab = math.sqrt(((vel[:, agent] - v_obs_virtual) ** 2).sum())

            if dist_ab < min_dist_obs:
                min_dist_obs = dist_ab

            v_obs_max = get_v_max(0.0, dist_ab - p_swarm.r0_shill,
                                  p_swarm.a_shill, p_swarm.p_shill)

            if vel_ab > v_obs_max:
                vel_obs += ((vel_ab - v_obs_max)
                            * (v_obs_virtual - vel[:, agent]) / vel_ab)

    # Compute cylindric effect
    if p_swarm.is_active_cyl:

        for obs in range(p_swarm.n_cyl):
            # Get obstacle center and radius
            c_obs = p_swarm.cylinders[0:2, obs]
            r_obs = p_swarm.cylinders[2, obs]

            # Compute distance agent(a)-obstacle(b)
            dist_ab = math.sqrt(((pos[0:2, agent] - c_obs) ** 2).sum()) - r_obs
            nb_obs_collisions += int(dist_ab < r_agent)

            # Set the virtual speed of the obstacle direction out of the
            # obstacle
            v_obs_virtual = ((pos[0:2, agent] - c_obs) / (dist_ab + r_obs)
                             * p_swarm.v_shill)

            # Compute relative velocity agent-obstacle
            vel_ab = math.sqrt(((vel[0:2, agent] - v_obs_virtual) ** 2).sum())

            if dist_ab < min_dist_obs:
                min_dist_obs = dist_ab

            v_obs_max = get_v_max(0.0, dist_ab - p_swarm.r0_shill,
                                  p_swarm.a_shill, p_swarm.p_shill)

            if vel_ab > v_obs_max:
                vel_obs[0:2] += ((vel_ab - v_obs_max)
                                 * (v_obs_virtual - vel[0:2, agent]) / vel_ab)

    # ---- Sum agent-agent and obstacle contributions -----------------------

    vel_command = vel_rep + vel_fric + vel_obs + vel_wall

    # Add self propulsion OR migration term
    v_norm = math.sqrt((vel[:, agent] ** 2).sum())

    if p_swarm.is_active_migration:                            # migration
        vel_command = vel_command + p_swarm.v_ref * p_swarm.u_ref
    elif p_swarm.is_active_goal:
        x_goal_rel = p_swarm.x_goal - pos[:, agent]
        u_goal = x_goal_rel / np.linalg.norm(x_goal_rel)
        vel_command = vel_command + p_swarm.v_ref * u_goal
    else:
        # self-propulsion
        if v_norm > 0:
            vel_command = vel_command + p_swarm.v_ref * vel[:, agent] / v_norm

    # ---- Bound velocities and accelerations -------------------------------

    # Add random effect on velocities
    if p_swarm.c_r is not None:
        draw = rng.standard_normal(3) if rng is not None \
            else np.random.standard_normal(3)
        vel_command = vel_command + p_swarm.c_r * draw

    if p_swarm.max_v is not None:
        vel_cmd_norm = math.sqrt((vel_command ** 2).sum())
        if vel_cmd_norm > p_swarm.max_v:
            vel_command = p_swarm.max_v * vel_command / vel_cmd_norm

    if p_swarm.max_a is not None:
        accel_cmd = (vel_command - vel[:, agent]) / dt
        accel_cmd_norm = math.sqrt((accel_cmd ** 2).sum())
        if accel_cmd_norm > p_swarm.max_a:
            vel_command = (vel[:, agent]
                           + dt * p_swarm.max_a * accel_cmd / accel_cmd_norm)

    return (vel_command, nb_agent_collisions, nb_obs_collisions, min_dist_obs,
            neighbours)


def compute_vel_vasarhelyi(swarm, p_swarm, r_agent: float, dt: float,
                           rng: Optional[np.random.Generator] = None
                           ) -> Tuple[np.ndarray, Tuple[float, int, float]]:
    """Commanded velocities for the whole swarm.

    Inputs:
      swarm: the :class:`~vasarhelyi.swarm.Swarm` being commanded
      p_swarm: swarm parameters
      r_agent: safety radius of agents
      dt: time step

    Outputs:
      vel_command: ``3 x nb_agents`` commanded velocities
      collisions: ``(nb_agent_collisions, nb_obs_collisions, min_dist_obs)``
    """
    pos = swarm.get_pos_ned()
    vel = swarm.get_vel_ned()

    nb_agents = swarm.nb_agents
    M = np.zeros((nb_agents, nb_agents), dtype=bool)   # Neighborhood matrix
    vel_command = np.zeros((3, nb_agents))             # Total commanded vel

    nb_agent_collisions = 0.0   # Nb of collisions among agents
    nb_obs_collisions = 0       # Nb of collisions against obstacles
    min_dist_obs = MIN_DIST_OBS_INIT

    for agent in range(nb_agents):
        command, agent_coll, obs_coll, d_obs, neighbours = compute_vel_agent(
            agent, pos, vel, p_swarm, r_agent, dt, rng=rng)
        vel_command[:, agent] = command
        M[agent, :] = neighbours
        nb_agent_collisions += agent_coll
        nb_obs_collisions += obs_coll
        min_dist_obs = min(min_dist_obs, d_obs)

    # Total number of collisions per time step
    nb_agent_collisions = nb_agent_collisions / 2.0   # reciprocal
    collisions = (nb_agent_collisions, nb_obs_collisions, min_dist_obs)

    swarm.neighborhood = M
    return vel_command, collisions
