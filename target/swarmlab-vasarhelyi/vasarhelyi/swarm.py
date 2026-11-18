# -*- coding: utf-8 -*-
"""``@Drone/Drone.m`` and ``@Swarm/Swarm.m``, for the point-mass drone.

Only the ``point_mass`` branch is ported.  ``example_vasarhelyi.m`` says
"swarming mode supports only quadcopter and point_mass", and the quadcopter
branch is an autopilot, an estimator, a battery model and a full rigid-body
integrator -- none of which the swarming law reads.  The point-mass branch is
four statements, and they are here verbatim:

    self.vel_xyz = self.command(2:4);
    self.pos_ned = self.pos_ned + self.vel_xyz * self.p_sim.dt;
    self.attitude(3) = self.command(1);

That is the whole of the world's kinematics, which is why a runtime adapter
can swap the control law and leave the flight alone.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .control import compute_vel_vasarhelyi
from .params import SimParams


def Rb2i(phi: float, theta: float, psi: float) -> np.ndarray:
    """``math_tools/Rb2i.m`` -- body frame to inertial frame."""
    cr, cp, cy = math.cos(phi), math.cos(theta), math.cos(psi)
    sr, sp, sy = math.sin(phi), math.sin(theta), math.sin(psi)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp,     sr * cp,                cr * cp]])


class Drone:
    """One point-mass agent: a state, a 4-vector command, and a history."""

    def __init__(self, drone_type: str = "point_mass",
                 p_sim: Optional[SimParams] = None,
                 color: Optional[np.ndarray] = None):
        self.drone_type = drone_type
        self.p_sim = p_sim if p_sim is not None else SimParams()

        self.pos_ned = np.zeros(3)
        self.vel_xyz = np.zeros(3)
        self.attitude = np.zeros(3)
        self.rates = np.zeros(3)

        self.pos_ned_history: List[np.ndarray] = []
        self.vel_xyz_history: List[np.ndarray] = []
        self.path_len = 0.0

        #: ``[psi_command, vn, ve, vd]`` -- the swarm writes 1:4
        self.command = np.zeros(4)
        self.prev_command = np.zeros(4)
        self.color = np.array([0.0, 0.45, 0.74]) if color is None \
            else np.asarray(color, dtype=float)

    def set_pos(self, pos) -> None:
        self.pos_ned = np.asarray(pos, dtype=float).reshape(3).copy()

    def set_vel(self, vel) -> None:
        self.vel_xyz = np.asarray(vel, dtype=float).reshape(3).copy()

    def update_state(self, wind=None, time: float = 0.0) -> None:
        """Compute the new drone state.

        Computes the new drone position with Euler forward method.  This
        method does not take the attitude into account.  We suppose that the
        attitude is always (0,0,0), so the velocity in the body frame
        corresponds to the velocity in the inertial frame.  Only usable with
        the velocity controller.
        """
        if self.drone_type != "point_mass":
            raise NotImplementedError(
                "this port carries the point_mass branch only; "
                "%r needs SwarmLab's autopilot stack" % self.drone_type)

        self.vel_xyz = self.command[1:4].copy()
        self.pos_ned = self.pos_ned + self.vel_xyz * self.p_sim.dt
        self.attitude[2] = self.command[0]   # to plot drone psi angle

        self.pos_ned_history.append(self.pos_ned.copy())
        self.vel_xyz_history.append(self.vel_xyz.copy())


class Swarm:
    """An ensemble of dynamic agents of type :class:`Drone`."""

    def __init__(self):
        self.drones: List[Drone] = []
        self.nb_agents = 0
        self.algorithm = "vasarhelyi"
        self.collisions_history: List[Tuple[float, int, float]] = []
        #: the adjacency matrix the last command was computed over
        self.neighborhood: Optional[np.ndarray] = None

    # -- construction --------------------------------------------------------

    def add_drone(self, drone_type: str = "point_mass",
                  p_sim: Optional[SimParams] = None,
                  color: Optional[np.ndarray] = None) -> Drone:
        self.nb_agents += 1
        drone = Drone(drone_type, p_sim, color)
        self.drones.append(drone)
        return drone

    def set_pos(self, pos) -> None:
        pos = np.asarray(pos, dtype=float)
        for i in range(self.nb_agents):
            self.drones[i].set_pos(pos[:, i])

    def set_vel(self, vel) -> None:
        vel = np.asarray(vel, dtype=float)
        for i in range(self.nb_agents):
            self.drones[i].set_vel(vel[:, i])

    # -- reading the state ---------------------------------------------------

    def get_pos_ned(self) -> np.ndarray:
        """Positions of the agents of the swarm, ``3 x nb_agents``::

                   agent_1   agent_2   ...   agent_N
              pn
              pe
              pd
        """
        pos_ned = np.zeros((3, self.nb_agents))
        for i, drone in enumerate(self.drones):
            pos_ned[:, i] = drone.pos_ned
        return pos_ned

    def get_vel_ned(self) -> np.ndarray:
        """Velocities of the agents of the swarm, ``3 x nb_agents``.

        The body-to-inertial rotation is the original's.  For a point mass the
        commanded heading is written as ``0`` by :meth:`set_vel_commands`, so
        it is the identity -- but it is applied rather than assumed, because
        the attitude is state and a caller may have set one.
        """
        vel_ned = np.zeros((3, self.nb_agents))
        for i, drone in enumerate(self.drones):
            phi, theta, psi = drone.attitude
            vel_ned[:, i] = Rb2i(phi, theta, psi) @ drone.vel_xyz
        return vel_ned

    def get_pos_ned_history(self) -> np.ndarray:
        """``t_steps x (3*nb_agents)``, agent-major -- the layout
        ``example_vasarhelyi.m`` saves and the performance metrics read."""
        return np.hstack([np.array(d.pos_ned_history) for d in self.drones]) \
            if self.drones and self.drones[0].pos_ned_history \
            else np.zeros((0, 3 * self.nb_agents))

    def get_vel_xyz_history(self) -> np.ndarray:
        return np.hstack([np.array(d.vel_xyz_history) for d in self.drones]) \
            if self.drones and self.drones[0].vel_xyz_history \
            else np.zeros((0, 3 * self.nb_agents))

    def get_colors(self) -> np.ndarray:
        return np.column_stack([d.color for d in self.drones])

    # -- commanding it -------------------------------------------------------

    def set_vel_commands(self, commands) -> None:
        commands = np.asarray(commands, dtype=float)
        for i, drone in enumerate(self.drones):
            drone.command[0] = 0.0
            drone.command[1:4] = commands[:, i]

    def update_command(self, p_swarm, r_coll: float, dt: float, rng=None):
        """Select the swarm algorithm and call the associated update."""
        if self.algorithm == "vasarhelyi":
            vel_commands, collisions = compute_vel_vasarhelyi(
                self, p_swarm, r_coll, dt, rng=rng)
        else:
            raise NotImplementedError(
                "this port carries the vasarhelyi law only, not %r"
                % (self.algorithm,))
        self.collisions_history.append(collisions)
        self.set_vel_commands(vel_commands)
        return vel_commands, collisions

    def update_state(self, wind=None, time: float = 0.0) -> None:
        for drone in self.drones:
            drone.update_state(wind, time)
