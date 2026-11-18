# -*- coding: utf-8 -*-
"""The parameter scripts, as data.

A port of ``parameters/param_sim.m``, ``parameters/param_map.m``,
``parameters/param_swarm/param_swarm.m`` and
``parameters/param_swarm/param_vasarhelyi.m``.

SwarmLab's parameter files are *scripts*: they run in the caller's workspace,
read whatever is already there (``DRONE_TYPE``, ``map``, an
``app``-supplied ``nb_agents``) and leave a struct behind.  Here each one is a
function returning that struct, and "the field the script only sets when the
caller has not" is a ``None`` default -- which also preserves the two
``isfield`` tests the control law itself makes: ``p_swarm.r``,
``p_swarm.max_neig`` and ``p_swarm.c_r`` are *optional*, and leaving one unset
is how the original turns that part of the neighbourhood rule off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# param_sim.m
# --------------------------------------------------------------------------- #

@dataclass
class SimParams:
    """Generic parameters for the simulation."""

    dt: float = 0.01           # simulation dt [s]
    dt_plot: float = 0.5       # period of plots update [s]
    dt_video: float = 0.1      # period of video update [s]
    start_time: float = 0.0    # [s]
    end_time: float = 120.0    # [s]

    def times(self) -> np.ndarray:
        """``start_time:dt:end_time`` -- MATLAB's inclusive colon range."""
        n = int(round((self.end_time - self.start_time) / self.dt)) + 1
        return self.start_time + self.dt * np.arange(n)


def param_sim(end_time: Optional[float] = None) -> SimParams:
    p = SimParams()
    if end_time is not None:
        p.end_time = float(end_time)
    return p


# --------------------------------------------------------------------------- #
# param_map.m
# --------------------------------------------------------------------------- #

@dataclass
class MapParams:
    """The city the swarm flies through.

    ``buildings_north`` / ``buildings_east`` are the cylinder centres in NED's
    horizontal plane, which is the only thing the control law reads them for.
    """

    width: float = 300.0
    max_height: float = 100.0
    active_environment: bool = True
    bl_corner_north: float = 0.0
    bl_corner_east: float = 0.0
    nb_blocks: int = 4
    street_width_perc: float = 0.5
    building_width: float = 0.0
    street_width: float = 0.0
    building_shape: str = "cylinder"
    buildings_north: np.ndarray = field(default_factory=lambda: np.zeros(0))
    buildings_east: np.ndarray = field(default_factory=lambda: np.zeros(0))
    buildings_heights: np.ndarray = field(default_factory=lambda: np.zeros(0))


def create_shifted_buildings(map_params: MapParams) -> MapParams:
    """``create_shifted_buildings.m``: a city on a half-offset grid.

    Odd rows carry ``nb_blocks`` buildings on the plain grid; even rows carry
    one fewer, shifted east by half a block pitch.  The heights are fixed.
    """
    m = map_params
    grid = np.array([0.5 * m.width / m.nb_blocks * (2 * i + 1)
                     for i in range(m.nb_blocks)])

    north, east = [], []
    for i in range(1, m.nb_blocks + 1):
        if i % 2 == 1:
            north.append(np.repeat(grid[i - 1], m.nb_blocks))
            east.append(grid.copy())
        else:
            north.append(np.repeat(grid[i - 1], m.nb_blocks - 1))
            east.append(grid[:-1] + (m.building_width + m.street_width) / 2.0)

    m.buildings_north = np.concatenate(north)
    m.buildings_east = np.concatenate(east)
    m.buildings_heights = m.max_height * np.ones(len(m.buildings_east))
    return m


def param_map(drone_type: str = "point_mass",
              active_environment: bool = True) -> MapParams:
    m = MapParams(active_environment=active_environment)
    if drone_type in ("quadcopter", "point_mass"):
        m.width, m.max_height = 300.0, 100.0
    elif drone_type == "fixed_wing":
        m.width, m.max_height = 4000.0, 100.0

    if not active_environment:
        return m                      # the script `return`s before the city

    m.building_width = m.width / m.nb_blocks * (1.0 - m.street_width_perc)
    m.street_width = m.width / m.nb_blocks * m.street_width_perc
    return create_shifted_buildings(m)


# --------------------------------------------------------------------------- #
# param_swarm.m + param_vasarhelyi.m
# --------------------------------------------------------------------------- #

@dataclass
class SwarmParams:
    """``p_swarm``: everything :func:`~vasarhelyi.control.compute_vel_vasarhelyi`
    reads.

    The three optional fields are the ones the control law guards with
    ``isfield``.  ``None`` means the field is absent, exactly as in a struct
    that never had it assigned:

    * ``r``        -- absent: the metric neighbourhood is the whole swarm
    * ``max_neig`` -- absent: no topological cap
    * ``c_r``      -- absent: no random effect on the commanded velocities
    """

    # -- what is switched on -------------------------------------------------
    is_active_migration: bool = True
    is_active_goal: bool = False
    is_active_arena: bool = False
    is_active_spheres: bool = False
    is_active_cyl: bool = True

    nb_agents: int = 5

    # -- neighbourhood -------------------------------------------------------
    r: Optional[float] = 150.0       # metric radius of influence
    max_neig: Optional[int] = 10     # topological cap
    r_coll: float = 0.5              # sphere approximating one drone

    # -- arena ---------------------------------------------------------------
    x_arena: np.ndarray = field(default_factory=lambda: np.array(
        [[-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0]]))
    center_arena: np.ndarray = field(default_factory=lambda: np.zeros(3))
    d_arena: float = 1.5
    c_arena: float = 10.0

    # -- obstacles -----------------------------------------------------------
    #: 4 x n_spheres -- rows are x, y, z, radius
    spheres: np.ndarray = field(default_factory=lambda: np.array(
        [[-70.0, 70.0, 0.0],
         [50.0, 50.0, 200.0],
         [5.0, 5.0, 5.0],
         [50.0, 50.0, 50.0]]))
    n_spheres: int = 3
    #: 3 x n_cyl -- rows are x (north), y (east), radius
    cylinders: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))
    n_cyl: int = 0

    # -- references ----------------------------------------------------------
    d_ref: float = 25.0
    u_ref: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    v_ref: float = 6.0
    #: where ``is_active_goal`` points the swarm
    x_goal: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # -- bounds --------------------------------------------------------------
    max_a: Optional[float] = 10.0
    max_v: Optional[float] = 7.0

    # -- spawn ---------------------------------------------------------------
    P0: np.ndarray = field(default_factory=lambda: np.array([-10.0, 150.0, -50.0]))
    P: float = 20.0
    V0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    V: float = 0.0
    seed: int = 5
    Pos0: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))
    Vel0: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))

    # -- random effect on the command (absent in the shipped parameters) -----
    c_r: Optional[float] = None

    # -- param_vasarhelyi.m --------------------------------------------------
    r0_rep: float = 25.0        # radius of repulsion (= d_ref)
    p_rep: float = 0.03         # repulsion gain

    r0_fric: float = 85.3       # stopping point offset of alignment
    C_fric: float = 0.05        # coefficient of velocity alignment
    v_fric: float = 0.63        # velocity slack of alignment
    p_fric: float = 3.2         # gain of braking curve
    a_fric: float = 4.16        # acceleration of braking curve

    r0_shill: float = 0.3       # stopping point offset of walls
    v_shill: float = 13.6       # velocity of virtual shill agents
    p_shill: float = 3.55       # gain of braking curve for walls
    a_shill: float = 3.02       # acceleration of braking curve for walls


def param_vasarhelyi(p_swarm: SwarmParams) -> SwarmParams:
    """``param_vasarhelyi.m`` -- the algorithm-specific block.

    Only ``r0_rep`` is derived: it is ``d_ref``, so a caller that moved the
    reference inter-agent distance moves the repulsion range with it.
    """
    p_swarm.r0_rep = p_swarm.d_ref
    p_swarm.p_rep = 0.03

    p_swarm.r0_fric = 85.3
    p_swarm.C_fric = 0.05
    p_swarm.v_fric = 0.63
    p_swarm.p_fric = 3.2
    p_swarm.a_fric = 4.16

    p_swarm.r0_shill = 0.3
    p_swarm.v_shill = 13.6
    p_swarm.p_shill = 3.55
    p_swarm.a_shill = 3.02
    return p_swarm


def param_swarm(map_params: Optional[MapParams] = None,
                active_environment: bool = True,
                nb_agents: Optional[int] = None,
                d_ref: Optional[float] = None,
                v_ref: Optional[float] = None,
                swarm_algorithm: str = "vasarhelyi") -> SwarmParams:
    """``param_swarm.m``, then the algorithm's own parameter script.

    ``nb_agents`` / ``d_ref`` / ``v_ref`` are the three the GUI overrides in
    ``example_vasarhelyi.m``; passing one here is that same override, and the
    script's ``if ~isfield`` default stands otherwise.
    """
    p = SwarmParams()
    if nb_agents is not None:
        p.nb_agents = int(nb_agents)
    if d_ref is not None:
        p.d_ref = float(d_ref)
    if v_ref is not None:
        p.v_ref = float(v_ref)

    p.center_arena = p.x_arena.sum(axis=1) / 2.0
    p.n_spheres = p.spheres.shape[1]

    if map_params is not None and active_environment:
        radius = map_params.building_width / 2.0
        n = len(map_params.buildings_east)
        p.cylinders = np.vstack([map_params.buildings_north,
                                 map_params.buildings_east,
                                 np.repeat(radius, n)])
        p.n_cyl = n
    else:
        p.cylinders = np.zeros((3, 0))
        p.n_cyl = 0

    # The spawn.  MATLAB's `rand` after `rng(seed)` is not reproducible from
    # Python, so this is the same *construction* -- a uniform draw in the cube
    # at [P0, P0+P] -- from numpy's generator rather than the same numbers.
    rng = np.random.default_rng(p.seed)
    p.Pos0 = p.P0[:, None] + p.P * rng.random((3, p.nb_agents))
    p.Vel0 = p.V0[:, None] + p.V * rng.random((3, p.nb_agents))

    if swarm_algorithm == "vasarhelyi":
        p = param_vasarhelyi(p)
    return p
