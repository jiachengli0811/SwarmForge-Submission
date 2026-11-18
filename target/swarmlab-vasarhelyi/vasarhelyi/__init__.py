# -*- coding: utf-8 -*-
"""SwarmLab's Vasarhelyi controller, in Python.

A port of ``@Swarm/compute_vel_vasarhelyi.m`` and the point-mass swarm
simulation ``examples/examples_swarm/example_vasarhelyi.m`` drives it with.

    from vasarhelyi import Swarm, param_map, param_sim, param_swarm

    p_sim = param_sim()
    map_params = param_map()
    p_swarm = param_swarm(map_params)

    swarm = Swarm()
    for _ in range(p_swarm.nb_agents):
        swarm.add_drone("point_mass", p_sim)
    swarm.set_pos(p_swarm.Pos0)

    for time in p_sim.times():
        swarm.update_command(p_swarm, p_swarm.r_coll, p_sim.dt)
        swarm.update_state(time=time)
"""

from . import law                                                 # noqa: F401
from .control import (MIN_DIST_OBS_INIT, compute_vel_agent,        # noqa: F401
                      compute_vel_vasarhelyi, get_v_max)
from .params import (MapParams, SimParams, SwarmParams,            # noqa: F401
                     create_shifted_buildings, param_map, param_sim,
                     param_swarm, param_vasarhelyi)
from .performance import (compute_neighborhood,                    # noqa: F401
                          compute_swarm_performance)
from .swarm import Drone, Rb2i, Swarm                              # noqa: F401

__all__ = ["Swarm", "Drone", "Rb2i", "law",
           "compute_vel_vasarhelyi", "compute_vel_agent", "get_v_max",
           "MIN_DIST_OBS_INIT",
           "SimParams", "MapParams", "SwarmParams",
           "param_sim", "param_map", "param_swarm", "param_vasarhelyi",
           "create_shifted_buildings",
           "compute_swarm_performance", "compute_neighborhood"]
