# -*- coding: utf-8 -*-
"""``compute_swarm_performance.m`` and the helpers it calls.

Six series, one value per recorded time step:

    safety      1 - (share of agent pairs closer than 2*r_coll)
    order       mean over agents of the mean cosine between an agent's
                velocity and its neighbours' -- an agent with no neighbour
                scores 1, as in the original
    union       (nb_agents - connected components) / (nb_agents - 1)
    alg_conn    the graph Laplacian's second-smallest eigenvalue, per agent
    safety_obs  1 - (share of agent-obstacle pairs in contact)
    min_d_obs   per agent, the distance to the nearest obstacle *surface*

``compute_neighborhood.m`` and ``network_components.m`` are ported alongside,
because the metrics are only comparable to SwarmLab's if the graph is built
the same way -- range first, then a topological cut that keeps the ``max_neig``
nearest.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_neighborhood(distance_matrix: np.ndarray, r_comm: Optional[float],
                         max_neig: Optional[int]) -> np.ndarray:
    """The adjacency matrix for a given graph.

    ``M[i, j]`` is 1 when ``i`` is a neighbour of ``j``; column ``j`` holds
    agent ``j``'s neighbours, the diagonal is 0, and the result may be
    asymmetric once the topological cut has been applied.
    """
    n = distance_matrix.shape[1]
    M = np.ones((n, n)) - np.eye(n)
    d = distance_matrix.astype(float).copy()
    np.fill_diagonal(d, np.inf)

    # Constraint on neighborhood given by the limited radius of communication
    if r_comm is not None:
        M[d > r_comm] = 0.0
    nn = M.sum(axis=0)

    # Constraint on neighborhood given by the topological distance
    if max_neig is not None:
        for j in np.flatnonzero(nn > max_neig):
            order = np.argsort(d[:, j], kind="stable")
            M[order[max_neig:], j] = 0.0
    return M


def network_components(adjacency: np.ndarray) -> int:
    """The number of connected components of an undirected graph.

    ``network_components.m`` also returns the component sizes and members;
    only the count is read here, which is the one thing ``union`` needs.
    """
    n = adjacency.shape[0]
    seen = np.zeros(n, dtype=bool)
    components = 0
    for start in range(n):
        if seen[start]:
            continue
        components += 1
        stack = [start]
        seen[start] = True
        while stack:
            v = stack.pop()
            for w in np.flatnonzero(adjacency[v] & ~seen):
                seen[w] = True
                stack.append(int(w))
    return components


def compute_alg_connectivity(A: np.ndarray) -> float:
    """The algebraic connectivity of the graph: the Fiedler eigenvalue."""
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0


def _pdist(points: np.ndarray) -> np.ndarray:
    """MATLAB ``pdist``: the condensed pairwise distance vector of rows."""
    n = points.shape[0]
    if n < 2:
        return np.zeros(0)
    diff = points[:, None, :] - points[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))
    iu = np.triu_indices(n, k=1)
    return d[iu]


def _pdist2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATLAB ``pdist2``: rows of ``a`` against rows of ``b``."""
    if b.shape[0] == 0:
        return np.zeros((a.shape[0], 0))
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def compute_swarm_performance(pos_history: np.ndarray, vel_history: np.ndarray,
                              p_swarm) -> Tuple[np.ndarray, ...]:
    """Performance from the history of the swarming variables.

    Inputs:
      pos_history: ``t_steps x (3*nb_agents)`` series of agents' positions
      vel_history: the matching series of agents' velocities
      p_swarm: structure with swarm parameters

    Outputs:
      ``(safety, order, union, alg_conn, safety_obs, min_d_obs)``
    """
    t_steps, nx = pos_history.shape
    nb_agents = nx // 3

    safety = np.ones(t_steps)
    order = np.zeros(t_steps)
    union = np.zeros(t_steps)
    alg_conn = np.zeros(t_steps)
    safety_obs = np.zeros(t_steps)
    min_d_obs = np.zeros((t_steps, nb_agents))

    for k in range(t_steps):

        # -- Safety: reflects the number of collisions among the swarm agents
        pos_k = pos_history[k, :].reshape(nb_agents, 3)   # rows: agents
        dist_vect_k = _pdist(pos_k)
        if dist_vect_k.size:
            nb_ag_coll = int((dist_vect_k < 2 * p_swarm.r_coll).sum())
            safety[k] = 1.0 - nb_ag_coll / len(dist_vect_k)
        else:
            safety[k] = 1.0

        # -- Order: reflects the correlation of the velocity vectors
        distance_matrix_k = _pdist2(pos_k, pos_k)
        M_k = compute_neighborhood(distance_matrix_k, p_swarm.r,
                                   p_swarm.max_neig)
        vel_k = vel_history[k, :].reshape(nb_agents, 3).T   # 3 x nb_agents
        for agent in range(nb_agents):
            neig = M_k[:, agent] > 0
            nn = int(neig.sum())
            if nn != 0:
                scalar_prod = vel_k[:, agent] @ vel_k[:, neig]
                speed_agent = np.linalg.norm(vel_k[:, agent])
                speed_neig = np.sqrt((vel_k[:, neig] ** 2).sum(axis=0))
                denom = speed_agent * speed_neig
                with np.errstate(divide="ignore", invalid="ignore"):
                    cosines = np.where(denom > 0, scalar_prod / denom, 0.0)
                order[k] += cosines.sum() / nn
            else:
                order[k] += 1.0
        order[k] /= nb_agents

        # -- Union: the number of connected components of the related
        #    symmetric graph
        A = (M_k + M_k.T) > 0
        nb_conn_comp = network_components(A)
        union[k] = (nb_agents - nb_conn_comp) / (nb_agents - 1) \
            if nb_agents > 1 else 1.0

        # -- Connectivity: the algebraic connectivity of the same graph
        alg_conn[k] = compute_alg_connectivity(A.astype(float)) / nb_agents

        # -- Safety with obstacles
        nb_obs_coll = 0
        nb_possible_coll = 0
        if (p_swarm.is_active_spheres or p_swarm.is_active_cyl
                or p_swarm.is_active_arena):

            if p_swarm.is_active_spheres:
                c_spheres = p_swarm.spheres[0:3, :].T
                r_spheres = p_swarm.spheres[3, :]
                D_spheres = _pdist2(pos_k, c_spheres)
                min_d_obs[k, :] = (_pdist2(pos_k[:, 0:2], c_spheres[:, 0:2])
                                   - r_spheres).min(axis=1)
                nb_obs_coll += int((D_spheres < r_spheres).sum())
                nb_possible_coll += nb_agents * len(r_spheres)

            if p_swarm.is_active_cyl and p_swarm.n_cyl:
                c_cyl = p_swarm.cylinders[0:2, :].T
                r_cyl = p_swarm.cylinders[2, :]
                D_cyl = _pdist2(pos_k[:, 0:2], c_cyl)
                min_d_obs[k, :] = (D_cyl - r_cyl).min(axis=1)
                nb_obs_coll += int((D_cyl < r_cyl).sum())
                nb_possible_coll += nb_agents * len(r_cyl)

            if p_swarm.is_active_arena:
                arena = p_swarm.x_arena
                for axis in range(3):
                    nb_obs_coll += int(((pos_k[:, axis] < arena[axis, 0])
                                        | (pos_k[:, axis] > arena[axis, 1])).sum())
                nb_possible_coll += nb_agents

            safety_obs[k] = 1.0 - nb_obs_coll / nb_possible_coll \
                if nb_possible_coll else 1.0

    return safety, order, union, alg_conn, safety_obs, min_d_obs
