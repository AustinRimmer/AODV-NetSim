from __future__ import annotations

import networkx as nx
import numpy as np
from typing import List, Optional



#Algorithm 1 – hop count AODV

def hop_count_route(G: nx.Graph, src: int, dst: int,) -> Optional[List[int]]: #returns a lsit or nothing
    #all edges have weight 1, ie standard AODV behaviour.

    try:
        return nx.shortest_path(G, source=src, target=dst)
        #this is quite literally just returning dijistrka
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

#Algorithm 2 – energy aware routing (AODVM-style)

def energy_aware_route(
        G: nx.Graph,
        src: int,
        dst: int,
        energies: np.ndarray,
        alpha: float = 1.0,
        beta: float = 50.0,
) -> Optional[List[int]]:
    """returns the lowest cost path under the combined hop + energy metric

    edge cost (u -> v) = alpha  +  beta * (1 / energy[v])

    we sum over the path to give:
        total_cost = alpha * hops + beta * SUM(1/energy[relay])
    matches the vibe of
    Ket and Hippargi's AODVM formula except we dont div over hop count

    params:
    G       :current network graph (only living nodes/edges)
    src     :src node id
    dst     :dest node id
    energees:array of residual energy indexed by node id
    alpha   :weight on hop count
    beta    :weight on inverse energy penalty
    """
    if src not in G or dst not in G:
        return None

    #build a directed weighted graph so we can apply asymmetric edge costs
    #(the energy at the *receiving* node is what matters for the path direction).
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())
    #DG is directed graph

    #u, v are endpoints of the wireless link it checks endpoints per edge in sim
    for u, v in G.edges():
        e_v = max(float(energies[v]), 1e-9) #residual energy for node v
        e_u = max(float(energies[u]), 1e-9) # ''''''
        DG.add_edge(u, v, weight=alpha + beta / e_v)
        DG.add_edge(v, u, weight=alpha + beta / e_u)
    # basically adds to a directed graph, then we use networkx to findshortest path( big d's alg) weights are the weights (duh)
    try:
        return nx.shortest_path(DG, source=src, target=dst, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


#convenience wrapper used by the experiment runner

ALGORITHMS = {
    "hop_count":    hop_count_route,
    "energy_aware": energy_aware_route,
}

