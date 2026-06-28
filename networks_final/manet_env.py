#note that a majority of the code here was ripped from my RL project last semester (essentially slapped together)
from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple


import gymnasium as gym
from gymnasium import spaces

class _BaseEnv(gym.Env):
    pass

class _DiscreteSpace(spaces.Discrete):
    pass

from routing import hop_count_route, energy_aware_route

#energy model constants

TX_COST  = 0.3   #src transmits
RX_COST  = 0.1   #dest receives
FWD_COST = 0.3   #relay forwards (receive + re-transmit)


class MANETEnv(_BaseEnv):
    """
    params (assume all in meters (where applicable))
    ----------______---------____________------------_----------_________-------
    n_nodes         :number of mobile nodes
    area_size       :side length of the 2D simulation area
    tx_range        :transmission range, nodes within this range are neighbours
    n_flows         :number of fixed source dest traffic flows
    pkts_per_step   :packets generated per flow per timestep
    max_steps       :maximum timesteps before truncation
    alpha, beta     :cost funct weights for energy aware routing
    energy_min/max  :uniform distribution bounds for initial node energy
    node_speed      :maximum node speed per timestep (Random Waypoint model)
    render_mode     :None | human | rgb_array
    _______------------___________------------______________-------------_____________
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}   #   ---> made 30 so it's smoother on speedup, if you want to run full speed sims you can do 30 - 90

    def __init__(
        self,
        n_nodes: int = 30,
        area_size: float = 500.0,
        tx_range: float = 200.0,
        n_flows: int = 5,
        pkts_per_step: int = 1,
        max_steps: int = 800,
        alpha: float = 1.0,
        beta: float = 500.0,
        energy_min: float = 80.0,
        energy_max: float = 100.0,
        node_speed: float = 2.0,
        warmup_steps: int = 30,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # --- Configuration ---
        self.n_nodes      = n_nodes
        self.area_size    = area_size
        self.tx_range     = tx_range
        self.n_flows      = n_flows
        self.pkts_per_step = pkts_per_step
        self.max_steps    = max_steps
        self.alpha        = alpha
        self.beta         = beta
        self.energy_min   = energy_min
        self.energy_max   = energy_max
        self.node_speed   = node_speed
        self.warmup_steps = warmup_steps
        self.render_mode  = render_mode


        #spaces (literally jst for gym compatability) kind of depreciated
        self.action_space = _DiscreteSpace(2)   #0 = stndrd, 1 = energy aware

        #internal state (which is set in reset)
        N = n_nodes
        self.positions:   np.ndarray = np.zeros((N, 2), dtype=np.float32)
        self.waypoints:   np.ndarray = np.zeros((N, 2), dtype=np.float32)
        self.energies:    np.ndarray = np.zeros(N,       dtype=np.float32)
        self.alive:       np.ndarray = np.ones(N,        dtype=bool)
        self.flows:       List[Tuple[int, int]] = []
        self.graph:       nx.Graph = nx.Graph()

        #metrics stuffs
        self._step:               int           = 0
        self._pkts_sent:          int           = 0
        self._pkts_delivered:     int           = 0
        self._path_lengths:       List[int]     = []
        self._dead_count:         int           = 0
        self._first_death:        Optional[int] = None  #doing optional lists here so can default no prob without big phat errors, since may not reach these
        self._time_30pct:         Optional[int] = None
        self._time_disconnect:    Optional[int] = None
        self._energy_var_history: List[float]   = []

        #TA feedback said we should implement logging so we added per step energy snapshots and per packet route logs (these result in HUGE lists)
        self._energy_history:     List[List[float]] = []
        self._route_log:          List[Dict]        = []

        #pygame renderer, were doing lazy init so that we can run sims (using experiment arg) without opening the actual pygame window everysetep
        #since experiment creates hundreds of env instances and we only actual want to see it from visualize
        self._renderer: Optional[Any] = None

    #gymnasium API
    #artifacts from inital heavier gym usage, keeps everything nice and clean so still in use
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None,) -> Tuple[Dict, Dict]:
        #sets up gym rand so we can do the paired trials deterministically from seeds to make scientifically valid
        super().reset(seed=seed, options=options)
        rng = self.np_random

        N = self.n_nodes

        #random initial positions and waypoints
        self.positions = rng.uniform(0, self.area_size, size=(N, 2)).astype(np.float32)
        self.waypoints = rng.uniform(0, self.area_size, size=(N, 2)).astype(np.float32)

        #rand init energy
        self.energies = rng.uniform(self.energy_min, self.energy_max, size=(N,)).astype(np.float32)
        self.alive = np.ones(N, dtype=bool)

        #FIXED, no more double setting src and dest as same node -sorry
        indices = np.arange(N)
        self.flows = []
        used = set()
        for _ in range(self.n_flows):
            available = [i for i in indices if i not in used]
            if len(available) < 2:
                break
            pair = rng.choice(available, size=2, replace=False)
            used.update(pair)
            self.flows.append((int(pair[0]), int(pair[1])))

        #reseting the metrics
        self._step               = 0
        self._pkts_sent          = 0
        self._pkts_delivered     = 0
        self._path_lengths       = []
        self._dead_count         = 0
        self._first_death        = None
        self._time_30pct         = None
        self._time_disconnect    = None
        self._energy_var_history = []
        self._energy_history     = [] 
        self._route_log          = []

        self.graph = self._build_graph()
        return self._obs(), self._info()

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        #1. move nodes, rand waypoitn
        self._move_nodes()

        #2. rebuild graph
        self.graph = self._build_graph()

        #3. route packets for all flows
        ## reassign src and dest of dead nodes to keep it transmitting (purely for our metrics), transversly nodes whose src or dest have died are given new ones
        for i, (src, dst) in enumerate(self.flows):
            living = np.where(self.alive)[0]
            if len(living) < 2:
                break
            if not self.alive[src]:
                candidates = living[living != dst]
                if len(candidates) > 0:
                    src = int(self.np_random.choice(candidates))
                    self.flows[i] = (src, dst)
            if not self.alive[dst]:
                candidates = living[living != src]
                if len(candidates) > 0:
                    dst = int(self.np_random.choice(candidates))
                    self.flows[i] = (src, dst)

        for src, dst in self.flows:
            if not (self.alive[src] and self.alive[dst]):
                continue

            for _ in range(self.pkts_per_step):
                self._pkts_sent += 1

                if action == 0:
                    path = hop_count_route(self.graph, src, dst)
                else:
                    path = energy_aware_route(self.graph, src, dst, self.energies, self.alpha, self.beta)

                if path is not None and len(path) >= 2:
                    self._pkts_delivered += 1
                    self._path_lengths.append(len(path) - 1)

                    #log route slctn with node energies (TA suggested)
                    self._route_log.append({
                        "step":          self._step,
                        "algorithm":     "hop_count" if action == 0 else "energy_aware",
                        "src":           int(src),
                        "dst":           int(dst),
                        "path":          [int(n) for n in path], # fixes int64 serialization error by converting to plain Python int
                        "path_energies": [round(float(self.energies[n]), 2) for n in path],
                        "path_length":   len(path) - 1,
                    })

                    if self._step >= self.warmup_steps:
                        self._apply_energy(path)

        #4.check disconnection milestone
        self._check_disconnect()

        #5. record energy variance
        alive_e = self.energies[self.alive]
        self._energy_var_history.append(float(np.var(alive_e)) if len(alive_e) > 1 else 0.0)

        #snapshot every node's energy at this timestep (from TA suggestion)
        self._energy_history.append([round(float(e), 2) for e in self.energies])

        self._step += 1

        obs  = self._obs()
        info = self._info()
        reward = info["pdr"]  #reward for potential RL use later (were not doing this)

        terminated = self.alive.sum() == 0
        truncated  = self._step >= self.max_steps
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from visualize import MANETRenderer
            self._renderer = MANETRenderer(self)
        return self._renderer.draw()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    #internal helpers

    def _build_graph(self) -> nx.Graph:
        """build undirected graph from crnt pos and alive status"""
        #basically builds networkX graph from current node positions,
        #loops over all living nodess then if theyre close enough (according to TX_range) then a line is drawn between them
        #dead nodes exluded
        G = nx.Graph()
        alive_idx = np.where(self.alive)[0]
        G.add_nodes_from(alive_idx)

        for ii, i in enumerate(alive_idx):
            for j in alive_idx[ii + 1:]:
                dist = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if dist <= self.tx_range:
                    G.add_edge(int(i), int(j))
        #G is what were passing to the routing functions later
        return G

    def _move_nodes(self):
        """advance every live node one step toward its current waypoint"""
        #this is our random waypoint mobility model
        #pretty standard and mentioned in all 3 MANET papers, once nodes arrive at their waypoints they pick a new one, normalized so they all move at same speed
        #dead nodes skipped
        rng = self.np_random
        for i in range(self.n_nodes):
            if not self.alive[i]:
                continue
            diff = self.waypoints[i] - self.positions[i]
            dist = float(np.linalg.norm(diff))
            if dist < self.node_speed:
                self.positions[i] = self.waypoints[i].copy()
                self.waypoints[i] = rng.uniform(0, self.area_size, size=2).astype(np.float32)
            else:
                self.positions[i] += (self.node_speed * diff / dist).astype(np.float32)

    def _apply_energy(self, path: List[int]):
        """subtract energy for one packet routed along path and mark dead nodes"""
        #basic energy model stuffs, applied to path
        #src node transmit
        self.energies[path[0]] -= TX_COST
        #Destt node receive
        self.energies[path[-1]] -= RX_COST
        #relay nodes forward (receive + re-transmit)
        for node in path[1:-1]:
            self.energies[node] -= FWD_COST

        #clamp to zero amd mark newly dead nodes
        for node in path:
            if self.energies[node] <= 0.0:
                self.energies[node] = 0.0
                if self.alive[node]:
                    self.alive[node] = False
                    self._dead_count += 1
                    #milestone for first death
                    if self._first_death is None:
                        self._first_death = self._step
                    #milestoen for 30% dead
                    if (self._time_30pct is None and
                            self._dead_count >= 0.3 * self.n_nodes):
                        self._time_30pct = self._step

    def _check_disconnect(self):
        """mark disconnection time if no flow has a viable path"""
        if self._time_disconnect is not None:
            return
        for src, dst in self.flows:
            if (self.alive[src] and self.alive[dst] and
                    src in self.graph and dst in self.graph and
                    nx.has_path(self.graph, src, dst)):
                return   # at least one flow is still connected
        self._time_disconnect = self._step

    #obs / info builders

    def _obs(self) -> Dict:
        N = self.n_nodes
        adj = np.zeros((N, N), dtype=np.int8)
        for u, v in self.graph.edges():
            adj[u, v] = adj[v, u] = 1
        return {
            "positions":  self.positions.copy(),
            "energies":   self.energies.copy(),
            "alive":      self.alive.astype(np.int8),
            "adjacency":  adj,
        }

    def _info(self) -> Dict:
        alive_e = self.energies[self.alive]
        return {
            "step":               self._step,
            "alive_nodes":        int(self.alive.sum()),
            "dead_nodes":         self._dead_count,
            "pkts_sent":          self._pkts_sent,
            "pkts_delivered":     self._pkts_delivered,
            "pdr":                self._pkts_delivered / max(1, self._pkts_sent),
            "mean_path_length":   float(np.mean(self._path_lengths)) if self._path_lengths else 0.0,
            "energy_variance":    float(np.var(alive_e)) if len(alive_e) > 1 else 0.0,
            "first_death_time":   self._first_death,
            "time_30pct_death":   self._time_30pct,
            "time_disconnection": self._time_disconnect,
            "energy_var_history": list(self._energy_var_history),
            "flows":              list(self.flows),
            #logging (TA suggested)
            "energy_history":     self._energy_history,
            "route_log":          self._route_log,
        }


    #convenience, run full ep with fixed action and return metrics

    def run_episode(self, action: int, seed: int) -> Dict:
        """run one full episode under a fixed routing action then return final metrics"""
        self.reset(seed=seed)
        terminated = truncated = False
        alive_history = []
        while not (terminated or truncated):
            _, _, terminated, truncated, info = self.step(action)
            alive_history.append(info["alive_nodes"])

        info["alive_history"] = alive_history
        return info
