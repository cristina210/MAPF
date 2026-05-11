from typing import Optional
from Network_graph import NetworkGraph
from fleet import Fleet
from shortest_path_algorithm.A_star import a_star
from extended_time_graph import TimeExpandedGraph


class MAPFInstance:
    """
    Description of MAPF problem.
    Contains graph, fleet and time horizon 
    """

    def __init__(self, graph: NetworkGraph, fleet: Fleet, name: str = "unnamed"):
        """
        Args:
            graph: graph
            fleet: fleet of agents that move on the graph
            T:     time horizon (number of time steps)
            name:  name of instance
        """
        self.graph = graph
        self.fleet = fleet
        self.name = name
        # compute T min as the longest shortest path
        self.T_min = self._compute_T_min()

        delta_edges_num_agents = len(self.graph.edges) - self.fleet.num_agents()
        congestion_term = round(len(self.graph.edges) / delta_edges_num_agents) if delta_edges_num_agents > 0 else self.fleet.num_agents()
        self.T = self.T_min + self.fleet.num_agents() + congestion_term 

        self._validate()
    

    def _compute_T_min(self) -> int:
        ''' Compute makespan if each agent plan its shortest path neglecting conflicts with others'''
        self._shortest_paths = {}   # cache riusata in _validate
        max_len = 0
        for a in self.fleet.agents.values():
            if a.goal is None:
                continue
            path = a_star(self.graph, a.start, a.goal)
            if path is None:
                raise ValueError(f"Agent {a.id}: no path from {a.start} to {a.goal}")
            self._shortest_paths[a.id] = path
            max_len = max(max_len, len(path) - 1)
        return max_len

    def _validate(self) -> None:
        """
        Checks instance consistency. Most useful when graph and fleet are loaded
        from external files.
        Checks:
        - Graph has strictly more nodes than agents 
        - Every agent's start and goal exist in the graph
        - No two agents share the same start node
        - A path exists from start to goal for every active agent
        - T is large enough for every active agent to reach its goal (if considered each path indipendently)
        """
        all_nodes = set(self.graph.nodes)

        # graph capacity 
        if len(all_nodes) <= self.fleet.num_agents():
            raise ValueError(f"Graph has {len(all_nodes)} nodes but {self.fleet.num_agents()} agents: " "not enough nodes to assign unique start positions." )

        # per-agent node existence + start uniqueness 
        starts = []
        for a in self.fleet.agents.values():
            if a.start not in all_nodes:
                raise ValueError(f"Agent {a.id}: start node {a.start} not in graph.")
            if a.goal is not None and a.goal not in all_nodes:
                raise ValueError(f"Agent {a.id}: goal node {a.goal} not in graph.")
            starts.append(a.start)

        if len(starts) != len(set(starts)):
            raise ValueError("Two or more agents share the same start node.")

        # path existence + T check 
        for a in self.fleet.agents.values():
            if a.goal is None:
                continue

            path = self._shortest_paths.get(a.id)
            if path is None:
                raise ValueError( f"Agent {a.id}: no path from start={a.start} to goal={a.goal}." )

            min_steps = len(path) - 1
            if self.T <= min_steps:
                raise ValueError(f"Agent {a.id}: T={self.T} is too small — need at least " f"{min_steps + 1} timesteps to reach goal={a.goal} " f"from start={a.start}.")


    def num_agents(self) -> int:
        return self.fleet.num_agents()
