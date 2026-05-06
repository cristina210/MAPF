from typing import Optional
from Network_graph import NetworkGraph
from fleet import Fleet
from shortest_path_algorithm.A_star import a_star


class MAPFInstance:
    """
    Description of MAPF problem.
    Contains graph, fleet and time horizon 
    """

    def __init__(self, graph: NetworkGraph, fleet: Fleet, T_sim: int, name: str = "unnamed"):
        """
        Args:
            graph: graph
            fleet: fleet of agents that move on the graph
            T:     time horizon (number of time steps)
            name:  name of instance
        """
        self.graph = graph
        self.fleet = fleet
        self.T = T_sim + 3
        self.name = name

        self._validate()

    def _validate(self) -> None:
        """
        # diventa utile soprattutto quando si legge da file flotta e grafo
        Verified if the instance is consistent:
        - start and goal related to each agent exists on the graph
        - No agent share the same starting node
        - Should not be too small: at least equal to the time to pursue the longest shortest path
        """
        all_nodes = set(self.graph.nodes)
        starts    = []

        # start and goal exist in the graph
        for a in self.fleet.agents.values():
            if a.start not in all_nodes:
                raise ValueError(f"Agent {a.id}: start {a.start} doesn't exist on the graph")
            if a.goal is not None and a.goal not in all_nodes:
                raise ValueError(f"Agent {a.id}: goal {a.goal} doesn't exist on the graph")
            starts.append(a.start)

        # no agent share the same starting node
        if len(starts) != len(set(starts)):
            raise ValueError("Two or more agents share the same starting node")
 
        # Check on T
        for a in self.fleet.agents.values():
            if a.goal is None:
                continue
            shortest = a_star(self.graph, a.start, a.goal, extended=False)
            if shortest is None:
                raise ValueError(
                    f"Agent {a.id}: no path exists from start={a.start} to goal={a.goal} in the graph"
                )
            min_steps = len(shortest) - 1   # number of edges = timesteps needed
            if self.T <= min_steps:
                raise ValueError( f"Agent {a.id}: T={self.T} is too small — at least {min_steps + 1} timesteps "
                    f"are needed to reach goal={a.goal} from start={a.start}. "
                    f"Set T_sim >= {min_steps + 1}.")

    def num_agents(self) -> int:
        return self.fleet.num_agents()

    def __repr__(self) -> str:
        return (f"MAPFInstance(name={self.name}, "
                f"nodes={len(self.graph.nodes)}, "
                f"agents={self.num_agents()}, "
                f"T={self.T})")