from shortest_path_algorithm.A_star import a_star
from extended_time_graph import TimeExpandedGraph

class PrioritizedPlanner:
    """
    Prioritized Planning MAPF solver using a time-expanded graph.

    Agents are planned one at a time in priority order (order in fleet).
    Each planned path is converted into vertex and edge constraints
    for subsequent agents, ensuring conflict-free paths.
    The time-expanded graph is rebuilt from scratch after each agent
    to enforce the updated constraints.

    Note: solution quality depends on agent priority order.
    Note: completeness is not guaranteed (may fail if lower-priority
    agents have no feasible path given the constraints).
    """

    def __init__(self, G_original: NetworkGraph, T: int):
        """
        Args:
            G_original: original NetworkGraph
            T: number of timesteps in the time-expanded graph
        """
        self.teg = TimeExpandedGraph(G_original, T)
        self.paths = []

    def plan_path(self, fleet):
        """
        Plan a conflict-free path for each agent in the fleet.
        Agents are planned in order.

        Args:
            fleet: Fleet object with initial_locations and goal_locations
        Returns:
            list of paths, one per agent (each path is a list of expanded node ids
            in the time-expanded graph), or None if planning fails for any agent
        """
        initial_locations = fleet.initial_locations
        goal_locations    = fleet.goal_locations

        for i, agent in enumerate(fleet.list_id):
            start_i = initial_locations[i]
            goal_i  = goal_locations[i]

            # map start from original node id to time-expanded node id at t=0
            start_expanded = self.teg.get_expanded_id(start_i, t=0)

            # run A* on the time-expanded graph
            path = a_star(self.teg.G_expanded, start_expanded, goal_i, extended=True)
            if path is None:
                print(f"Agent {agent}: no feasible path found — planning failed")
                return None
            self.paths.append(path)

            # obtain constraints from path already planned
            nodes_constr = set(path)
            edges_constr = set()
            for j in range(len(path) - 1):
                edges_constr.add((path[j], path[j+1]))

            # add constraints from this agent's path and rebuild time extended graph for subsequent agents
            self.teg.update_teg_with_adding_constraints(nodes_constr, edges_constr)

        return self.paths