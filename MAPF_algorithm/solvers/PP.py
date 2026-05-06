from Network_graph import NetworkGraph
from fleet import Fleet
from extended_time_graph import TimeExpandedGraph
from shortest_path_algorithm.A_star import a_star
from MAPF_algorithm.plan_result import PlanResult



class PrioritizedPlanner:
    """
    Prioritized Planning MAPF is a MAPF solver working on a time-expanded graph.
    Agents are planned one by one in a specific order (in this implementation the order is the in fleet.agents)
    Each planned path is converted in constraints related to nodes or edges for the following agents guaranteeing the absence of conflicts.


    Note: quality of solution depends on the processing order. 
    Nota: completeness and optimality are not guaranteed.
    """

    def __init__(self, G_original: NetworkGraph, T: int):
        """
        Args:
            G_original: original graph (spatial)
            T: time horizon (taking into consideration in building the time extended graph)
        """
        self.G_original = G_original
        self.T = T
        self.teg  = None 

    def plan(self, fleet: Fleet) -> PlanResult:
        """
        Plan conflict-free path for each agent of the fleet.

        Args:
            fleet: Fleet object
        Returns:
            PlanResult bject
        """
        self.teg = TimeExpandedGraph(self.G_original, self.T)
        result   = PlanResult(success=True)

        for agent in fleet.agents.values():

            # map starting node in from the original graph to the expanded node at t=0
            start_exp = self.teg.get_expanded_id(agent.start, t=0)

            # pursue A* on time-expanded graph till the agent goal node
            path = a_star(self.teg.G_expanded, start_exp, agent.goal, extended=True)

            if path is None:
                # No path found
                return PlanResult(success=False, failed_agent=agent.id)

            # converti in original ids ORA, mentre il teg è ancora integro
            path_original = [self.teg.get_original_id(n) for n in path]

            result.add_path(agent.id, path, path_original)

            # extract constraints from the path just planned
            # all visited nodes become vertex constraints
            nodes_constr = set(path)

            # all visited edges become edges constraints
            edges_constr = {(path[j], path[j + 1]) for j in range(len(path) - 1)}

            # update teg removing nodes and edges
            self.teg.update_teg_with_adding_constraints(nodes_constr, edges_constr)

        return result
