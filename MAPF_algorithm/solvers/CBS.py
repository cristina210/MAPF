import heapq
from typing import Optional
from Network_graph import NetworkGraph
from fleet import Fleet
from extended_time_graph import TimeExpandedGraph
from shortest_path_algorithm.A_star import a_star
from MAPF_algorithm.plan_result import PlanResult


class CBSNode:
    """
    Represents a node in the CBS constraint tree.
    Each node contains a set of constraints (vertex and edge) accumulated
    from the root down to this node, and a solution: one path per agent
    that satisfies those constraints.
    The total cost is the sum of all path lengths.

    The tree is explored in best-first order (lowest cost first).
    """


    def __init__(self,vertex_constraints: dict = None,edge_constraints: dict = None,solution: dict = None, parent: "CBSNode" = None):

        self.vertex_constraints = vertex_constraints if vertex_constraints is not None else {}
        self.edge_constraints = edge_constraints if edge_constraints is not None else {}
        self.solution = solution if solution is not None else {}
        self.parent = parent
        self.cost = self._compute_cost()


    def _compute_cost(self) -> int:
        """
        Computes the total cost of the solution as the sum of all path lengths.
        This is the standard Sum-of-Costs objective used in CBS.
        """
        return sum(len(path) for path in self.solution.values())

    def __lt__(self, other: "CBSNode") -> bool:
        # required by heapq to compare nodes with equal cost
        return self.cost < other.cost


class CBSSolver:
    """
    Conflict-Based Search (CBS) MAPF solver.

    CBS operates on two levels:
    - High level: explores a constraint tree in best-first order. Each node holds a set of constraints and a solution.
      When a conflict is detected, the node is split into two children, each adding one new constraint for one of the two agents involved.
    - Low level: for each agent, runs A* on a time-expanded graph built with the constraints of the current node.

    Choice: the time-expanded graph is rebuilt from scratch at each node instead of being stored as a node attribute. Teg is used for the search 
    with A*  (vedere se poi ha senso provare a implementare A* che lavora direttamente nello spazio tempo)

    CBS is complete and optimal under standard assumptions.
    """

    def __init__(self, G_original: NetworkGraph, T: int):
        """
        Args:
            G_original: original spatial graph
            T: time horizon for the time-expanded graph
        """
        self.G_original = G_original
        self.T = T



    def plan(self, fleet: Fleet) -> PlanResult:
        """
        Finds a conflict-free optimal plan for all agents using CBS.

        Args:
            fleet: Fleet object with start and goal for each agent
        Returns:
            PlanResult with planned paths, or success=False if no solution exists
        """
        # root node: no constraints, each agent gets its individual shortest path
        self.iter = 0 # debug
        
        root_solution = self._plan_all_agents(fleet, vertex_constraints={}, edge_constraints={})
        if root_solution is None:
            # at least one agent has no path even without constraints
            return PlanResult(success=False)

        root = CBSNode(solution=root_solution)

        # min-heap ordered by cost
        heap = [root]

        while heap:

            ##
            self.iter += 1
            if self.iter % 10 == 0:
                print(f"[CBS] iterations={self.iter}, heap={len(heap)}")
            ##

            node = heapq.heappop(heap)
            print(f"[CBS] expanded node cost={node.cost}, heap_size={len(heap)}")   # debug

            # detect the first conflict in the current solution
            conflict = self._detect_conflict(node.solution, fleet) 

            print(f"[CBS] conflict detected: {conflict}")  # debug

            if conflict is None:
                # no conflicts: this solution is valid and optimal
                return self._build_plan_result(node.solution, fleet)

            agent_tuple, vertex_constr, edge_constr = conflict

            # split: generate one child per agent involved in the conflict
            # each child inherits all parent constraints plus one new constraint
            for agent_id in agent_tuple:

                # copy parent constraints and add the new one for this child
                new_vertex = {k: set(v) for k, v in node.vertex_constraints.items()}
                new_edge = {k: set(v) for k, v in node.edge_constraints.items()}


                if agent_id not in new_vertex:
                    new_vertex[agent_id] = set()
                if agent_id not in new_edge:
                    new_edge[agent_id] = set()

                if vertex_constr is not None:
                    new_vertex[agent_id].add(vertex_constr)

                if edge_constr is not None:
                    new_edge[agent_id].add(edge_constr)

                new_path = self._plan_single_agent(fleet.get_agent(agent_id), vertex_constraints=new_vertex[agent_id], edge_constraints=new_edge[agent_id])


                # prune this child if the constrained agent has no feasible path
                if new_path is None:
                    continue

                new_solution = dict(node.solution)
                new_solution[agent_id] = new_path

                child = CBSNode(vertex_constraints=new_vertex, edge_constraints=new_edge, solution=new_solution, parent=node)
                heapq.heappush(heap, child)
            print(f"[CBS] branching agent {agent_id}") # debug

        # heap exhausted without finding a conflict-free solution
        return PlanResult(success=False)


    def _plan_all_agents(self, fleet: Fleet,vertex_constraints: dict = None, edge_constraints: dict = None) -> Optional[dict]:
        """
        Plans a path for every active agent independently using A*.
        Used to build the solution of the root node (no constraints).

        A new time-expanded graph is built with the given constraints before running A* for each agent.

        Args:
            fleet:               fleet of agents
            vertex_constraints:  expanded node ids forbidden for all agents
            edge_constraints:    expanded edge pairs forbidden for all agents
        Returns:
            dict agent_id -> path (expanded node ids), or None if any agent
            has no feasible path under the given constraints
        """
        teg = TimeExpandedGraph(self.G_original, self.T, vertex_constraints, edge_constraints)
        solution = {}

        for agent in fleet.agents.values():
            if agent.goal is None:
                continue   # idle agents are not planned

            print(f"[CBS] Agent {agent.id} - TE nodes: {len(teg.G_expanded.nodes)}")  # debug

            start_exp = teg.get_expanded_id(agent.start, t=0)
            path = a_star(teg.G_expanded, start_exp, agent.goal, extended=True)

            if path is None:
                return None   # no feasible path for this agent

            print(f"[CBS] Agent {agent.id} - path length (depth): {len(path)}")    # debug

            solution[agent.id] = path

        return solution

    def _plan_single_agent(self, agent, vertex_constraints: set, edge_constraints: set) -> Optional[list]:
        """
        Plans a path for a single agent using A* with the given constraints.
        Used when splitting a CBS node to replan the constrained agent.
        A new time-expanded graph is built from scratch with the constraints
        of the child node. 
        Args:
            agent:               Agent object to replan
            vertex_constraints:  expanded node ids forbidden for this agent
            edge_constraints:    expanded edge pairs forbidden for this agent
        Returns:
            path as list of expanded node ids, or None if no feasible path exists
        """
        teg = TimeExpandedGraph(self.G_original, self.T, vertex_constraints, edge_constraints)
        start_exp = teg.get_expanded_id(agent.start, t=0)
        return a_star(teg.G_expanded, start_exp, agent.goal, extended=True)

    def _detect_conflict(self, solution: dict, fleet: Fleet) -> Optional[tuple]:
        """
        Finds the first conflict in the current solution.
        Two types of conflicts are detected:
        - Vertex conflict: two agents occupy the same node at the same timestep.
        - Edge conflict (swap): two agents swap positions between t and t+1: agent i moves from vi to vj while agent j moves from vj to vi.

        Agents that have already reached their goal are assumed to stay on their last node for all subsequent timesteps.

        Args:
            solution: dict agent_id -> path (expanded node ids)
            fleet: Fleet object
        Returns:
            ((ai, aj), vertex_constr, edge_constr) if a conflict is found:
                - vertex_constr: forbidden expanded node id (None for edge conflicts)
                - edge_constr:   forbidden (src, dst) edge pair (None for vertex conflicts)
            None if no conflict is found
        """
        agent_ids = list(fleet.agents.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                ai, aj = agent_ids[i], agent_ids[j]

                # skip agents without a planned path (idle agents)
                if ai not in solution or aj not in solution:
                    continue

                path_i = solution[ai]
                path_j = solution[aj]
                max_t  = max(len(path_i), len(path_j))

                for t in range(max_t):
                    # agents that finished stay on their last node
                    vi = path_i[t] if t < len(path_i) else path_i[-1]
                    vj = path_j[t] if t < len(path_j) else path_j[-1]

                    # vertex conflict: same expanded node at the same timestep
                    if vi == vj:
                        return (ai, aj), vi, None

                    # edge conflict: agents swap positions between t and t+1
                    if t + 1 < max_t:
                        vi_next = path_i[t + 1] if t + 1 < len(path_i) else path_i[-1]
                        vj_next = path_j[t + 1] if t + 1 < len(path_j) else path_j[-1]
                        if vi == vj_next and vj == vi_next:
                            return (ai, aj), None, (vi, vj_next)

        return None   # no conflict found

    def _build_plan_result(self, solution: dict, fleet: Fleet) -> PlanResult:
        """
        Converts the CBS solution from expanded node ids to a PlanResult
        containing paths in both expanded and original node ids.
        A new teg is built with the winning node constraints to resolve
        the expanded ids back to original ids.
        Args:
            solution:            dict agent_id -> path (expanded node ids)
            fleet:               Fleet object
            vertex_constraints:  constraints of the winning CBS node
            edge_constraints:    constraints of the winning CBS node
        Returns:
            PlanResult with paths in both expanded and original node ids
        """
        teg = TimeExpandedGraph(self.G_original, self.T)
        result = PlanResult(success=True)

        for agent in fleet.agents.values():
            if agent.id not in solution:
                continue   # idle agent: no path to register        

            path = solution[agent.id]
            path_original = [teg.get_original_id(n) for n in path]
            result.add_path(agent.id, path, path_original)

        return result