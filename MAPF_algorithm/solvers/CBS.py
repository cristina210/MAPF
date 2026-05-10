import heapq
from typing import Optional
from Network_graph import NetworkGraph
from fleet import Fleet
from extended_time_graph import TimeExpandedGraph
from shortest_path_algorithm.A_star import a_star
from MAPF_algorithm.plan_result import PlanResult
import time


class CBSNode:
    """
    Represents a node in the CBS constraint tree (CT).
    Each node stores:
    - per-agent vertex and edge constraints accumulated from the root down to this node
    - a solution: one path (in expanded node ids) per active agent
    - the costs of the solution (sum of costs of each solution path of each agent)
    - depth in the constraint tree 

    Constraints are stored as dicts {agent_id: set} so each agent has its own
    independent constraint set. 
    The tree is explored in best-first order by costs using a min-heap.
    The first conflict-free node extracted is guaranteed to be optimal.
    """

    def __init__(self, vertex_constraints=None, edge_constraints=None, solution=None, parent=None, cost_for_each_agent=None):
            """
            Args:
                vertex_constraints: dict {agent_id: set of forbidden expanded node ids}
                edge_constraints:   dict {agent_id: set of forbidden (src, dst) expanded edge pairs}
                solution:           dict {agent_id: list of expanded node ids} — empty at creation
                parent:             parent CBSNode in the constraint tree, None for root
            """
            self.vertex_constraints = vertex_constraints if vertex_constraints is not None else {}
            self.edge_constraints = edge_constraints if edge_constraints is not None else {}
            self.solution = solution if solution is not None else {}
            self.parent = parent
            self.cost_for_each_agent = cost_for_each_agent if cost_for_each_agent is not None else {}
            self.cost = self._compute_cost() 
            self.depth = 0 if parent is None else parent.depth + 1


    def _compute_cost(self) -> float:
        """
        Sum-of-costs: sum of costs of paths of single agent
        """
        return sum(self.cost_for_each_agent.values())

    def __lt__(self, other: "CBSNode") -> bool:
        # required by heapq when two nodes have equal cost in a tuple comparison
        return self.cost < other.cost

    def update_constr_in_node_for_agent(self, vertex_constr, edge_constr, agent_id):
        """
        Initialises this node's constraints by copying the parent's constraints
        and adding one new constraint for the specified agent.
        Called immediately after child node creation.
        Note: edge constraints are expressed as expanded node id pairs (src, dst).
        Swap-pair symmetry (preventing swaps) is handled automatically by
        TimeExpandedGraph when the TEG is constructed for replanning.

        Args:
            vertex_constr: expanded node id to forbid for agent_id, or None
            edge_constr:   (src, dst) expanded edge pair to forbid for agent_id, or None
            agent_id:      agent to whom the new constraint applies
        """

        # copy parent constraints and add the new one for this child
        new_vertex_constr_diz = {k: set(v) for k, v in self.parent.vertex_constraints.items()}
        new_edge_constr_diz = {k: set(v) for k, v in self.parent.edge_constraints.items()}

        # initialise constraint sets for this agent if not already present
        if agent_id not in new_vertex_constr_diz:
            new_vertex_constr_diz[agent_id] = set()
        if agent_id not in new_edge_constr_diz:
            new_edge_constr_diz[agent_id] = set()

        if vertex_constr is not None:
            new_vertex_constr_diz[agent_id].add(vertex_constr)

        if edge_constr is not None:
            new_edge_constr_diz[agent_id].add(edge_constr)
        self.vertex_constraints = new_vertex_constr_diz
        self.edge_constraints = new_edge_constr_diz


class CBSSolver:
    """
    Conflict-Based Search (CBS) MAPF solver.

    Two-level structure:
    HIGH LEVEL — constraint tree (CT) with best-first search:
      Explores CT nodes ordered by sum-of-costs using a min-heap.
      When a conflict is detected in the current solution, the node is
      split into two children — one adding a constraint for each of the
      two conflicting agents. The first conflict-free node extracted is
      guaranteed to be optimal.
    LOW LEVEL — single-agent A* planning:
      For each agent, builds a TEG with that agent's constraints and
      runs standard A* to find the shortest feasible path.

    Guarantees:
      - Conflict-free: yes
      - Complete:      yes
      - Optimal:       yes (sum-of-costs)
    """

    def __init__(self, G_original: NetworkGraph, T: int):
        """
        Args:
            G_original: original spatial graph (not time-expanded)
            T: time horizon = length of the time-expanded graph
        """
        self.G_original = G_original
        self.T = T
        self.stats = {"expanded_nodes": 0, "generated_nodes": 0, "max_tree_depth": 0, "runtime_sec": 0.0, "max_heap_size": 0}



    def plan(self, fleet: Fleet) -> PlanResult:
        """
        Finds a conflict-free optimal plan for all agents using CBS.

        Args:
            fleet: Fleet object with start and goal for each agent
        Returns:
            PlanResult with planned paths and solver stats or PlanResult(success=False) if no solution exists
        """
        
        # root node: no constraints, each agent gets its individual shortest path
        t0 = time.perf_counter()    # stat
        
        root_solution, diz_costs_for_agent = self._plan_all_agents(fleet)
        if root_solution is None:
            # at least one agent has no path even without constraints
            return PlanResult(success=False)

        root = CBSNode(solution=root_solution, cost_for_each_agent = diz_costs_for_agent)

        # min-heap ordered by (cost, node)
        # counter breaks ties deterministically and prevents direct CBSNode comparison
        heap = [root]

        while heap:

            self.stats["max_heap_size"] = max(self.stats["max_heap_size"], len(heap))
            node = heapq.heappop(heap)
            self.stats["expanded_nodes"] += 1
            
            # detect the first conflict in the current solution
            conflict = self._detect_conflict(node.solution, fleet)

            if conflict is None:
                # no conflicts: solution is valid and optimal (best-first guarantees this)
                result = self._build_plan_result(node.solution, fleet)

                self.stats["runtime_sec"] = time.perf_counter() - t0    # stat
                result.solver_stats = self.stats.copy()
                return result

            agent_tuple, vertex_constr, edge_constr = conflict

            # split: one child per agent involved in the conflict
            # each child inherits parent constraints plus one new constraint
            for agent_id in agent_tuple:
                # create child node
                child = CBSNode(parent=node)
                # add constraints in CBS node
                child.update_constr_in_node_for_agent(vertex_constr, edge_constr, agent_id)

                new_path, cost_sol = self._plan_single_agent(fleet.get_agent(agent_id), vertex_constraints=child.vertex_constraints[agent_id], edge_constraints=child.edge_constraints[agent_id])

                # if no feasible path exists under the new constraints, skip this child
                if new_path is None:
                    continue

                # update child solution: copy parent solution and replace replanned agent
                new_solution = dict(node.solution)
                new_solution[agent_id] = new_path

                # update costs 
                new_costs_diz = dict(node.cost_for_each_agent)
                new_costs_diz[agent_id] = cost_sol

                child.solution = new_solution
                child.cost_for_each_agent = new_costs_diz

                # recompute cost
                child.cost = child._compute_cost()

                self.stats["generated_nodes"] += 1       # stat
                self.stats["max_tree_depth"] = max(self.stats["max_tree_depth"],child.depth)        # stat

                heapq.heappush(heap, child)


        # heap exhausted: no conflict-free solution exists
        result = PlanResult(success=False)
        self.stats["runtime_sec"] = time.perf_counter() - t0     # stat
        result.solver_stats = self.stats.copy()

        return result

    def _plan_all_agents(self, fleet: Fleet) -> Optional[dict]:
        """
        Plans a path for every active agent independently using standard A*.
        Used to initialise the root CT node (no constraints).
        A single unconstrained TEG is built and reused for all agents.
        Idle agents (goal=None) are skipped.

        Args:
            fleet: Fleet object
        Returns:
            dict {agent_id: path (expanded node ids)},
            or None if any agent has no feasible path
        """
        teg = TimeExpandedGraph(self.G_original, self.T)
        solution = {}
        diz_costs_for_agent = {}

        for agent in fleet.agents.values():
            if agent.goal is None:
                continue   # idle agents are not planned

            start_exp = teg.get_expanded_id(agent.start, t=0)
            path = a_star(teg.G_expanded, start_exp, agent.goal, extended=True)

            
            if path is None:
                return None, None  # no feasible path for this agent

            # computing costs
            cost_sol = 0
            for u, v in zip(path[:-1], path[1:]):
                cost_sol += teg.G_expanded[u][v]["weight"]  # sum of weights


            solution[agent.id] = path
            diz_costs_for_agent[agent.id] = cost_sol

        return solution, diz_costs_for_agent


    def _plan_single_agent(self, agent, vertex_constraints: set, edge_constraints: set) -> Optional[list]:
        """
        Replans a single agent using A* with the given constraints.
        Called when splitting a CT node to resolve a conflict.
        A new TEG is built from scratch with the agent's constraints.
        Swap-pair inverses are handled automatically by TimeExpandedGraph:
        forbidding edge (u_t, v_t+1) also forbids (v_t, u_t+1).
        Args:
            agent:  Agent object to replan
            vertex_constraints:  set of forbidden expanded node ids for this agent
            edge_constraints:    set of forbidden (src, dst) expanded edge pairs for this agent
        Returns:
            path as list of expanded node ids, or None if no feasible path exists
        """
        teg = TimeExpandedGraph(self.G_original, self.T, vertex_constraints, edge_constraints)
        start_exp = teg.get_expanded_id(agent.start, t=0)
        path = a_star(teg.G_expanded, start_exp, agent.goal, extended=True)
        if path is None:
            return None, None

        cost_sol = 0
        for u, v in zip(path[:-1], path[1:]):
            cost_sol += teg.G_expanded[u][v]["weight"]

        return path, cost_sol
    
    
    def _detect_conflict(self, solution: dict, fleet: Fleet) -> Optional[tuple]:
        """
        Finds the first conflict in the current solution.
        Two conflict types are detected:
        - Vertex conflict: two agents at the same original node at the same timestep.
        - Edge conflict (swap): agent i moves u->v while agent j moves v->u
          in the same timestep interval [t, t+1].

        Only timesteps where both agents are still active are checked.
        Positions are compared in the original graph (via compute_original_id)
        to correctly handle the expanded id representation.

        Args:
            solution: dict {agent_id: path (expanded node ids)}
            fleet:    Fleet object
        Returns:
            ((ai, aj), vertex_constr, edge_constr) where:
                vertex_constr: expanded node id to forbid (None for edge conflicts)
                edge_constr:   (src_exp, dst_exp) arc of agent i (None for vertex conflicts)
            None if no conflict found
        """
        agent_ids = list(fleet.agents.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                ai, aj = agent_ids[i], agent_ids[j]

                if ai not in solution or aj not in solution:
                    continue

                path_i = solution[ai]
                path_j = solution[aj]

                for t in range(min(len(path_i), len(path_j)) - 1):
                    # current positions in original graph
                    or_i_t  = TimeExpandedGraph.compute_original_id(path_i[t], self.T)
                    or_j_t  = TimeExpandedGraph.compute_original_id(path_j[t], self.T)
                    # next positions in original graph
                    or_i_t1 = TimeExpandedGraph.compute_original_id(path_i[t + 1], self.T)
                    or_j_t1 = TimeExpandedGraph.compute_original_id(path_j[t + 1], self.T)

                    # vertex conflict at t
                    if or_i_t == or_j_t:
                        return (ai, aj), path_i[t], None

                    # edge conflict: swap between t and t+1
                    if or_i_t == or_j_t1 and or_j_t == or_i_t1:
                        return (ai, aj), None, (path_i[t], path_i[t + 1])

                # check vertex conflict at the last common timestep
                last_t = min(len(path_i), len(path_j)) - 1
                or_i = TimeExpandedGraph.compute_original_id(path_i[last_t], self.T)
                or_j = TimeExpandedGraph.compute_original_id(path_j[last_t], self.T)
                if or_i == or_j:
                    return (ai, aj), path_i[last_t], None

        return None


    def _build_plan_result(self, solution: dict, fleet: Fleet) -> PlanResult:
        """
        Converts the winning CT node solution from expanded node ids to a PlanResult.

        Args:
            solution: dict {agent_id: path (expanded node ids)}
            fleet:    Fleet object
        Returns:
            PlanResult with paths in both expanded and original node ids
        """
        result = PlanResult(success=True, T=self.T)
        for agent in fleet.agents.values():
            if agent.id not in solution:
                continue

            path = solution[agent.id]

            path_original = [TimeExpandedGraph.compute_original_id(n, self.T) for n in path]

            result.add_path(agent.id, path, path_original)

        return result