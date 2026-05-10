import heapq
from typing import Optional
from Network_graph import NetworkGraph
from fleet import Fleet
from extended_time_graph import TimeExpandedGraph
from shortest_path_algorithm.A_star import a_star_with_focal_search, a_star
from MAPF_algorithm.plan_result import PlanResult
import time
from MAPF_algorithm.heuristicsBCBS import HEURISTICS_HIGH_L
from MAPF_algorithm.heuristicsBCBS import HEURISTICS_LOW_L

class BCBSNode:
    """
    Represents a node in the BCBS constraint tree (CT).

    Each node stores:
    - per-agent vertex and edge constraints accumulated from the root down to this node
    - a solution: one path (in expanded node ids) per active agent
    - the sum-of-costs of the solution
    - g_c: conflict heuristic used by the high-level focal search to prefer
      nodes whose solutions have fewer inter-agent conflicts

    g_c is initialized to 0 and must be set explicitly after solution is
    assigned (see BCBSSolver.plan), because solution is empty at __init__ time.

    The tree is explored via focal search: OPEN ordered by cost, FOCAL ordered by g_c.
    """

    def __init__(self, vertex_constraints=None, edge_constraints=None, solution=None, parent=None, cost_for_each_agent=None):
        """
        Args:
            vertex_constraints: dict {agent_id: set of forbidden expanded node ids}
            edge_constraints:   dict {agent_id: set of forbidden (src, dst) expanded edge pairs}
            solution:           dict {agent_id: list of expanded node ids} — inizialized empty at creation
            parent:             parent BCBSNode in the constraint tree, None for root
        """

        self.vertex_constraints = vertex_constraints if vertex_constraints is not None else {}
        self.edge_constraints = edge_constraints if edge_constraints is not None else {}
        self.solution = solution if solution is not None else {}
        self.parent = parent
        self.cost_for_each_agent = cost_for_each_agent if cost_for_each_agent is not None else {}
        self.cost = self._compute_cost()
        self.depth = 0 if parent is None else parent.depth + 1
        self.g_c = 0    # to update after creation (depends on solution)


    def _compute_cost(self) -> float:
        """
        Sum-of-costs: sum of costs of paths of single agent
        """
        return sum(self.cost_for_each_agent.values())

    def __lt__(self, other: "BCBSNode") -> bool:
        # required by heapq when two nodes have equal cost in a tuple comparison
        return self.cost < other.cost
    
    def compute_how_many_fixed_agent_in_each_node(self, id_agent = None):
        """
        Counts how many agents pass through each expanded node in the current solution.
        Used both to build the congestion_dict for the low-level planner and to
        compute g_c for the high-level focal search.
        Indeed, it is a metrics that helps in computing approximatly the number of vertex constraints violated

        Args:
            id_agent: if provided, excludes this agent from the count.
                      Used when replanning agent i: counts congestion from all other agents (used for focal search in the low level).
                      If None, counts all agents (used for g_c computation).
        Returns:
            dict {expanded_node_id: number of agents passing through it}
        """
        dict_number_agent_in_node = {}
        for agent_id, path in self.solution.items():  # self.solution, non solution
            if (id_agent is not None) and  (agent_id == id_agent):
                continue
            for node in path:
                dict_number_agent_in_node[node] = dict_number_agent_in_node.get(node, 0) + 1
        return dict_number_agent_in_node
    
    def update_constr_in_node_for_agent(self, vertex_constr, edge_constr, agent_id):
        """
        Initialises this node's constraints by copying the parent's constraints
        and adding one new constraint for the specified agent.
        Called immediately after child node creation in high level CBS.
        Note: edge constraints here are expressed as expanded node id pairs (src, dst).
        The TEG builder handles swap-pair symmetry automatically when the TEG is constructed.

        Args:
            vertex_constr: expanded node id to forbid for agent_id, or None
            edge_constr:   (src, dst) expanded edge pair to forbid for agent_id, or None
            agent_id:      agent to whom the new constraint applies
        """
        # copy parent constraints and add the new one for this child
        new_vertex_constr_diz = {k: set(v) for k, v in self.parent.vertex_constraints.items()}
        new_edge_constr_diz = {k: set(v) for k, v in self.parent.edge_constraints.items()}

        # initialise sets for this agent if not already present
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

class BCBSSolver:
    """
    Bounded Suboptimal CBS (BCBS) solver using focal search at both levels.

    Based on: Sharon et al., "Conflict-Based Search for Optimal Multi-Agent Path Finding"
    and the BCBS extension described in the same literature.

    Two-level structure:
    HIGH LEVEL — constraint tree (CT) with focal search:
      OPEN:  all CT nodes, ordered by sum-of-costs.
      FOCAL: subset of OPEN with cost <= w_h * cost_min,
             ordered by g_c (conflict heuristic).
      The node with lowest g_c in focal is expanded, favouring solutions with
      fewer conflicts and expected to converge faster.

    LOW LEVEL — single-agent planning with focal A*:
      For each agent, builds a TEG with that agent's constraints and
      runs a_star_with_focal_search with factor w_l, guided by a
      congestion_dict built from the other agents' paths.

    Parameters:
      w_l: low-level suboptimality factor  (w_l=1 -> optimal low-level)
      w_h: high-level suboptimality factor (w_h=1 -> optimal high-level)

    Special cases:
      BCBS(1, 1)   — equivalent to CBS in solution cost
      BCBS(w, w)   — BCBS(w_H, w_L) with equal factors
      BCBS(inf, inf) — greedy CBS (GCBS-LH), fastest but no cost guarantee

    Guarantees:
      - Conflict-free: yes
      - Complete:      yes
      - Cost bound:    sum-of-costs <= w_h * w_l * C*
    """

    def __init__(self, G_original: NetworkGraph, T: int, w_l = 1, w_h = 1, conflict_heuristic_high_l="h3", conflict_heuristic_low_l = "h3"):
        """
        Args:
            G_original: original spatial graph (not time-expanded)
            T: time horizon = length of the time-expanded graph
            w_l: low-level suboptimality factor  (default 1: optimal)
            w_h: high-level suboptimality factor (default 1: optimal)
            conflict_heuristic: heuristic function for approximate the goodness of the solution in terms of feasibility (no conflicts)
        """
        self.G_original = G_original
        self.T = T
        self.stats = {"expanded_nodes": 0, "generated_nodes": 0, "max_tree_depth": 0, "runtime_sec": 0.0, "max_heap_size": 0}
        self.w_l = w_l
        self.w_h = w_h
        self.conflict_heuristic_L_level = conflict_heuristic_low_l   
        self.conflict_heuristic_H_level = conflict_heuristic_high_l 

    def plan(self, fleet: Fleet) -> PlanResult:
        """
        Finds a conflict-free bounded-suboptimal plan for all agents using BCBS.

        Args:
            fleet: Fleet object with start and goal for each agent
        Returns:
            PlanResult with planned paths and solver stats,
            or PlanResult(success=False) if no solution exists
        """
        # root node: no constraints, each agent gets its individual shortest path
        # self.iter = 0 # debug
        t0 = time.perf_counter()    # stat
        
        # root node: no constraints, each agent gets its individual shortest path
        root_solution, diz_costs_for_agent = self._plan_all_agents(fleet)
        if root_solution is None:
            # at least one agent has no feasible path even without constraints
            return PlanResult(success=False)

        root = BCBSNode(solution=root_solution, cost_for_each_agent = diz_costs_for_agent)
        # root.g_c stays 0 — no conflict information at the root

        # OPEN list: all unexpanded CT nodes
        # implemented as a list; focal is rebuilt each iteration (O(|OPEN|)  [si potrebbe ottimizzare]
        list_open = [root]

        while list_open:

            # f_min: lowest cost among all nodes in OPEN
            # used to define the focal threshold
            f_min = min(node.cost for node in list_open)

            # build FOCAL (list with nodes in increasing order of g_c)
            focal = self._get_focal_list(list_open, f_min)

            self.stats["max_heap_size"] = max(self.stats["max_heap_size"], len(list_open))
            self.stats["expanded_nodes"] += 1

            # expand the node with lowest g_c from FOCAL
            node = focal.pop(0)
            list_open.remove(node)

            # detect the first conflict in the current solution
            conflict = self._detect_conflict(node.solution, fleet) 

            if conflict is None:
                # no conflicts: solution is valid and within the cost bound
                result = self._build_plan_result(node.solution, fleet)

                self.stats["runtime_sec"] = time.perf_counter() - t0    # stat
                result.solver_stats = self.stats.copy()
                return result

            agent_tuple, vertex_constr, edge_constr = conflict

            # split: one child per agent involved in the conflict
            # each child inherits parent constraints plus one new constraint
            for agent_id in agent_tuple:

                # create child node
                child = BCBSNode(parent=node)
                child.update_constr_in_node_for_agent(vertex_constr, edge_constr, agent_id)    
                
                # congestion_dict: how many other agents pass through each expanded node (apart from agent_id)
                # used by the low-level focal search to prefer less congested paths
                congestion_dict = node.compute_how_many_fixed_agent_in_each_node(agent_id)
                new_path, cost_sol = self._plan_single_agent(fleet.get_agent(agent_id), vertex_constraints=child.vertex_constraints[agent_id], edge_constraints=child.edge_constraints[agent_id], congestion_dict=congestion_dict)

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

                # generate cost (classic and heuristic) of solution in child node
                child.g_c = HEURISTICS_HIGH_L[self.conflict_heuristic_H_level](child.solution, self.T)


                self.stats["generated_nodes"] += 1       # stat
                self.stats["max_tree_depth"] = max(self.stats["max_tree_depth"],child.depth)        # stat

                list_open.append(child)

        # OPEN exhausted: no conflict-free solution exists within the constraints
        result = PlanResult(success=False)
        self.stats["runtime_sec"] = time.perf_counter() - t0     # stat
        result.solver_stats = self.stats.copy()

        return result

    def _get_focal_list(self, listt: list, f_min:float) -> list:
        """
        Builds the high-level FOCAL list from OPEN.

        Selects all CT nodes with cost <= w_h * f_min (within the suboptimality bound),
        then sorts them by (g_c, cost) so the least conflicting node is expanded first.

        Args:
            listt:  current OPEN list of BCBSNodes
            f_min:  minimum cost among all nodes in OPEN
        Returns:
            sorted list of BCBSNodes in FOCAL, ordered by (g_c, cost)
        """
        threshold = f_min * self.w_h
        focal_list = []
        for CBSnode in listt:
            if CBSnode.cost <= threshold:
                focal_list.append(CBSnode)
        focal_list.sort(key=lambda x: (x.g_c, x.cost))
        return focal_list



    def _plan_all_agents(self, fleet: Fleet) -> Optional[dict]:
        """
        Plans a path for every active agent independently using standard A*.
        Used to initialise the root CT node (no constraints).

        A single unconstrained TEG is built and reused for all agents.

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



    def _plan_single_agent(self, agent, vertex_constraints: set, edge_constraints: set, congestion_dict: dict) -> Optional[list]:
        """
        Replans a single agent using focal A* with the given constraints.
        Called when splitting a CT node to resolve a conflict.
        A new TEG is built from scratch with the agent's constraints.
        Swap-pair inverses are handled automatically by TimeExpandedGraph.

        Args:
            agent:               Agent object to replan
            vertex_constraints:  set of forbidden expanded node ids for this agent
            edge_constraints:    set of forbidden (src, dst) expanded edge pairs for this agent
            congestion_dict:     {expanded_node_id: count} of other agents — guides focal search
        Returns:
            path as list of expanded node ids, or None if no feasible path exists
        """
        teg = TimeExpandedGraph(self.G_original, self.T, vertex_constraints, edge_constraints)
        start_exp = teg.get_expanded_id(agent.start, t=0)
        path = a_star_with_focal_search(teg.G_expanded, start_exp, agent.goal, congestion_dict = congestion_dict, conflict_heuristic = self.conflict_heuristic_L_level, extended=True, w = self.w_l )
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
        Only timesteps where both agents are still active are checked
        (agents that have reached their goal are not considered).

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
                    continue     # skip idle agents

                path_i = solution[ai]
                path_j = solution[aj]

                for t in range(min(len(path_i), len(path_j)) - 1):
                    # convert expanded ids to original ids for position comparison
                    or_i_t  = TimeExpandedGraph.compute_original_id(path_i[t], self.T)
                    or_j_t  = TimeExpandedGraph.compute_original_id(path_j[t], self.T)
                    # next positions in original graph
                    or_i_t1 = TimeExpandedGraph.compute_original_id(path_i[t + 1], self.T)
                    or_j_t1 = TimeExpandedGraph.compute_original_id(path_j[t + 1], self.T)

                    # vertex conflict at timestep t
                    if or_i_t == or_j_t:
                        return (ai, aj), path_i[t], None

                    # edge conflict: i goes u->v while j goes v->u
                    # constraint returned is the arc traversed by agent i
                    # the inverse arc (for agent j) is added via swap_pairs in TimeExpandedGraph
                    if or_i_t == or_j_t1 and or_j_t == or_i_t1:
                        return (ai, aj), None, (path_i[t], path_i[t + 1])

                # check vertex conflict at the last common timestep
                last_t = min(len(path_i), len(path_j)) - 1
                or_i = TimeExpandedGraph.compute_original_id(path_i[last_t], self.T)
                or_j = TimeExpandedGraph.compute_original_id(path_j[last_t], self.T)
                if or_i == or_j:
                    return (ai, aj), path_i[last_t], None

        return None   # no conflict found



    def _build_plan_result(self, solution: dict, fleet: Fleet) -> PlanResult:
        """
        Converts the winning CT node solution from expanded node ids to a PlanResult.

        Uses the static conversion method compute_original_id — no TEG construction needed,
        as expanded ids are deterministic given the formula: expanded_id = original_id * T + t.

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