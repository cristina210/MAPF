from typing import Optional
from Network_graph import NetworkGraph
from extended_time_graph import TimeExpandedGraph
# plan_result.py

from typing import Optional
from Network_graph import NetworkGraph


class PlanResult:
    """
    Contains the result of a MAPF planning session.
    Stores paths for each agent in both expanded and original node ids,
    and provides metrics on the solution quality.
    """

    def __init__(self, success: bool, T: Optional[int] = None, failed_agent: Optional[int] = None):
        """
        Args:
            success:      True if planning succeeded for all agents
            failed_agent: id of the agent for which planning failed, None if success
        """
        self.success        = success
        self.failed_agent   = failed_agent
        self.paths          = {}   # agent_id -> list of expanded node ids
        self.paths_original = {}   # agent_id -> list of original node ids
        self.solver_stats   = {}   # statistics from the solver
        self.T = T


    def add_path(self, agent_id: int, path: list, path_original: list) -> None:
        """
        Registers the planned path for an agent in both representations.

        Args:
            agent_id:      agent id
            path:          list of expanded node ids (time-expanded graph)
            path_original: list of original node ids
        """
        self.paths[agent_id]          = path
        self.paths_original[agent_id] = path_original

    def path_for(self, agent_id: int) -> Optional[list]:
        """Returns the expanded path for the agent, None if not present."""
        return self.paths.get(agent_id)

    def path_original_for(self, agent_id: int) -> Optional[list]:
        """Returns the original-graph path for the agent, None if not present."""
        return self.paths_original.get(agent_id)

    # ── Time metrics ───────────────────────────────────────────────────────────

    def makespan(self) -> int:
        """
        Makespan: timestep at which the last agent reaches its goal.
        Counts wait steps as time cost.
        """
        if not self.paths_original:
            return 0
        return max(len(p) - 1 for p in self.paths_original.values())

    def path_length_time(self, agent_id: int) -> Optional[int]:
        """
        Number of timesteps used by a single agent (including waits).

        Args:
            agent_id: agent id
        Returns:
            number of timesteps, or None if no path exists
        """
        p = self.paths_original.get(agent_id)
        return len(p) - 1 if p is not None else None

    def cumulative_time_total(self) -> int:
        """Sum of timesteps across all agents (time-based sum-of-costs)."""
        return sum(len(p) - 1 for p in self.paths_original.values())

    # ── Physical distance metrics (original graph only, no waits) ─────────────

    def path_length_space(self, agent_id: int,
                          G_original: NetworkGraph) -> Optional[float]:
        """
        Physical distance of an agent's path: sum of edge weights on the
        original graph, excluding wait steps (consecutive equal nodes).

        Args:
            agent_id:  agent id
            G_original: original NetworkGraph
        Returns:
            physical distance, or None if no path exists
        """
        p = self.paths_original.get(agent_id)
        if p is None:
            return None
        total = 0.0
        for u, v in zip(p[:-1], p[1:]):
            if u != v:   # skip waits — no edge in original graph
                total += G_original[u][v]["weight"]
        return total

    def total_path_length_space(self, G_original: NetworkGraph) -> float:
        """Sum of physical distances across all agents."""
        total = 0.0
        for agent_id in self.paths_original:
            d = self.path_length_space(agent_id, G_original)
            if d is not None:
                total += d
        return total

    # ── Cost metrics (edge weights on expanded or original graph) ──────────────

    def compute_path_cost(self, agent_id: int, G_original) -> Optional[float]:
        """
        Travel cost of an agent's path as sum of edge weights on the
        time-expanded graph, including wait edges.
        If an expanded path exists in self.paths, it is used directly.
        If not (e.g. naive A* that only stores paths_original), the expanded
        path is reconstructed using compute_expanded_id(original_id, t, T),
        where t is the index in paths_original (each position = one timestep).

        Args:
            agent_id: agent id
            G_expanded: unconstrained time-expanded NetworkGraph
            T: time horizon used to build G_expanded
        Returns:
            total cost, or None if no path exists for this agent
        """
        from extended_time_graph import TimeExpandedGraph

        if agent_id not in self.paths_original or not self.paths_original[agent_id]:
            return None

        cost = 0.0
        if self.T == None:    # A_naive
            path_orig = self.paths_original[agent_id]
            for u, v in zip(path_orig[:-1], path_orig[1:]):
                cost += G_original[u][v]["weight"]
            return cost

        path_orig = self.paths_original[agent_id]
        path_exp  = [ TimeExpandedGraph.compute_expanded_id(node, t, self.T) for t, node in enumerate(path_orig) ]

        teg = TimeExpandedGraph(G_original, self.T)
        G_expanded = teg.G_expanded
        for u, v in zip(path_exp[:-1], path_exp[1:]):
            cost += G_expanded[u][v]["weight"]
        return cost

    def total_cost(self, G_original) -> float:
        """
        Sum-of-costs across all agents using compute_path_cost.

        Args:
            G_expanded: unconstrained time-expanded NetworkGraph
            T:          time horizon used to build G_expanded
        Returns:
            total sum-of-costs
        """
        total = 0.0
        for agent_id in self.paths_original:
            cost = self.compute_path_cost(agent_id, G_original)
            if cost is not None:
                total += cost
        return total



    def __repr__(self) -> str:
        if not self.success:
            return f"PlanResult(success=False, failed_agent={self.failed_agent})"
        summary = {aid: len(p) for aid, p in self.paths_original.items()}
        return f"PlanResult(success=True, path_lengths={summary})"