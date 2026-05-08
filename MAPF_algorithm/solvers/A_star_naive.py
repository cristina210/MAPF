# MAPF_algorithm/solvers/A_star_naive.py

from Network_graph import NetworkGraph
from fleet import Fleet
from MAPF_algorithm.plan_result import PlanResult
from shortest_path_algorithm.A_star import a_star
import time

class AStarNaive:
    """
    Naive MAPF baseline solver.
    This solver plans each agent independently using A* on the original graph,
    without considering interactions between agents.
    Properties:
    - No collision checking (vertex or edge conflicts are ignored)
    - No coordination between agents
    - Fast but not valid for MAPF in general (can produce conflicting paths)

    It is mainly used as a baseline for comparison with other conflicts free methods
    (e.g., CBS or time-expanded approaches).
    """

    def __init__(self, G_original: NetworkGraph):
        """
        Args:
            G_original: original spatial graph (no time expansion)
        """
        self.G_original = G_original
        self.stats = {"runtime_sec": 0.0, "planned_agents": 0, "failed_agents": 0}

    def plan(self, fleet: Fleet) -> PlanResult:
        """
        Computes a path for each agent independently using A*.

        Each agent is processed sequentially:
        - A* is executed from start to goal on the original graph
        - No constraints from other agents are considered
        If any agent has no valid path, the solver stops immediately
        and returns failure.

        Args:
            fleet: set of agents with start/goal definitions

        Returns:
            PlanResult containing:
            - success flag
            - per-agent paths (if successful)
            - solver statistics
        """
        t0 = time.perf_counter()
        result = PlanResult(success=True)

        for agent in fleet.agents.values():
            if agent.goal is None:
                continue

            path = a_star(self.G_original, agent.start, agent.goal, extended=False)

            if path is None:
                self.stats["failed_agents"] += 1
                self.stats["runtime_sec"] = time.perf_counter() - t0
                r = PlanResult(success=False, failed_agent=agent.id)
                r.solver_stats = self.stats.copy()
                return r

            self.stats["planned_agents"] += 1
            # path_original = path itself (no expanded ids here)
            result.add_path(agent.id, path, path)

        self.stats["runtime_sec"] = time.perf_counter() - t0
        result.solver_stats = self.stats.copy()
        return result