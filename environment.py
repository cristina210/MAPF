from typing import Optional
from instance import MAPFInstance
from MAPF_algorithm.plan_result import PlanResult
from extended_time_graph import TimeExpandedGraph


class MAPFEnvironment:
    """
    Simulation environment MAPF.
    Get in input an instance of the problem and a plan and execute the
    simulation step by step.
    """

    def __init__(self, instance: MAPFInstance):
        """
        Args:
            instance: MAPF instances
        """
        self.instance = instance

        # agent_id -> original_node_id
        self.current_positions = {}        # dynamic state: current position of each agent in the (original) graph

        self.timestep = 0      # time of simulation
        self.done     = False     # end of the simulation if TRUE
        self.history  = []   # lista di snapshot dello stato ad ogni timestep

        self._reset()

    def _reset(self) -> None:
        """
        Reset the environment
        """
        for agent in self.instance.fleet.agents.values():
            # initialize agent positions on the start nodes
            self.current_positions[agent.id] = agent.start

        self.timestep = 0     # inizialize time of simulation
        self.done     = False
        self.history  = [self._snapshot()]   # first snapshot

    def step(self, actions: dict) -> dict:
        """
        Execute a single time step in the simulation

        Args:
            actions: dict agent_id -> next_original_node_id
        Returns:
            snapshot of the state after the step
        """
        assert not self.done, "End of the simulation"

        # update position of each agent
        for agent_id, next_node in actions.items():
            self.current_positions[agent_id] = next_node

        self.timestep += 1          # update time step
        self.done = self._all_agents_at_goal()          # check if all agent arrive to the final goal

        snapshot = self._snapshot()      # snapshot: shot of the current situation
        self.history.append(snapshot)
        return snapshot

    def run_plan(self, plan_result: PlanResult) -> list:
        """
        Execute the plan and return the list of visited states.

        Args:
            plan_result: plan path from solvers
            teg:         time-expanded graph 
        Returns:
            lista di snapshot, uno per ogni timestep (incluso t=0)
        """
        assert plan_result.success, "Impossibile eseguire un piano fallito"

        self._reset()

        max_steps = max(len(p) for p in plan_result.paths_original.values())

        for t in range(1, max_steps):
            actions = {}
            for agent_id, path in plan_result.paths_original.items():
                next_node = path[t] if t < len(path) else path[-1]
                actions[agent_id] = next_node
            self.step(actions)

        return self.history

    def _all_agents_at_goal(self) -> bool:
        """
        True if all agents reach their goal
        """
        for agent in self.instance.fleet.agents.values():
            if agent.goal is None:
                continue   
            if self.current_positions[agent.id] != agent.goal:
                return False
        return True

    def _snapshot(self) -> dict:
        """
        Return a snapshot of the current state: timestep, position of each agent, flag if simulation is finished.
        """
        return {
            "timestep":  self.timestep,
            "positions": dict(self.current_positions),   
            "done":      self.done,
        }

    def __repr__(self) -> str:
        return (f"MAPFEnvironment("
                f"timestep={self.timestep}, "
                f"done={self.done}, "
                f"positions={self.current_positions})")