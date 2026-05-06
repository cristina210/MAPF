from typing import Optional


class PlanResult:
    """
    Contain the result of MAPF implementation.
    Memorize paths found for each agents and the success of the planning
    """

    def __init__(self, success: bool, failed_agent: Optional[int] = None):
        """
        Args:
            success:      True if the planning succeded for all the agent
            failed_agent: id of the agent for which the planning didn't succeded
        """
        self.success      = success
        self.failed_agent = failed_agent
        self.paths          = {}   # for each agent_id -> lists of nodes in the expanded nel time-expanded graph
        self.paths_original = {}   # # for each agent_id -> lists of nodes in the graph

    def add_path(self, agent_id: int, path: list, path_original: list) -> None:
        """
        Register the planned route for each agent

        Args:
            agent_id: agent id
            path:     list of nodes in the expanded nel time-expanded graph
        """
        self.paths[agent_id] = path
        self.paths_original[agent_id] = path_original

    def path_for(self, agent_id: int) -> Optional[list]:
        """
        Return the MAPF path for the specified agent, None if doesn't exist

        Args:
            agent_id: agent id 
        Returns:
            list expanded node ids, oppure None
        """
        return self.paths.get(agent_id)

    def path_original_for(self, agent_id: int) -> Optional[list]:
        """
        Return the MAPF path for the specified agent, None if doesn't exist

        Args:
            agent_id: agent id 
        Returns:
            list node ids, oppure None
        """
        return self.paths_original.get(agent_id)

    def __repr__(self) -> str:
        if not self.success:
            return f"PlanResult(success=False, failed_agent={self.failed_agent})"
        summary = {aid: len(p) for aid, p in self.paths.items()}
        return f"PlanResult(success=True, path_lengths={summary})"
