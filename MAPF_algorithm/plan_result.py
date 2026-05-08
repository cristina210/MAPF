from typing import Optional


class PlanResult:
    '''
    Contain the result of MAPF implementation.
    Memorize paths found for each agents and the success of the planning
    '''

    def __init__(self, success: bool, failed_agent: Optional[int] = None):
        '''
        Args:
            success:      True if the planning succeded for all the agent
            failed_agent: id of the agent for which the planning didn't succeded
        '''
        self.success      = success
        self.failed_agent = failed_agent
        self.paths          = {}   # for each agent_id -> lists of nodes in the expanded nel time-expanded graph
        self.paths_original = {}   # # for each agent_id -> lists of nodes in the graph
        self.solver_stats = {}   # statistics regarding finding the solution by solver

    def add_path(self, agent_id: int, path: list, path_original: list) -> None:
        '''
        Register the planned route for each agent

        Args:
            agent_id: agent id
            path:     list of nodes in the expanded nel time-expanded graph
        '''
        self.paths[agent_id] = path
        self.paths_original[agent_id] = path_original

    def path_for(self, agent_id: int) -> Optional[list]:
        '''
        Return the MAPF path for the specified agent, None if doesn't exist

        Args:
            agent_id: agent id 
        Returns:
            list expanded node ids, oppure None
        '''
        return self.paths.get(agent_id)

    def path_original_for(self, agent_id: int) -> Optional[list]:
        '''
        Return the MAPF path for the specified agent, None if doesn't exist

        Args:
            agent_id: agent id 
        Returns:
            list node ids, oppure None
        '''
        return self.paths_original.get(agent_id)

    def makespan(self) -> int:
        '''Makespan: timestep in which the last agent arrives at the goal node'''
        if not self.paths_original:
            return 0
        return max(len(p) - 1 for p in self.paths_original.values())

    def path_length_time(self, agent_id: int) -> Optional[int]:
        '''Number of timesteps by agent per l'agente'''
        p = self.paths_original.get(agent_id)
        return len(p) - 1 if p is not None else None

    def cumulative_time_total(self) -> int:
        '''Sum of timesteps used by agent in their paths.'''
        return sum(len(p) - 1 for p in self.paths_original.values())

    def path_length_space(self, agent_id: int, G: NetworkGraph) -> Optional[float]:
        '''
        Spatial distances computed (weight sum of edges).
        '''
        p = self.paths_original.get(agent_id)
        if p is None:
            return None
        total = 0.0
        for u, v in zip(p, p[1:]):
            if u != v:  # exclude wait
                total += G[u][v]["weight"]
        return total
    
    def total_path_length_space(self, G: NetworkGraph) -> float:
        """Sum of spatial distances across all agents."""
        total = 0.0
        for agent_id in self.paths_original:
            d = self.path_length_space(agent_id, G)
            if d is not None:
                total += d
        return total
