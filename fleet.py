class Fleet:
    def __init__(self, list_id, initial_locations, goal_locations):
        self.id: list                = list_id
        self.initial_locations: list = initial_locations
        self.goal_locations: list    = goal_locations
        self.num_of_agents: int      = len(list_id)
        self.state: list             = self._init_state(goal_locations)

    def _init_state(self, goal_locations):
        state_list = []
        for g in goal_locations:
            if g is None:
                state_list.append("idle")
            else:
                state_list.append("busy")
        return state_list

    def update_goal_location(self, id_agent, goal_location):
        self.goal_locations[id_agent] = goal_location
        if goal_location is None:
            self.state[id_agent] = "idle"
        else:
            self.state[id_agent] = "busy"
