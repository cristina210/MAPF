import random
from Network_graph import NetworkGraph  # o dal path corretto
from fleet import Fleet



def make_random_fleet(G: NetworkGraph, num_agents: int = 3) -> Fleet:
    """
    Generate a fleet of agents with unique starts and goals.
    Each goal is sampled avoiding:
    - all start nodes
    - already assigned goals
    - its own start node
    """

    all_nodes = list(G.nodes)

    # 1. sample unique starts
    starts = random.sample(all_nodes, num_agents)

    available_goals = set(all_nodes)
    available_goals -= set(starts)

    goals = []

    for start in starts:
        # exclude own start + already assigned goals
        candidates = list(available_goals - {start})

        if not candidates:
            raise ValueError("Not enough nodes to assign unique goals")

        goal = random.choice(candidates)

        goals.append(goal)
        #available_goals.remove(goal)

    ids = list(range(num_agents))

    return Fleet(ids=ids, starts=starts, goals=goals)  