import random
from Network_graph import NetworkGraph  # o dal path corretto
from fleet import Fleet

def make_random_fleet(G: NetworkGraph, num_agents: int = 3) -> Fleet:
    """
    Generate a fleet of agents with initial position and goals chosen from nodes of a graph.
    Start e goal are guaranteed to be different between agents.

    Args:
        G:          graph
        num_agents: number of agents to generate
    Returns:
        Object fleet
    """
    all_nodes = list(G.nodes)

    starts = random.sample(all_nodes, num_agents)

    remaining = list(set(all_nodes) - set(starts))
    goals = random.sample(remaining, num_agents)

    # Ids are generated from 0 to num agents - 1
    ids = list(range(num_agents))

    return Fleet(ids=ids, starts=starts, goals=goals)