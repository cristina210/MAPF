import random
from fleet import Fleet

def make_random_fleet(G: NetworkGraph, num_agents: int = 3) -> Fleet:
    all_nodes = list(G.nodes)
    
    # num_agents nodi iniziali casuali
    start_nodes = random.sample(all_nodes, num_agents)
    
    # num_agents nodi goal casuali tra i nodi rimanenti
    remaining_nodes = list(set(all_nodes) - set(start_nodes))
    goal_nodes = random.sample(remaining_nodes, num_agents)
    
    list_id    = []
    initial_locations = []
    goal_locations    = []

    for i in range(num_agents):
        list_id.append(i)
        initial_locations.append(start_nodes[i])
        goal_locations.append(goal_nodes[i])

    fleet = Fleet( list_id = list_id,initial_locations = initial_locations, goal_locations = goal_locations )
    return fleet