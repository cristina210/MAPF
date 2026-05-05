from Network_graph import NetworkGraph
from shortest_path_algorithm.A_star import a_star
from extended_time_graph import TimeExpandedGraph
from time_graph_builder import time_expansion_graph_with_constr  # se serve ancora qui

'''
ALTRO:
# possibilità: Iterative Deepening on T / Incremental Time Expansion: is parte da un T piccolo (esempio makespan non considerando vincoli)
# e si incrementa man mano se non si trova soluzione ammissibile. Ma incrementando da T a T+1 bisogna risolvere di nuovo MAPF
# bisogna capire come farlo ottimizzando 

T = T_lowerbound

while True:
    build_or_update_model(T)

    solution = solve_MAPF(T)

    if solution exists:
        return solution, T

    T += 1
'''

def precomputed_shortest_path_A_star(G: NetworkGraph) -> list: 
    """
    Precompute shortest path lengths between all pairs of nodes using A*.
    
    Builds a matrix where matrix[i][j] represents the number of nodes
    in the shortest path from node i to node j.
    Note: node indices must be consecutive starting from 0 since the
    correspondence between matrix position and node id is exploited
    (matrix[i][j] refers to the path from node with id=i to node with id=j).
    In addition it assumes that distance between nodes / weight on edge / cost of action
    is equal to 1 as in classic MAPF.

    Args:
        G: directed NetworkGraph
    Returns:
        2D list (matrix) of shortest path lengths between all node pairs
    """
    all_nodes = list(G.nodes)
    matrix_all_shortest_path_len = []
    for i in range(len(all_nodes)):
        inner_list = []
        for j in range(len(all_nodes)):
            sp_len = len(a_star(G, all_nodes[i], all_nodes[j]))    # weight for each move in the graph is equal to one
            inner_list.append(sp_len)
        matrix_all_shortest_path_len.append(inner_list) 
    return matrix_all_shortest_path_len    

def approximated_makespan(list_starting_node: list, list_goal_node: list, matrix_shortest_path: list) -> int:
    """
    Compute the approximated makespan for a fleet of agents given shortest paths and start and goal node for each agent.
    
    The makespan is the time at which the last agent reaches its goal.
    This is a lower bound approximation: it assumes no conflicts between
    agents when each of them use its shortest path, so the real makespan will be >= this value.
    Note: valid only if edge traversal time equals edge weight,
    as in classic MAPF formulation (unit cost per step).

    Args:
        list_starting_node: list of start node ids, one per agent
        list_goal_node: list of goal node ids, one per agent
        matrix_shortest_path: precomputed matrix of shortest path lengths
    Returns:
        approximated makespan (length of the longest individual shortest path)
    """
    max_sp = 0
    for i in range(len(list_starting_node)):
        node_start = list_starting_node[i]
        node_goal  = list_goal_node[i]
        sp_ij = matrix_shortest_path[node_start][node_goal]
        # find the makespan (maximum length for the shortest paths considered)
        if sp_ij > max_sp:   
            max_sp = sp_ij
    return max_sp 


# vedere https://github.com/GavinPHR/Multi-Agent-Path-Finding/blob/master/cbs_mapf/planner.py
# vedere https://github.com/GavinPHR/Space-Time-AStar/blob/master/stastar/neighbour_table.py