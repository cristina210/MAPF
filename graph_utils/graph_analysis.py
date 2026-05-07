from Network_graph import NetworkGraph
from shortest_path_algorithm.A_star import a_star


def graph_difference(G1, G2):
    """
    Compute differences between two NetworkX graphs.

    Args:
        G1, G2: networkx graphs

    Returns:
        dict with:
            - nodes_only_in_G1
            - nodes_only_in_G2
            - edges_only_in_G1
            - edges_only_in_G2
    """

    # --- Nodes ---
    nodes_G1 = set(G1.nodes())
    nodes_G2 = set(G2.nodes())

    nodes_only_in_G1 = nodes_G1 - nodes_G2
    nodes_only_in_G2 = nodes_G2 - nodes_G1

    # --- Edges ---
    edges_G1 = set(G1.edges())
    edges_G2 = set(G2.edges())

    edges_only_in_G1 = edges_G1 - edges_G2
    edges_only_in_G2 = edges_G2 - edges_G1

    return {
        "nodes_only_in_G1": nodes_only_in_G1,
        "nodes_only_in_G2": nodes_only_in_G2,
        "edges_only_in_G1": edges_only_in_G1,
        "edges_only_in_G2": edges_only_in_G2,
    }



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
