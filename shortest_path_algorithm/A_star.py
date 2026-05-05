import heapq   
from Network_graph import NetworkGraph
import math


def reconstruct_path(node_to_predecessor: dict, current: int) -> list:
    """
    Function use in A star function.
    Reconstruct the optimal path from start to current node.
    Follows the predecessor chain stored in node_to_predecessor
    until the start node is reached.
    
    Args:
        node_to_predecessor: dictionary mapping each node to its predecessor
        current: goal node from which to start backtracking
    Returns:
        list of node ids from start to goal that identify the shortest path
    """
    total_path = [current]
    while current in node_to_predecessor:
        current = node_to_predecessor[current]
        total_path.insert(0, current) 
    return total_path


def a_star(G: NetworkGraph, start: int, goal: int, extended = False) -> list | None:
    """
    A* pathfinding algorithm on a NetworkGraph. This function support both the shortest path search
    in graph and in time-expanded graph
    
    Supports two modes:
    - standard (extended=False): finds shortest path from start to goal node id
    - extended (extended=True): works on a time-expanded graph where goal is
      matched by original_id attribute, allowing the agent to reach the goal
      at any timestep

    Args:
        G: directed NetworkGraph (standard or time-expanded)
        start: id of the start node in G. id of the start node (standard) or the id of (start_original, t=0)
        goal: id of the goal node (standard) or original_id (extended)
        extended: if True, uses time-expanded graph logic
    Returns:
        list of node ids representing the optimal path (in the extended-time graph in case), or None if no path exists
    """

    # early exit if start == goal
    if not extended and start == goal:
        return [start]
    if extended and G.nodes[start]["original_id"] == goal:
        return [start]

    # node_to_predecessor[n] = predecessor of n on the current best path (shortest) from start
    node_to_predecessor = {}    

    # g[n] = cost of the best known path from start to n
    g = {}   
    for id_node in G.nodes:
        g[id_node] = float("inf")   
    g[start] = 0   

    # f[n] = g[n] + h(n) = estimated total cost through n 
    # with: g the cost function for path from start to n and h the estimated cost function for path from n to goal node
    f = {}
    for id_node in G.nodes:
        f[id_node] = float("inf")
    f[start] = h_euclidean(G, start, goal, extended)    # choosen between admissible heuristics (ATTENZIONE: se pesi diversi da distanza allora non ha più senso l'euclidea)

    # min-heap priority queue: (f_score, node_id)
    heap_list = []    
    heapq.heappush(heap_list, (f[start], start))    

    while heap_list:   # ciclo heap

        # extract node with lowest f score
        f_current, current = heapq.heappop(heap_list)

        # skip outdated entries (duplicates with higher cost)
        if f_current > f[current]:   
            continue

        # check if goal is reached
        if extended and G.nodes[current]["original_id"] == goal:
            return reconstruct_path(node_to_predecessor, current)
        if current == goal and extended == False:
            return reconstruct_path(node_to_predecessor, current)

        # explore outgoing edges
        for src, neighbor, edge_attrs in G.out_edges(current, data=True):
            
            d =edge_attrs["weight"]   # d(current, neighbor)

            # optimal path for reaching neighbor passing from the current node
            possible_g = g[current] + d

            if possible_g < g[neighbor]:
                # better path from start to neighbor found through current
                node_to_predecessor[neighbor] = current  # update optimal path to neighbor
                g[neighbor]=possible_g   # update best cost found so far 
                f[neighbor] = possible_g + h_euclidean(G, neighbor, goal, extended)   # update estimated cost for optimal path to the goal

                heapq.heappush(heap_list,(f[neighbor], neighbor))  

    return None


def h_euclidean(G: NetworkGraph, n: int, goal: int, extended: bool = False) -> float:
    """
    Euclidean distance heuristic between node n and goal.
    
    In extended mode, goal is an original_id so the function looks up
    the corresponding node at t=0 to retrieve its spatial coordinates.
    Note: admissible only if edge weights equal Euclidean distances.

    Args:
        G: the graph
        n: current node id
        goal: goal node id (standard) or original_id (extended)
        extended: if True, resolves goal as original_id
    Returns:
        Euclidean distance estimate from n to goal in Graph
    """
    if extended:
        # retrieve id node in the extended-time graph corresponding to (goal, t=0)
        goal_nodes = [nid for nid, d in G.nodes(data=True) if d["original_id"] == goal and d["t"] == 0]
        goal_node = goal_nodes[0]
    else:
        goal_node = goal

    # retrieve spatial coordinates
    x_goal = G.nodes[goal_node]["x"]
    y_goal = G.nodes[goal_node]["y"]
    x_n = G.nodes[n]["x"]
    y_n = G.nodes[n]["y"]

    return math.sqrt((x_goal - x_n)**2 + (y_goal - y_n)**2)