import heapq
import math
from Network_graph import NetworkGraph


def reconstruct_path(node_to_predecessor, current):
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
    total_path = []
    while current in node_to_predecessor:
        total_path.append(current)
        current = node_to_predecessor[current]
    total_path.append(current)  
    total_path.reverse()
    return total_path


def a_star(G: NetworkGraph, start: int, goal: int, extended = False, heuristic = None) -> list | None:
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
    if heuristic is None:
        heuristic = h_manhattan

    # early exit if start == goal
    if not extended and start == goal:
        return [start]
    if extended and G.nodes[start]["original_id"] == goal:
        return [start]

    # find the corresponding nodes:
    # in extended mode find the node (goal_original_id, t=0) for spatial coordinates
    # in standard mode goal_ref is goal itself

    if extended:
        goal_ref = next( (nid for nid, d in G.nodes(data=True) if d["original_id"] == goal and d["t"] == 0), None )
        if goal_ref is None:
            return None   # goal does not exist in the expanded graph
    else:
        goal_ref = goal

    # node_to_predecessor[n] = predecessor of n on the current best path (shortest) from start
    node_to_predecessor = {}    

    # g[n] = cost of the best known path from start to n
    g = {start: 0}
    f = {start: heuristic(G, start, goal_ref)}

    # min-heap priority queue: (f_score, node_id)
    heap = [(f[start], start)]   

    while heap:   # ciclo heap

        # extract node with lowest f score
        f_current, current = heapq.heappop(heap)

        # skip outdated entries (duplicates with higher cost)
        if f_current > f.get(current, float("inf")):
            continue

        # check if goal is reached
        if extended and G.nodes[current]["original_id"] == goal:
            return reconstruct_path(node_to_predecessor, current)
        if not extended and current == goal:
            return reconstruct_path(node_to_predecessor, current)


        # explore outgoing edges
        for _, neighbor, edge_attrs in G.out_edges(current, data=True):

            # optimal path for reaching neighbor passing from the current node
            possible_g = g[current] + edge_attrs["weight"]

            if possible_g < g.get(neighbor, float("inf")):
                # better path from start to neighbor found through current
                node_to_predecessor[neighbor] = current   # update optimal path to neighbor
                g[neighbor] = possible_g   # update best cost found so far 
                f[neighbor] = possible_g + heuristic(G, neighbor, goal_ref)   # update estimated cost for optimal path to the goal
                heapq.heappush(heap, (f[neighbor], neighbor))

    return None


def h_euclidean(G: NetworkGraph, n: int, goal_node: int) -> float:
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
    return math.sqrt((x_goal - x_n) ** 2 + (y_goal - y_n) ** 2)

def h_manhattan(G: NetworkGraph, n: int, goal_node: int) -> float:
    """
    Manhattan distance heuristic between node n and goal_node.

    Args:
        G: the graph
        n: current node id
        goal_node: goal node id (già risolto, non original_id)
    Returns:
        Manhattan distance from n to goal_node
    """
    x_n,y_n = G.nodes[n]["x"], G.nodes[n]["y"]
    x_goal, y_goal = G.nodes[goal_node]["x"], G.nodes[goal_node]["y"]
    return abs(x_goal - x_n) + abs(y_goal - y_n)