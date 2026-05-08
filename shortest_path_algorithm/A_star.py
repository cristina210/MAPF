import heapq
import math
from Network_graph import NetworkGraph


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


def a_star_with_focal_search(G: NetworkGraph, start: int, goal: int, congestion_dict: dict, extended: bool = False, heuristic=None, w: float = 1) -> list | None:
    """
    A* with focal search (low-level planner for BCBS).

    Maintains two lists:
    - OPEN: dictionary containing nodes and corresponding costs 
    - FOCAL: subset of OPEN containing all nodes with f <= w * f_min that make a priority queue ordered by g_c (accumulated congestion cost along the path).

    At each iteration the node with lowest g_c is extracted from FOCAL, guaranteeing that the returned solution has cost at most w * C*
    (bounded suboptimal) while minimising conflicts with other agents.

    Args:
        G:               directed NetworkGraph (standard or time-expanded)
        start:           id of the start node in G
        goal:            id of the goal node (standard) or original_id (extended)
        congestion_dict: maps expanded node id -> number of other agents passing through it.
                         Used to compute g_c: the accumulated conflict cost along the path.
        extended:        if True, works on a time-expanded graph
        heuristic:       spatial heuristic f1 (default: h_manhattan)
        w:               suboptimality factor. w=1 -> optimal. w>1 -> bounded suboptimal, faster search: bias towards avoiding conflicts.
    Returns:
        list of node ids representing the path, or None if no path exists.
    """
    if heuristic is None:
        heuristic = h_manhattan  # default spatial heuristic

    # early exit: agent is already at goal
    if not extended and start == goal:
        return [start]
    if extended and G.nodes[start]["original_id"] == goal:
        return [start]

    # in extended mode, goal_ref is the expanded node (goal_original_id, t=0)
    # used to retrieve spatial coordinates for the heuristic
    if extended:
        goal_ref = next((nid for nid, d in G.nodes(data=True)
                         if d["original_id"] == goal and d["t"] == 0), None)
        if goal_ref is None:
            return None  # goal node does not exist in the expanded graph
    else:
        goal_ref = goal  # standard mode: goal_ref is goal itself

    # node_to_predecessor[n] = predecessor of n on the current best path from start
    # used to reconstruct the path once the goal is reached
    node_to_predecessor = {}

    # g[n]: best known travel cost from start to n
    g = {start: 0}

    # g_c[n]: accumulated congestion cost along the best known path from start to n
    # counts how many other agents occupy the nodes visited so far
    g_c = {start: congestion_dict.get(start, 0)}

    # f[n] = g[n] + h(n): estimated total cost from start to goal through n
    f = {start: heuristic(G, start, goal_ref)}


    # use a dictionary for saving node to be explored (dictionary is chosen since is necessary to know to extract elements not necessarily in the first position)
    open_dict = {start: (f[start], g_c[start])}

    while open_dict:

        # find f_min: lowest f value currently in OPEN dict
        f_min = open_dict[next(iter(open_dict))][0]
        for st in open_dict:
            if open_dict[st][0] <= f_min:
                f_min = open_dict[st][0]

        # threshold: any node in OPEN with f <= w * f_min enters FOCAL
        threshold = f_min * w

        # Use a heap for the focal: extract the lowest element respect to g_c
        focal = [] 
        for node in open_dict: 
            if f[node] <= threshold: 
                heapq.heappush(focal, (g_c[node], node))

        # remove from focal the best node in term of g_c
        _, current = heapq.heappop(focal)
        
        # remove from open the best node
        open_dict.pop(current, None)
        f_current = f[current]

        # skip outdated entries (duplicates with higher cost)
        if f_current > f.get(current, float("inf")):
            continue

        # goal check
        if extended and G.nodes[current]["original_id"] == goal:
            return reconstruct_path(node_to_predecessor, current)
        if not extended and current == goal:
            return reconstruct_path(node_to_predecessor, current)

        # expand current: explore all outgoing edges
        for _, neighbor, edge_attrs in G.out_edges(current, data=True):

            # tentative cost to reach neighbor through current
            possible_g = g[current] + edge_attrs["weight"]

            # only update if a better path to neighbor is found
            if possible_g < g.get(neighbor, float("inf")):

                node_to_predecessor[neighbor] = current  # record best predecessor
                g[neighbor] = possible_g                 # update best travel cost
                f[neighbor] = possible_g + heuristic(G, neighbor, goal_ref)  # update f score

                # accumulate congestion: add occupancy of neighbor by other agents
                g_c[neighbor] = g_c[current] + congestion_dict.get(neighbor, 0)
                # add new nodes to be explored
                open_dict[neighbor] = (f[neighbor], g_c[neighbor])
    return None  # no path found


def h_euclidean(G: NetworkGraph, n: int, goal_node: int) -> float:
    """
    Euclidean distance heuristic between node n and goal_node.
    Admissible only if edge weights equal Euclidean distances.
    goal_node must already be resolved to a node id in G.

    Args:
        G:         the graph (standard or time-expanded)
        n:         current node id
        goal_node: goal node id, already resolved
    Returns:
        Euclidean distance from n to goal_node
    """
    x_n,y_n = G.nodes[n]["x"], G.nodes[n]["y"]
    x_goal, y_goal = G.nodes[goal_node]["x"], G.nodes[goal_node]["y"]
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
