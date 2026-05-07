from Network_graph import NetworkGraph

def time_expansion_graph_with_constr(G: NetworkGraph, T: int, vertex_constraints = None, edge_constraints = None) -> NetworkGraph:
    """
    Build a time-expanded graph from a standard NetworkGraph.

    Each node in G is replicated T times (one per timestep), creating
    a layered graph where movement across time is explicit.
    Two types of edges are created:
    - wait edges: same node at t -> same node at t+1 (agent stays in place)
    - move edges: node at t -> adjacent node at t+1 (agent moves)

    This function provide the possibility to force two type of constraint:
    - Node constraint: a node of the extended-time graph can't be used -> in-edges and out-edges are removed 
    - Edge constraint: an edge of the extended-time graph can't be used -> edge removed
    Actually, the graph extended-time is built from scratch taking into consideration this constraints.
    This constraints appear in MAPF solver that exploits extended-time graph. Indeed this two type of constraint are the two classical ones.
    Note: node indices in G must be consecutive starting from 0.

    Args:
        G: original directed NetworkGraph
        T_lowerbound: minimum number of timesteps (T = T_lowerbound + delta_t)   DA MIGLIORARE 
        vertex_constraints: set of node ids in time-extended graph that cannot be visited
        edge_constraints: set of (src, dst) expanded edge pairs in time-extended graph that cannot be used
    Returns:
        time-expanded NetworkGraph with wait and move edges satisfying the constraints
    """

    if vertex_constraints is None:
        vertex_constraints = set()
    if edge_constraints is None:
        edge_constraints = set()
    G_expanded = NetworkGraph()

     # mapping from original node id to list of expanded node ids (one per number of timestep T)
    old_id_to_new_list = []  

    count = 0   # helps in creating new node ids

    for id_n, node_attrs in G.nodes(data=True):
        new_di_nodes_from_id_n_expansion = []   # list of expanded nodes for id_n

        for t in range(0,T):
            # append count in anycase to keep mapping consistent
            new_di_nodes_from_id_n_expansion.append(count)
            if count not in vertex_constraints:
                # create expanded node with spatial coords, timestep and original id
                new_node_from_id_n_attr = {"x": node_attrs["x"], "y": node_attrs["y"], "t": t, "original_id": id_n}
                G_expanded.add_node(count, **new_node_from_id_n_attr)
            count = count + 1

            # add wait edge from t-1 to t if not forbidden by constraints
            if t > 0:
                node_prev = new_di_nodes_from_id_n_expansion[t-1]   # count
                node_curr = new_di_nodes_from_id_n_expansion[t]     # count + 1
                if node_curr not in vertex_constraints and node_prev not in vertex_constraints and (node_prev, node_curr) not in edge_constraints and (node_curr, node_prev) not in edge_constraints:
                    G_expanded.add_edge(node_prev, node_curr, weight=1, type_edge="wait")  

        old_id_to_new_list.append(new_di_nodes_from_id_n_expansion)

    # add move edges: src at t -> dst at t+1 if not forbidden by constraints
    for src, dst, edge_attrs in G.edges(data=True):
        new_nodes_correspondent_to_src = old_id_to_new_list[src]
        new_nodes_correspondent_to_dst = old_id_to_new_list[dst]
        for t in range(0,T-1):
            node_t_src = new_nodes_correspondent_to_src[t]
            node_tplus1_dst =  new_nodes_correspondent_to_dst[t+1]
            if node_tplus1_dst not in vertex_constraints and node_t_src not in vertex_constraints and (node_t_src, node_tplus1_dst) not in edge_constraints and (node_tplus1_dst, node_t_src) not in edge_constraints:
                G_expanded.add_edge(node_t_src, node_tplus1_dst, weight = 1, type_edge="move")
    return G_expanded, old_id_to_new_list

