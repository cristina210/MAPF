from Network_graph import NetworkGraph


def create_edges_constraints(diz_swaps, edge_constraints_set_of_sets):
    """
    Expands a set of edge constraints by adding the swap-pair inverse of each edge.
    In a time-expanded graph, forbidding a move edge (u_t, v_t+1) must also
    forbid the reverse move (v_t, u_t+1) to prevent swap conflicts — two agents
    exchanging positions in the same timestep interval. This function enforces
    that symmetry automatically using the precomputed swap_pairs dictionary.

    Args:
        diz_swaps: swap_pairs dict from build_teg_mappings: {(u_t, v_t+1): (v_t, u_t+1), ...}
        edge_constraints_set_of_sets: set of (src, dst) expanded edge pairs to forbid.
    Returns:
        expanded set of edge constraints including swap-pair inverses
    """
    if edge_constraints_set_of_sets is None:
        return set()
    edge_constraints_final = set()
    for (src, dst) in edge_constraints_set_of_sets:
        edge_constraints_final.add((src, dst))
        if (src, dst) in diz_swaps:
            # add the swap-pair inverse if it exists in the original graph
            # (only bidirectional edges have a swap pair)
            edge_constraints_final.add(diz_swaps[(src, dst)])
    return edge_constraints_final


def build_teg_mappings(G: NetworkGraph, T: int) -> tuple:
    """
    Precomputes the node id mapping and swap pairs for a time-expanded graph,
    without constructing the graph itself.
    This is called once per TEG instantiation to obtain the structural data
    needed before applying constraints.

    Node id mapping: 
    Expanded node ids are assigned sequentially: expanded_id = original_id * T + t
    This formula is deterministic and enables the static conversion methods TimeExpandedGraph.compute_expanded_id / compute_original_id to work without a TEG instance.
    Requires: node ids in G must be consecutive integers starting from 0.

    Swap pairs:
    For each move edge (u, v) in G and each timestep t, the move (u_t -> v_t+1) has a possible swap counterpart (v_t -> u_t+1).
    Both directions are registered only if the reverse edge (v, u) exists in the original graph (the graph is bidirectional there).

    Args:
        G: original directed NetworkGraph with consecutive integer node ids
        T: number of timesteps in the time-expanded graph
    Returns:
        old_id_to_new: list where old_id_to_new[original_id][t] = expanded_id
        swap_pairs:    dict {(u_t, v_t+1): (v_t, u_t+1)} for all move edges
    """
    old_id_to_new = []
    swap_pairs = {}
    count = 0        # sequential expanded id counter

    # build node id mapping: one expanded id per (original_node, timestep) pair
    for id_n, _ in G.nodes(data=True):
        expanded = []
        for t in range(T):
            expanded.append(count)
            count += 1
        old_id_to_new.append(expanded)

    # build swap pairs: for each directed edge (src, dst) and each timestep t,
    # register the move arc and its temporal swap counterpart
    for src, dst, _ in G.edges(data=True):
        for t in range(T - 1):
            u_t  = old_id_to_new[src][t]
            v_t1 = old_id_to_new[dst][t + 1]
            v_t  = old_id_to_new[dst][t]
            u_t1 = old_id_to_new[src][t + 1]

            # forward arc -> its swap counterpart
            swap_pairs[(u_t, v_t1)] = (v_t, u_t1)

            # reverse arc -> its swap counterpart (only if reverse edge exists)
            if G.has_edge(dst, src):
                swap_pairs[(v_t, u_t1)] = (u_t, v_t1)

    return old_id_to_new, swap_pairs


def time_expansion_graph_with_constr(G: NetworkGraph, T: int, old_id_to_new: list, vertex_constraints: set = None, edge_constraints: set = None) -> NetworkGraph:
    """
    Builds the time-expanded graph (TEG).
    All possible nodes outside the ones restricted for constraints are added
    Two edge types are added:
    - Wait edges:  (node, t) -> (node, t+1)  — agent stays in place for one timestep
    - Move edges:  (u, t) -> (v, t+1)     — agent moves from u to v in one timestep

    Constraints are enforced by omitting forbidden nodes and edges during construction:
    - Vertex constraints: expanded node ids to exclude — their incident edges are also omitted
    - Edge constraints:   (src, dst) expanded edge pairs to exclude
    Note: edge constraints should already include swap-pair inverses at this point.

    Args:
        G: original directed NetworkGraph
        T: number of timesteps
        old_id_to_new: precomputed mapping from build_teg_mappings
        vertex_constraints: set of expanded node ids to exclude (default: empty)
        edge_constraints: set of (src, dst) expanded edge pairs to exclude (default: empty)
    Returns:
        time-expanded NetworkGraph with wait and move edges respecting all constraints
    """
    if vertex_constraints is None:
        vertex_constraints = set()
    if edge_constraints is None:
        edge_constraints = set()

    G_expanded = NetworkGraph()

    # --- add nodes and wait edges ---
    for id_n, node_attrs in G.nodes(data=True):
        for t in range(T):
            count = old_id_to_new[id_n][t]
            if count not in vertex_constraints:
                # add node only if not under vertex constraint
                G_expanded.add_node(count, x=node_attrs["x"], y=node_attrs["y"], t=t, original_id=id_n)
            # add wait edge from t-1 to t if both endpoints
            if t > 0:
                node_prev = old_id_to_new[id_n][t - 1]
                node_curr = old_id_to_new[id_n][t]
                if node_curr not in vertex_constraints and node_prev not in vertex_constraints and (node_prev, node_curr) not in edge_constraints:
                    # wait edges
                    G_expanded.add_edge(node_prev, node_curr, weight=1, type_edge="wait")

    # --- add move edges ---
    for src, dst, edge_attrs in G.edges(data=True):
        for t in range(T - 1):
            # move edges
            u_t  = old_id_to_new[src][t]
            v_t1 = old_id_to_new[dst][t + 1]
            # add move edge only if both endpoints and the edge are allowed
            if u_t not in vertex_constraints and v_t1 not in vertex_constraints and (u_t, v_t1) not in edge_constraints:
                G_expanded.add_edge(u_t, v_t1, weight=1, type_edge="move")

    return G_expanded



