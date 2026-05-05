from Network_graph import NetworkGraph
import matplotlib.pyplot as plt
import networkx as nx

def make_grid_graph(rows: int, cols: int, step: float = 1.0) -> NetworkGraph:
    """
    Builds a rows for cols grid graph where each node holds (x, y) coordinates
    and every cell is connected to its right and upper neighbours via
    bidirectional edges.

    Args:
        rows:  number of rows in the grid
        cols:  number of columns in the grid
        step:  distance between adjacent nodes (default 1.0, standard in MAPF)

    Returns:
        G: the constructed NetworkGraph
    """
    G = NetworkGraph()

    # --- Node creation ---
    # Each node gets a linear id  nid = r*cols + c  and spatial attributes x, y
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c
            attrs = {"x": float(c * step), "y": float(r * step)}
            G.add_node(nid, **attrs)

    # --- Edge creation ---
    # For every cell we add at most two bidirectional edges:
    #   • horizontal: current node ↔ right neighbour
    #   • vertical:   current node ↔ upper neighbour
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c

            # Horizontal edge (only if the next column exists)
            if c + 1 < cols:
                right = r * cols + (c + 1)
                G.add_edge(nid, right, weight=step)
                G.add_edge(right, nid, weight=step)

            # Vertical edge (only if the next row exists)
            if r + 1 < rows:
                up = (r + 1) * cols + c
                G.add_edge(nid, up, weight=step)
                G.add_edge(up, nid, weight=step)

    return G


def print_grid_on_terminal(G: NetworkGraph) -> None:
    """
    Prints the grid to the terminal showing each node's id,
    with the highest row (largest y) at the top, matching the
    expected visual orientation.
    """
    # Infer grid dimensions from the maximum coordinates stored in the nodes
    rows = int(max(attrs["y"] for _, attrs in G.nodes(data=True))) + 1
    cols = int(max(attrs["x"] for _, attrs in G.nodes(data=True))) + 1

    print("\nGrid (node ids):")
    # Print from the topmost row down to row 0
    for r in range(rows - 1, -1, -1):
        row_str = ""
        for c in range(cols):
            nid = r * cols + c
            row_str += f"{nid:3}"   # 3-character field for neat alignment
        print(row_str)


def plot_graph(G: NetworkGraph) -> None:
    """
    Renders the graph using matplotlib/networkx:
    - nodes drawn in steel-blue with a name or id label
    - directed edges in grey with a slight curve so both directions
      of a bidirectional pair remain visible
    - edge labels showing speed_limit or speed when available
    """
    # Map each node to its (x, -y) position so that y increases upward
    # in the coordinate system but downward on screen (image convention)
    pos = {node: (data["x"], -data["y"]) for node, data in G.nodes(data=True)}

    # Use "name" attribute as label when present, otherwise fall back to the numeric id
    labels = {
        node: data.get("name") or str(node)
        for node, data in G.nodes(data=True)
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw nodes, labels, and edges separately for fine-grained control
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color="steelblue")
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_color="white")
    nx.draw_networkx_edges(
        G, pos, ax=ax, arrows=True, arrowsize=15,
        edge_color="gray",
        connectionstyle="arc3,rad=0.1"  # slight arc to distinguish the two directions
    )

    # Edge labels: speed_limit takes priority over speed; empty string if neither is present
    edge_labels = {
        (u, v): str(d.get("speed_limit", d.get("speed", "")))
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)

    ax.set_title("MAPF Grid Graph")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def print_expanded_graph(G_expanded: NetworkGraph) -> None:
    """
    Prints the time-expanded graph to the terminal.
    Time-expanded graphs are common in MAPF formulations over a finite
    time horizon: each node represents a (original_node, timestep) pair,
    and edges are split into two categories:
      • "wait" – the agent stays in the same node at the next timestep
      • "move" – the agent moves to an adjacent node at the next timestep
    """
    print("Expanded graph")

    # --- Print nodes ---
    print("\nNodes:")
    for nid, attrs in G_expanded.nodes(data=True):
        # Show the expanded id, the original graph id, and the timestep
        print(f"  {nid}: (orig={attrs['original_id']}, t={attrs['t']})")

    # --- Print WAIT edges ---
    print("\nWAIT edges:")
    for src, dst, edge_attrs in G_expanded.edges(data=True):
        if edge_attrs.get("type_edge") == "wait":
            src_attrs = G_expanded.nodes[src]
            dst_attrs = G_expanded.nodes[dst]
            print(
                f"  ({src_attrs['original_id']},t={src_attrs['t']},id={src}) → "
                f"({dst_attrs['original_id']},t={dst_attrs['t']},id={dst})"
            )

    # --- Print MOVE edges ---
    print("\nMOVE edges:")
    for src, dst, edge_attrs in G_expanded.edges(data=True):
        if edge_attrs.get("type_edge") == "move":
            src_attrs = G_expanded.nodes[src]
            dst_attrs = G_expanded.nodes[dst]
            print(
                f"  ({src_attrs['original_id']},t={src_attrs['t']},id={src}) → "
                f"({dst_attrs['original_id']},t={dst_attrs['t']},id={dst})"
            )


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