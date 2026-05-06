from Network_graph import NetworkGraph

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
