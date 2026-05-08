import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import networkx as nx
from Network_graph import NetworkGraph
from MAPF_algorithm.plan_result import PlanResult

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

    # Print WAIT edges
    print("\nWAIT edges:")
    for src, dst, edge_attrs in G_expanded.edges(data=True):
        if edge_attrs.get("type_edge") == "wait":
            src_attrs = G_expanded.nodes[src]
            dst_attrs = G_expanded.nodes[dst]
            print(
                f"  ({src_attrs['original_id']},t={src_attrs['t']},id={src}) → "
                f"({dst_attrs['original_id']},t={dst_attrs['t']},id={dst})"
            )

    # Print MOVE edges 
    print("\nMOVE edges:")
    for src, dst, edge_attrs in G_expanded.edges(data=True):
        if edge_attrs.get("type_edge") == "move":
            src_attrs = G_expanded.nodes[src]
            dst_attrs = G_expanded.nodes[dst]
            print(
                f"  ({src_attrs['original_id']},t={src_attrs['t']},id={src}) → "
                f"({dst_attrs['original_id']},t={dst_attrs['t']},id={dst})"
            )


def animate_paths(G: NetworkGraph,  plan_result: PlanResult, name_solver: string) -> None:
    """
    Animation of agent on the graph

    Args:
        G:  original graph
        plan_result: resulting plan
    """

    colors = [ "blue","red","green","orange","purple","gold","deepskyblue","magenta","lime","darkcyan","brown","hotpink","olive","turquoise","navy","crimson","darkorange","teal","indigo","forestgreen"]

    pos = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}

    agent_ids = list(plan_result.paths_original.keys())
    paths     = {aid: plan_result.paths_original[aid] for aid in agent_ids}
    max_steps = max(len(p) for p in paths.values())

    fig, ax = plt.subplots(figsize=(8, 8))

    def draw_frame(t):
        ax.clear()

        nx.draw_networkx_edges(G, pos, ax=ax,edge_color="lightgray",arrows=True, arrowsize=10,connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_nodes(G, pos, ax=ax,node_size=300,node_color="whitesmoke",edgecolors="gray",linewidths=1.0)
        nx.draw_networkx_labels(G, pos, ax=ax,font_size=7,font_color="gray")

        for i, aid in enumerate(agent_ids):
            color = colors[i % len(colors)]
            path  = paths[aid]

            current_node = path[t] if t < len(path) else path[-1]
            goal_node    = path[-1]

            cx, cy = pos[current_node]
            gx, gy = pos[goal_node]

            ax.plot(gx, gy, "o",markersize=18,markerfacecolor="none",markeredgecolor=color,markeredgewidth=2.5)

            ax.plot(cx, cy, "o",markersize=14,color=color,zorder=5)

            ax.text(cx, cy, str(aid),ha="center", va="center",fontsize=8, fontweight="bold",color="white", zorder=6)

        ax.set_title(f"{name_solver} — timestep {t} / {max_steps - 1}")
        ax.axis("off")

    ani = animation.FuncAnimation(fig,draw_frame,frames=max_steps,interval=1000,  repeat=True)

    plt.tight_layout()
    plt.show()