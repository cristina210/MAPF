from Network_graph import NetworkGraph
from upload_graph import make_grid_graph, plot_graph,print_expanded_graph
from upload_fleet import make_random_fleet
from shortest_path_algorithm.A_star import a_star
from MAPF_algorithm.utils import precomputed_shortest_path_A_star, approximated_makespan, time_expansion_graph

G = make_grid_graph(rows=4, cols=4, step=1.0)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("\nNodes:")
for nid, d in G.nodes(data=True):
    print(f"  {nid}: {d}")

print("\nEdges:")
for src, dst, d in G.edges(data=True):
    print(f"  {src} → {dst}: {d}")

plot_graph(G)

fleet = make_random_fleet(G, num_agents=3)
print(f"Fleet: {fleet.num_of_agents} agents")
print(f"  initial_locations: {fleet.initial_locations}")
print(f"  goal_locations:    {fleet.goal_locations}")
print(f"  states:            {fleet.state}")

matrix_sp = precomputed_shortest_path_A_star(G)


# ...


