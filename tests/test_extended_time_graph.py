import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shortest_path_algorithm.A_star import a_star
from Network_graph import NetworkGraph
from upload_graph import make_grid_graph, plot_graph, print_grid_on_terminal,print_expanded_graph, graph_difference
from MAPF_algorithm.extended_time_graph import TimeExpandedGraph


## test of rebuild_with_constraints -> time_expansion_graph_with_constr
G = make_grid_graph(rows=3, cols=3, step=1.0)
plot_graph(G)
print_grid_on_terminal(G)

teg = TimeExpandedGraph(G, T = 4)
print_expanded_graph(teg.G_expanded)

teg.rebuild_with_constraints(vertex_constraints={8}, edge_constraints={(14, 15), (28,17)})
print_expanded_graph(teg.G_expanded)

teg2 = TimeExpandedGraph(G, T=4)

diff = graph_difference(teg2.G_expanded, teg.G_expanded)
mapping_per_print = {    "nodes_only_in_G1": "Nodes only in graph teg2 (no constr):", "nodes_only_in_G2": "Nodes only in graph teg (with constr):",
    "edges_only_in_G1": "Edges only in graph teg2 (no constr):", "edges_only_in_G2": "Edges only in graph teg (with constr):"}
[print(f"{label}\n{diff[key]}") for key, label in mapping_per_print.items()]

## test of update_teg_with_adding_constraints -> _remove_nodes_edges_from_graph
teg.update_teg_with_adding_constraints(new_vertex_constr={7}, new_edge_constr={(28,33)})

diff = graph_difference(teg2.G_expanded, teg.G_expanded)
mapping_per_print = {    "nodes_only_in_G1": "Nodes only in graph teg2 (no constr):", "nodes_only_in_G2": "Nodes only in graph teg (with constr):",
    "edges_only_in_G1": "Edges only in graph teg2 (no constr):", "edges_only_in_G2": "Edges only in graph teg (with constr):"}
[print(f"{label}\n{diff[key]}") for key, label in mapping_per_print.items()]




