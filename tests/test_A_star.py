import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shortest_path_algorithm.A_star import a_star
from Network_graph import NetworkGraph
from upload_graph import make_grid_graph, plot_graph, print_grid_on_terminal
from upload_fleet import make_random_fleet


G = make_grid_graph(rows=5, cols=5, step=1.0)
plot_graph(G)
print_grid_on_terminal(G)

fleet = make_random_fleet(G, num_agents=3)

for i in range(fleet.num_of_agents):
    start = fleet.initial_locations[i]
    goal  = fleet.goal_locations[i]
    path  = a_star(G, start=start, goal=goal)
    print(f"Robot {i}: S{i}={start} → G{i}={goal} : {path}")