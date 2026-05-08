from graph_utils.upload_graph import make_grid_graph
from graph_utils.graph_visualization import plot_graph, animate_paths
from graph_utils.graph_analysis import precomputed_shortest_path_A_star
from fleet_utils.upload_fleet import make_random_fleet
from instance import MAPFInstance
from environment import MAPFEnvironment
from MAPF_algorithm.solvers.PP import PrioritizedPlanner
from MAPF_algorithm.solvers.CBS import CBSSolver
from MAPF_algorithm.solvers.ECBS import ECBSSolver
from MAPF_algorithm.solvers.A_star_naive import AStarNaive
from shortest_path_algorithm.A_star import a_star
from results_handler import print_comparison, save_comparison, save_instance, save_simulation
import random
random.seed(20)

#### Build graph ####

G = make_grid_graph(rows=4, cols=4, step=1.0)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("\nNodes:")
for nid, d in G.nodes(data=True):
    print(f"  {nid}: {d}")

print("\nEdges:")
for src, dst, d in G.edges(data=True):
    print(f"  {src} → {dst}: {d}")

plot_graph(G)


#### Built fleet ####

fleet = make_random_fleet(G, num_agents=11)

print(f"\nFleet: {fleet.num_agents()} agents")
for agent in fleet.agents.values():
    print(f"  {agent}")


#### Instance definition ####

instance = MAPFInstance(graph=G, fleet=fleet, name="test_4x4")
print(f"\n{instance}")

print("time considered in the Extended time graph", instance.T)

#### Planning ####

## Plan path with PP

planner_PP = PrioritizedPlanner(instance.graph, instance.T) 
result_PP  = planner_PP.plan(instance.fleet)

if not result_PP.success:
    print(f"\nPlanning PP failed for agent {result_PP.failed_agent}")

## Plan path with CBS 

planner_CBS = CBSSolver(instance.graph, instance.T) 
result_CBS  = planner_CBS.plan(instance.fleet)

if not result_CBS.success:
    print(f"\nPlanning CBS failed for agent {result_CBS.failed_agent}")

## Plan path with ECBS w_l = 10, w_h = 1

planner_ECBS_1_10 = ECBSSolver(instance.graph, instance.T, w_l=10) 
result_ECBS_1_10  = planner_ECBS_1_10.plan(instance.fleet)

if not result_ECBS_1_10.success:
    print(f"\nPlanning ECBS failed for agent {result_ECBS_1_10.failed_agent}")

## Plan path with ECBS w_l = 5, w_h = 1

planner_ECBS_1_5 = ECBSSolver(instance.graph, instance.T, w_l=1.5) 
result_ECBS_1_5  = planner_ECBS_1_5.plan(instance.fleet)

if not result_ECBS_1_5.success:
    print(f"\nPlanning ECBS failed for agent {result_ECBS_1_5.failed_agent}")


#### Simulation ####

## Simulation with PP

env_PP = MAPFEnvironment(instance)
history_PP = env_PP.run_plan(result_PP)

print(f"\nSimulation completed for PP planner in {len(history_PP) - 1} timesteps")
for snapshot in history_PP:
    print(f"  t={snapshot['timestep']}: {snapshot['positions']}")


## Simulation with CBS

env_CBS = MAPFEnvironment(instance)
history_CBS = env_CBS.run_plan(result_CBS)

print(f"\nSimulation completed for CBS in {len(history_CBS) - 1} timesteps")
for snapshot in history_CBS:
    print(f"  t={snapshot['timestep']}: {snapshot['positions']}")

## Simulation with ECBS w_l = 10, w_h = 1

env_ECBS_1_10 = MAPFEnvironment(instance)
history_ECBS_1_10 = env_ECBS_1_10.run_plan(result_ECBS_1_10)

print(f"\nSimulation completed for ECBS in {len(history_ECBS_1_10) - 1} timesteps")
for snapshot in history_ECBS_1_10:
    print(f"  t={snapshot['timestep']}: {snapshot['positions']}")

## Simulation with ECBS w_l = 5, w_h = 1 

env_ECBS_1_5 = MAPFEnvironment(instance)
history_ECBS_1_5 = env_ECBS_1_5.run_plan(result_ECBS_1_5)

print(f"\nSimulation completed for ECBS in {len(history_ECBS_1_5) - 1} timesteps")
for snapshot in history_ECBS_1_5:
    print(f"  t={snapshot['timestep']}: {snapshot['positions']}")


#### Performance A* vs PP vs CBS vs ECBS ####

planner_astar = AStarNaive(instance.graph)
result_astar  = planner_astar.plan(instance.fleet)

solvers = {
    "A* Naive": result_astar,
    "PP": result_PP,
    "CBS": result_CBS,
    f"ECBS w_l={planner_ECBS_1_10.w_l}, w_h={planner_ECBS_1_10.w_h}": result_ECBS_1_10,
    f"ECBS w_l={planner_ECBS_1_5.w_l}, w_h={planner_ECBS_1_5.w_h}": result_ECBS_1_5,
}

## print on terminal
print_comparison(instance, solvers)

## output files
save_comparison(instance=instance, solvers=solvers,filepath="results/results.txt")
save_instance(instance,filepath="results/instance.txt")
save_simulation(history_PP,filepath="results/simulation_PP.txt")
save_simulation(history_CBS,filepath="results/simulation_CBS.txt")
# Visualization
animate_paths(G, result_CBS, "CBS")
animate_paths(G, result_PP, "PP")