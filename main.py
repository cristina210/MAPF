from graph_utils.upload_graph import make_grid_graph
from graph_utils.graph_visualization import plot_graph, animate_paths
from graph_utils.graph_analysis import precomputed_shortest_path_A_star
from fleet_utils.upload_fleet import make_random_fleet
from instance import MAPFInstance
from environment import MAPFEnvironment
from MAPF_algorithm.solvers.PP import PrioritizedPlanner
from MAPF_algorithm.solvers.CBS import CBSSolver
from shortest_path_algorithm.A_star import a_star


# Build graph

G = make_grid_graph(rows=4, cols=4, step=1.0)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("\nNodes:")
for nid, d in G.nodes(data=True):
    print(f"  {nid}: {d}")

print("\nEdges:")
for src, dst, d in G.edges(data=True):
    print(f"  {src} → {dst}: {d}")

plot_graph(G)


# Built fleet 

fleet = make_random_fleet(G, num_agents=10)

print(f"\nFleet: {fleet.num_agents()} agents")
for agent in fleet.agents.values():
    print(f"  {agent}")


# Shortest path

matrix_sp = precomputed_shortest_path_A_star(G)
print(f"\nShortest path matrix ({len(matrix_sp)}x{len(matrix_sp[0])}) computed")


# Instance definition

instance = MAPFInstance(graph=G, fleet=fleet, name="test_4x4")
print(f"\n{instance}")

print(instance.T)
# Plan path with PP

planner_PP = PrioritizedPlanner(instance.graph, instance.T) 
result_PP  = planner_PP.plan(instance.fleet)

if not result_PP.success:
    print(f"\nPlanning failed for agent {result_PP.failed_agent}")
else:
    print(f"\nPlanning succeeded:")
    for agent_id, path in result_PP.paths.items():
        print(f"  Agent {agent_id}: {len(path)} steps -> {path}")

# Plan path with CBS

planner_CBS = CBSSolver(instance.graph, instance.T) 
result_CBS  = planner_CBS.plan(instance.fleet)

if not result_PP.success:
    print(f"\nPlanning failed for agent {result_PP.failed_agent}")
else:
    print(f"\nPlanning succeeded:")
    for agent_id, path in result_PP.paths.items():
        print(f"  Agent {agent_id}: {len(path)} steps -> {path}")


# Simulation with PP

env_PP = MAPFEnvironment(instance)
history_PP = env_PP.run_plan(result_PP)

print(f"\nSimulation completed in {len(history_PP) - 1} timesteps")
for snapshot in history_PP:
    print(f"  t={snapshot['timestep']}: {snapshot['positions']}"
          f"  {'✓ DONE' if snapshot['done'] else ''}")

animate_paths(G, result_PP)


# Simulation with CBSNode

env_CBS = MAPFEnvironment(instance)
history_CBS = env_CBS.run_plan(result_CBS)

print(f"\nSimulation completed in {len(history_CBS) - 1} timesteps")
for snapshot in history_CBS:
    print(f"  t={snapshot['timestep']}: {snapshot['positions']}"
          f"  {'✓ DONE' if snapshot['done'] else ''}")

animate_paths(G, result_CBS)


# Performance A_star vs PP vs CBS

print("Performance MAPFs")
print("Prioritized Planner")
print("A* path: independent shortest path (no conflict avoidance)")
 
total_mapf_cost_PP  = 0
total_mapf_cost_CBS  = 0
total_a_star_cost = 0
 
for agent in instance.fleet.agents.values():
 
    mapf_path_original_PP = result_PP.path_original_for(agent.id)
    mapf_cost_PP = len(mapf_path_original_PP) - 1   # number of steps for reaching the goal with PP

    mapf_path_original_CBS = result_CBS.path_original_for(agent.id)
    mapf_cost_CBS = len(mapf_path_original_CBS) - 1   # number of steps for reaching the goal with CBS
 
    a_star_path = a_star(instance.graph, agent.start, agent.goal, extended=False)
    a_star_cost = len(a_star_path) - 1 if a_star_path is not None else float("inf")
 
    total_mapf_cost_PP  += mapf_cost_PP
    total_a_star_cost += a_star_cost
    total_mapf_cost_CBS  += mapf_cost_CBS
 
    extra_step_PP = mapf_cost_PP - a_star_cost     # extra step for avoiding conflicts
    extra_step_CBS = mapf_cost_CBS - a_star_cost     # extra step for avoiding conflicts

    print(f"\n  Agent {agent.id}")
    print(f"    Start : {agent.start}   Goal : {agent.goal}")
    print(f"    MAPF path PP   ({mapf_cost_PP:2d} steps) : {mapf_path_original_PP}")
    print(f"    MAPF path CBS   ({mapf_cost_PP:2d} steps) : {mapf_path_original_CBS}")
    if a_star_path is not None:
        print(f"    A* path ({a_star_cost:2d} steps) : {a_star_path}")
    else:
        print(f"    A* path : unreachable")
    print(f"    Overhead due to conflict avoidance in PP: +{extra_step_PP} step(s)")
    print(f"    Overhead due to conflict avoidance in CBS: +{extra_step_CBS} step(s)")
    print()
 
print()
print("SUMMARY (sum over all agents)")
print(f"  Total MAPF PP cost  (sum of path lengths) : {total_mapf_cost_PP }")
print(f"  Total MAPF BFS cost  (sum of path lengths) : {total_mapf_cost_CBS }")
print(f"  Total a* cost (sum of path lengths) : {total_a_star_cost}")