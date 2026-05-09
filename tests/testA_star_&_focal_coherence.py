"""
Test: compare a_star vs a_star_with_focal_search over N random instances.
Verifies that with w=1, focal search always returns a path of the same cost as A*.
"""
import sys, os, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_utils.upload_graph import make_grid_graph
from fleet_utils.upload_fleet import make_random_fleet
from instance import MAPFInstance
from extended_time_graph import TimeExpandedGraph
from shortest_path_algorithm.A_star import a_star, a_star_with_focal_search

random.seed(42)

G        = make_grid_graph(rows=4, cols=4, step=1.0)
fleet    = make_random_fleet(G, num_agents=9)
T        = MAPFInstance(graph=G, fleet=fleet).T
N_ITER   = 500
MAX_V    = 8
MAX_E    = 4
MAX_CONG = 10

stats = {"ok": 0, "no_path": 0, "diff_cost": 0, "only_astar": 0, "only_focal": 0}

teg_clean    = TimeExpandedGraph(G, T)
all_expanded = list(range(len(G.nodes) * T))
all_edges    = list(teg_clean.swap_pairs.keys())

for i in range(N_ITER):
    agent = random.sample(list(fleet.agents.values()), 2)[0]

    start_exp_clean = TimeExpandedGraph.compute_expanded_id(agent.start, 0, T)
    v_constr  = set(random.sample([n for n in all_expanded if n != start_exp_clean],  k=random.randint(0, MAX_V)))
    e_constr = set(random.sample(all_edges, k=min(random.randint(0, MAX_E), len(all_edges))))

    teg = TimeExpandedGraph(G, T, v_constr, e_constr)
    start_exp = teg.get_expanded_id(agent.start, t=0)

    cong_dict = {n: random.randint(1, 3) for n in random.sample(all_expanded, k=random.randint(0, MAX_CONG))}

    path_astar = a_star(teg.G_expanded, start_exp, agent.goal, extended=True)
    path_focal = a_star_with_focal_search(teg.G_expanded, start_exp, agent.goal, congestion_dict=cong_dict, extended=True, w=1)

    cost_a = len(path_astar) - 1 if path_astar else None
    cost_f = len(path_focal) - 1 if path_focal else None

    if   path_astar is None and path_focal is None:  stats["no_path"]    += 1
    elif path_astar is None:
        stats["only_focal"] += 1
        print(f"[{i:03d}] BUG: focal found path but A* did not — agent={agent.id}")
    elif path_focal is None:
        stats["only_astar"] += 1
        print(f"[{i:03d}] BUG: A* found path but focal did not — agent={agent.id}")
    elif cost_a != cost_f:
        stats["diff_cost"] += 1
        print(f"[{i:03d}] BUG: cost A*={cost_a} focal={cost_f} — agent={agent.id}")
        print(f" A*: {[TimeExpandedGraph.compute_original_id(n, T) for n in path_astar]}")
        print(f"focal: {[TimeExpandedGraph.compute_original_id(n, T) for n in path_focal]}")
    else:
        stats["ok"] += 1

print(f"\n{'='*50}\n  RESULTS over {N_ITER} iterations\n{'='*50}")
for k, v in stats.items():
    print(f"  {k:<20}: {v}")

if stats["diff_cost"] == 0 and stats["only_astar"] == 0 and stats["only_focal"] == 0:
    print("\n  a_star_with_focal_search(w=1) is consistent with a_star.")
else:
    print("\n  There are bugs in a_star_with_focal_search.")