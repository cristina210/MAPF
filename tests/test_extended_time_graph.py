"""
Tests for TimeExpandedGraph.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extended_time_graph import TimeExpandedGraph
from graph_utils.upload_graph import make_grid_graph
from shortest_path_algorithm.A_star import a_star

passed = 0
failed = 0

def check(condition, msg):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {msg}")
    else:
        failed += 1
        print(f"  [FAIL] {msg}")

def section(title):
    print(f"\n--- {title} ---")

def exp(orig, t):
    """Shorthand: expanded id of (original_node, timestep)."""
    return TimeExpandedGraph.compute_expanded_id(orig, t, T)

G = make_grid_graph(rows=3, cols=3, step=1.0)
T = 4

# ── 1. unconstrained TEG ──────────────────────────────────────────────────────

section("1. Unconstrained TEG")
teg = TimeExpandedGraph(G, T)

check(len(teg.G_expanded.nodes) == 36, "36 nodes (9 original * 4 timesteps)")
check(len(teg.G_expanded.edges) == 99, "99 edges (27 wait + 72 move)")

# ── 2. id conversion ──────────────────────────────────────────────────────────

section("2. Id conversion: original <-> expanded")

# node 5 at t=2 -> expanded id = 5*4+2 = 22 -> back to original = 5
check(exp(5, 2) == 22,                                    "exp(5, t=2) = 22")
check(TimeExpandedGraph.compute_original_id(22, T) == 5,  "original_id(22) = 5")
check(teg.get_expanded_id(5, t=2) == 22,                  "get_expanded_id(5, t=2) = 22")
check(teg.get_original_id(22) == 5,                       "get_original_id(22) = 5")

# ── 3. vertex constraint ──────────────────────────────────────────────────────

section("3. Vertex constraint: block node 4 at t=1")

# node 4 at t=1 -> expanded id = 4*4+1 = 17
teg_v = TimeExpandedGraph(G, T, vertex_constraints={17})

check(17 not in teg_v.G_expanded.nodes,   "node 17 (orig=4, t=1) removed")
check(len(teg_v.G_expanded.nodes) == 35,  "35 nodes remaining")

# no edge should touch node 17
incident = [(u,v) for u,v in teg_v.G_expanded.edges() if u==17 or v==17]
check(len(incident) == 0, "no edges incident to blocked node 17")

# node 4 at other timesteps should still exist
check(exp(4,0) in teg_v.G_expanded.nodes, "node 4 at t=0 still present")
check(exp(4,2) in teg_v.G_expanded.nodes, "node 4 at t=2 still present")

# ── 4. edge constraint and swap symmetry ─────────────────────────────────────

section("4. Edge constraint: block move 0->1 at t=0, check swap 1->0 also blocked")

# forward arc: node 0 at t=0 -> node 1 at t=1  (ids: 0 -> 5)
# swap arc:    node 1 at t=0 -> node 0 at t=1  (ids: 4 -> 1)
forward = (exp(0,0), exp(1,1))   # (0, 5)
swap    = (exp(1,0), exp(0,1))   # (4, 1)

teg_e = TimeExpandedGraph(G, T, edge_constraints={forward})

check(forward not in teg_e.G_expanded.edges(), f"forward arc {forward} blocked")
check(swap    not in teg_e.G_expanded.edges(), f"swap arc {swap} also blocked automatically")

# same move at a different timestep should still exist
other = (exp(0,1), exp(1,2))   # (1, 6)
check(other in teg_e.G_expanded.edges(), f"same move at t=1 still allowed {other}")

# ── 5. rebuild_with_constraints ──────────────────────────────────────────────

section("5. rebuild_with_constraints")

teg_r = TimeExpandedGraph(G, T)
teg_r.rebuild_with_constraints(vertex_constraints={exp(0,1)})   # block node 0 at t=1

check(exp(0,1) not in teg_r.G_expanded.nodes, "node blocked after rebuild")
check(len(teg_r.G_expanded.nodes) == 35,      "35 nodes after rebuild")

# rebuild with no constraints restores original state
teg_r.rebuild_with_constraints()
check(len(teg_r.G_expanded.nodes) == 36, "36 nodes restored after empty rebuild")
check(len(teg_r.G_expanded.edges) == 99, "99 edges restored after empty rebuild")

# ── 6. incremental update ─────────────────────────────────────────────────────

section("6. update_teg_with_adding_constraints (incremental)")

teg_u = TimeExpandedGraph(G, T)
teg_u.update_teg_with_adding_constraints(
    new_vertex_constr = {exp(8, 0)},        # block node 8 at t=0
    new_edge_constr   = {(exp(2,0), exp(1,1))},   # block move 2->1 at t=0
)

check(exp(8,0) not in teg_u.G_expanded.nodes,          "node 8 at t=0 removed incrementally")
check((exp(2,0), exp(1,1)) not in teg_u.G_expanded.edges(), "edge 2->1 at t=0 removed")
check(exp(8,0) in teg_u.vertex_constraints,             "vertex constraint recorded internally")

# ── 7. A* on constrained TEG ─────────────────────────────────────────────────

section("7. A* pathfinding on constrained TEG")

# longer horizon needed for complex routing
T_nav = 10
teg_nav = TimeExpandedGraph(G, T_nav)
start   = teg_nav.get_expanded_id(0, t=0)

# basic: path from 0 to 8 exists
path = a_star(teg_nav.G_expanded, start, goal=8, extended=True)
check(path is not None, "A* finds path 0->8 on unconstrained TEG")

# block centre node (4) at all timesteps — path must go around
centre_blocked = {TimeExpandedGraph.compute_expanded_id(4, t, T_nav) for t in range(T_nav)}
teg_b  = TimeExpandedGraph(G, T_nav, vertex_constraints=centre_blocked)
path_b = a_star(teg_b.G_expanded, teg_b.get_expanded_id(0, t=0), goal=8, extended=True)
check(path_b is not None, "A* finds alternative path when centre blocked")
if path_b:
    orig_path = [TimeExpandedGraph.compute_original_id(n, T_nav) for n in path_b]
    check(4 not in orig_path, f"path avoids node 4: {orig_path}")

# block everything except start — no path possible
all_but_zero = {TimeExpandedGraph.compute_expanded_id(n, t, T_nav)
                for n in range(9) for t in range(T_nav) if n != 0}
teg_f  = TimeExpandedGraph(G, T_nav, vertex_constraints=all_but_zero)
path_f = a_star(teg_f.G_expanded, teg_f.get_expanded_id(0, t=0), goal=8, extended=True)
check(path_f is None, "A* returns None when no path exists")

# ── summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*40}")
print(f"  PASSED: {passed}   FAILED: {failed}")
print(f"{'='*40}")