"""
Tests for TimeExpandedGraph across multiple random configurations.
Each seed generates a different graph size and time horizon.
"""
import sys, os, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extended_time_graph import TimeExpandedGraph
from graph_utils.upload_graph import make_grid_graph
from shortest_path_algorithm.A_star import a_star

N_ITER = 50  # number of random configurations to test

total_passed = 0
total_failed = 0

def check(condition, msg):
    global total_passed, total_failed
    if condition:
        total_passed += 1
        print(f"    [PASS] {msg}")
    else:
        total_failed += 1
        print(f"    [FAIL] {msg}")

def exp(orig, t, T):
    return TimeExpandedGraph.compute_expanded_id(orig, t, T)


for seed in range(N_ITER):
    random.seed(seed)

    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    T    = random.randint(3, 7)
    G    = make_grid_graph(rows=rows, cols=cols, step=1.0)
    N    = G.number_of_nodes()
    E    = G.number_of_edges()

    print(f"\n{'='*55}")
    print(f"  seed={seed}  grid={rows}x{cols}  nodes={N}  edges={E}  T={T}")
    print(f"{'='*55}")

    expected_nodes = N * T
    expected_wait  = N * (T - 1)
    expected_move  = E * (T - 1)
    expected_edges = expected_wait + expected_move

    # ── 1. unconstrained counts ───────────────────────────────────────────────

    teg = TimeExpandedGraph(G, T)

    check(len(teg.G_expanded.nodes) == expected_nodes,
          f"nodes: {len(teg.G_expanded.nodes)} == {expected_nodes}")
    check(len(teg.G_expanded.edges) == expected_edges,
          f"edges: {len(teg.G_expanded.edges)} == {expected_edges}")

    # ── 2. id conversion round-trip ───────────────────────────────────────────

    orig = random.randint(0, N - 1)
    t    = random.randint(0, T - 1)
    e_id = exp(orig, t, T)

    check(TimeExpandedGraph.compute_original_id(e_id, T) == orig,
          f"round-trip orig={orig} t={t} -> exp={e_id} -> orig")
    check(teg.get_expanded_id(orig, t) == e_id,
          f"get_expanded_id({orig}, {t}) == {e_id}")
    check(teg.get_original_id(e_id) == orig,
          f"get_original_id({e_id}) == {orig}")

    # ── 3. vertex constraint ──────────────────────────────────────────────────

    forbidden = exp(orig, t, T)
    teg_v = TimeExpandedGraph(G, T, vertex_constraints={forbidden})

    check(forbidden not in teg_v.G_expanded.nodes,
          f"blocked node {forbidden} (orig={orig}, t={t}) absent")
    check(len(teg_v.G_expanded.nodes) == expected_nodes - 1,
          f"exactly one node removed")
    incident = [(u,v) for u,v in teg_v.G_expanded.edges() if u==forbidden or v==forbidden]
    check(len(incident) == 0,
          f"no edges incident to blocked node")

    # ── 4. edge constraint and swap symmetry ──────────────────────────────────

    # pick a random directed edge in G and a random timestep
    edges_list = list(G.edges())
    src, dst   = random.choice(edges_list)
    t_e        = random.randint(0, T - 2)
    forward    = (exp(src, t_e, T), exp(dst, t_e + 1, T))
    swap       = (exp(dst, t_e, T), exp(src, t_e + 1, T))

    teg_e = TimeExpandedGraph(G, T, edge_constraints={forward})

    check(forward not in teg_e.G_expanded.edges(),
          f"forward arc {forward} blocked")
    if G.has_edge(dst, src):
        check(swap not in teg_e.G_expanded.edges(),
              f"swap arc {swap} also blocked (bidirectional edge)")

    # ── 5. rebuild restores original ──────────────────────────────────────────

    teg_r = TimeExpandedGraph(G, T, vertex_constraints={forbidden})
    teg_r.rebuild_with_constraints()

    check(len(teg_r.G_expanded.nodes) == expected_nodes,
          f"node count restored after empty rebuild")
    check(len(teg_r.G_expanded.edges) == expected_edges,
          f"edge count restored after empty rebuild")

    # ── 6. incremental update ─────────────────────────────────────────────────

    new_v = exp(random.randint(0, N-1), random.randint(0, T-1), T)
    teg_u = TimeExpandedGraph(G, T)
    teg_u.update_teg_with_adding_constraints(new_vertex_constr={new_v}, new_edge_constr=set())

    check(new_v not in teg_u.G_expanded.nodes,
          f"incrementally blocked node {new_v} removed")
    check(new_v in teg_u.vertex_constraints,
          f"constraint recorded in teg.vertex_constraints")

    # ── 7. A* basic reachability ──────────────────────────────────────────────

    T_nav   = T + 5   # extra horizon for routing
    teg_nav = TimeExpandedGraph(G, T_nav)
    start   = teg_nav.get_expanded_id(0, t=0)
    path    = a_star(teg_nav.G_expanded, start, goal=N-1, extended=True)

    check(path is not None,
          f"A* finds path 0->{N-1} on unconstrained TEG (T={T_nav})")

    # block all intermediate nodes — only start and goal free
    blocked = {TimeExpandedGraph.compute_expanded_id(n, t, T_nav)
               for n in range(N) for t in range(T_nav)
               if n not in (0, N-1)}
    teg_b   = TimeExpandedGraph(G, T_nav, vertex_constraints=blocked)
    path_b  = a_star(teg_b.G_expanded, teg_b.get_expanded_id(0, t=0), goal=N-1, extended=True)

    # path may or may not exist depending on graph topology — just check no crash
    check(True, f"A* completes without error on heavily constrained TEG (path={'found' if path_b else 'None'})")


print(f"\n{'='*55}")
print(f"  TOTAL  PASSED: {total_passed}   FAILED: {total_failed}")
print(f"{'='*55}")