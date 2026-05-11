# MAPF — Multi-Agent Path Finding

A Python framework for planning collision-free paths for multiple agents moving on a graph. Implements three solvers (PP, CBS, BCBS) on exploting a Time-Expanded Graph.

---

## Example of use

```python
from Network_graph import NetworkGraph
from graph_utils.grid_builder import make_grid_graph
from fleet_utils.fleet_utils import make_random_fleet
from instance import MAPFInstance
from MAPF_algorithm.solvers.CBS import CBSSolver

G  = make_grid_graph(rows=5, cols=5)
fleet = make_random_fleet(G, num_agents=4)
instance = MAPFInstance(graph=G, fleet=fleet)

solver = CBSSolver(inst.graph, instance.T)
result = solver.plan(instance.fleet)

print(result)
print("Sum of costs:", result.total_cost(instance.graph))
print("Makespan: ", result.makespan())
```

---

## Project structure

```
MAPF/
├── Network_graph.py              # NetworkGraph (wraps nx.DiGraph)
├── fleet.py                      # Agent, Fleet
├── instance.py                   # MAPFInstance — bundles graph + fleet + time horizon
├── environment.py                # MAPFEnvironment — step-by-step plan execution
├── extended_time_graph.py        # TimeExpandedGraph
│
├── graph_utils/
│   ├── grid_builder.py           # make_grid_graph
│   ├── graph_viz.py              # plotting and animation helpers
│   ├── graph_analysis.py         # graph_difference, precompute_shortest_paths
│   └── time_graph_builder.py     # TEG construction and constraint helpers
│
├── fleet_utils/
│   └── fleet_utils.py            # make_random_fleet
│
├── shortest_path_algorithm/
│   └── A_star.py                 # a_star, a_star_with_focal_search, heuristics
│
├── MAPF_algorithm/
│   ├── plan_result.py            # PlanResult — solver output and metrics
│   └── solvers/
│       ├── PP.py                 # PrioritizedPlanner
│       ├── CBS.py                # CBSSolver
│       └── BCBS.py               # BCBSSolver
│
├── conflict_heuristics.py        # h1/h2/h3 for high - and low-level focal search
├── results_handler.py            # print_comparison, save_comparison
└── main.py                      
```

---

## Solvers

```python
from MAPF_algorithm.solvers.PP   import PrioritizedPlanner
from MAPF_algorithm.solvers.CBS  import CBSSolver
from MAPF_algorithm.solvers.BCBS import BCBSSolver

pp   = PrioritizedPlanner(inst.graph, inst.T)
cbs  = CBSSolver(inst.graph, inst.T)
bcbs = BCBSSolver(inst.graph, inst.T, w_l=2, w_h=2)   # BCBS; w_h>1, w_l=1 → ECBS
```

---

## Result metrics

```python
result.makespan()                          # timestep when the last agent arrives
result.total_cost(inst.graph)              # sum-of-costs across all agents
result.path_length_time(agent_id)          # timesteps for one agent (including waits)
result.path_length_space(agent_id, G)      # physical distance (moves only)
result.solver_stats                        # runtime, expanded nodes, tree depth, …
```

---

## Key concepts

**Time-Expanded Graph (TEG)** — the original graph replicated across `T` timesteps. Agents plan on the TEG so waiting and time-dependent conflicts are handled.

**Time horizon `T`** — use in TEG - computed automatically in `MAPFInstance` as `T_min + congestion_term`, where `T_min` is the longest individual shortest path. Can be overridden if needed.

**Conflict types** — vertex (two agents on the same node at the same timestep) and swap (two agents exchanging positions in one step). Both are detected and resolved by CBS-based solvers.

**Sum of costs** — the default objective: sum of individual path costs (sum of sequence of  weights related to both wait and move action)

---

## Graph constraints

- Node ids must be consecutive integers starting from 0.
- Edge weights must be uniform (one timestep = one edge traversal).
- Currently supported: grid graphs via `make_grid_graph(rows, cols, step)`. External graph import can be added in `graph_utils/upload_graph.py` provided the two constraints above are satisfied.

---

## Dependencies

pip install -r requirements.txt
