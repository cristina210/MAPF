# MAPF Project Documentation

## Overview

**Multi-Agent Path Finding (MAPF)** is the problem of finding collision-free paths for a set of agents moving on a graph, where each agent must reach its assigned goal node from its starting node.

At each timestep, each agent can either:
- **Wait**: remain on the current node (wait action).
- **Move**: traverse an edge to an adjacent node (move action).

Collisions between agents come in two forms:
- **Vertex conflict**: two agents occupy the same node at the same timestep.
- **Edge conflict (swap)**: two agents swap positions between two adjacent nodes in the same timestep interval.

Since agents move on a discrete graph, time is also treated as discrete: one edge traversal costs exactly one timestep. All references to time in this project mean **timesteps**, not continuous time.

To handle waiting actions and time-dependent conflicts, a **Time-Expanded Graph (TEG)** is constructed — see the dedicated section below.

The objective function used in this project is the **Sum of Costs**: the sum of individual path costs across all agents, where:
- **Move cost**: the weight of the traversed edge, defined in the graph during the uploading and stored in the TEG.
- **Wait cost**: defined internally in `time_graph_builder.py / time_expansion_graph_with_constr`. Default value is `1` per timestep.

---

## Key Objects

### Graph — `NetworkGraph`

The graph is implemented as a directed graph using the `NetworkX` library (`nx.DiGraph`).
By default each node stores:
- A unique integer `id`
- Spatial coordinates `x`, `y`
- Additional attributes (type of nodes for example: waypoints, pick point, ... ecc) can be stored using the `NetworkX` attribute system.

Each edge indentified by two endpoints stores:
- `weight`: traversal cost (cost of the move action along that edge).

**Mandatory constraints:**
- Node ids should be consecutive integers starting from 0 (`0, 1, 2, ..., N-1`). This is required by the TEG builder and by the static conversion methods `compute_expanded_id` / `compute_original_id`. Note: This requirement should not be limitation considering that names and other properties can be stored as node attributes.
- Edge weights must be uniform (same value for all edges). MAPF solvers in this project assume discrete, uniform timesteps — one edge traversal costs exactly one timestep. Graphs with variable edge lengths would violate this assumption.

**Currently supported graph types:**
- Grid graphs via `make_grid_graph(rows, cols, step)` — graph is a grid and all edge weights equal `step` (default `1.0`).
- External graph import is possible (to implement in *upload_graph.py*), provided the two constraints above are satisfied.

> **Future expansion**: implement a discretization method in `NetworkGraph` that approximates an arbitrary graph by rounding edge lengths to multiples of a base unit, then subdividing edges to produce a uniform-weight graph.

*For details: `Network_graph.py`, `upload_graph.py`*

---

### Fleet and Agents — `Fleet`, `Agent`

An `Agent` represents a single entity (e.g. an AMR) and is defined by default by: an`id`- unique integer identifier - , `start`- Starting node id in the graph - , `goal` - Goal node id (`None` if idle) -,  `state`: `"busy"` if a goal is assigned, `"idle"` otherwise.
State can be extended adding for example `in charging` and an attribute relative to charge can be add as well.
A `Fleet` is a collection of agents stored in a dictionary `{agent_id: Agent}`. It provides methods to retrieve agents, update goals, and filter by state (`idle_agents()`, `busy_agents()`).
Agent ids should be assigned sequentially from `0` to `num_agents - 1`.

**Currently supported fleet creation:**
- Random fleet via `make_random_fleet(G, num_agents)`, which assigns random start/goal pairs ensuring no shared starting nodes and no shared goal nodes.

*For details: `fleet.py`, `fleet_utils/fleet_utils.py`*

---

### Instance — `MAPFInstance`

`MAPFInstance` collects all the static data that define a MAPF problem. Can be passed to any solver without modification.
Attribute:  
- `graph`: The `NetworkGraph` on which agents move 
-  `fleet`:  The `Fleet` of agents with their start/goal assignments 
| `T_min`: Minimum feasible time horizon: length of the longest individual shortest path across all agents
| `T`:  Actual time horizon used by planners: `T_min + congestion_term`  

**Note on T**: the TEG is built up to timestep `T`. If `T` is too small, some agents may have no feasible path in the TEG and planning fails. If `T` is too large, memory and computation are wasted. The current formula `T = T_min + congestion_term` where T_min is the makespan if each agent plan itself indipendently from others is a heuristic that can be overwrite externally. `congestion_term` take into consideration that the time needed MAPF problem increase with congestion: therefore increase when the number of agents increase and/or the number of edges decrease. Actually, congestions highly depends on the structure of the graph: for example in a Barbell graph the makespan can be very high because one single edge connect two connected components.  [corso dinamiche su network vedi cheeger inequality/conduttività/tempo di rilassamento grafo forse dà info su questo].

**Validation** performed automatically at creation:
- Every agent's `start` and `goal` node exist in the graph.
- No two agents share the same starting node.
- A path from `start` to `goal` exists for every active agent.
- `T >= T_min` for every active agent.

```python
instance = MAPFInstance(graph=G, fleet=fleet)
# T is computed automatically: T = T_min + num_agents
# but can be overwrite
```

*For details: `instance.py`*

---

### Time-Expanded Graph — `TimeExpandedGraph`

The Time-Expanded Graph (TEG) is a directed graph that makes the time dimension explicit. The class `TimeExpandedGraph` rapresents this time extension owning as attributes: the original and the expanded graph, the time horizon and the possible constraints. Includes method for adding constraints, updating teg adding constraints or for rebuilding the time extended graph from scratch.

There are two main functions for built a teg:
- `time_expansion_graph_with_constr`: the TEG is rebuilt from scratch with the possibility to impose constraints.
 - `update_teg_with_adding_constraints`: The incremental update method that modify an existence teg imposing new constraints passed in input.

Each node `(original_id, t)` in the original graph is replicated `T` times (horizon of expansion of teg), one per timestep, and assigned a unique expanded integer id:
```
expanded_id = original_id * T + t
```
This formula is deterministic and enables static conversion without a TEG instance:
- `TimeExpandedGraph.compute_expanded_id(original_id, t, T)` → expanded id
- `TimeExpandedGraph.compute_original_id(expanded_id, T)` → original id
This conversion is valid only when graph and T don't change.  

Two types of edges are added:
- **Wait edges**: `(node, t) → (node, t+1)` — agent stays in place for one timestep. Weight: `1` (default).
- **Move edges**: `(u, t) → (v, t+1)` — agent moves from `u` to `v`. Weight: same as the original edge weight.

**Constraint handling**: the TEG supports two types of constraints, used by CBS-based solvers to enforce conflict-free replanning:
- **Vertex constraints**: a set of expanded node ids to exclude — their incident edges are also omitted.
- **Edge constraints**: a set of `(src, dst)` expanded edge pairs to exclude. Swap-pair inverses (the reverse swap edge) are added automatically via `create_edges_constraints`.

Imposing constraints on a teg means considering only a subset of nodes/edges of the full teg.

```python                   
teg = TimeExpandedGraph(G, T, vertex_constraints, edge_constraints)  # constrained
```

*For details: `extended_time_graph.py`, `time_graph_builder.py`*


> **Choice of model**: agents that reach their goal before the last agent are assumed to "disappear" from the simulation so doesn't interfere anymore with other agents (are not constraints anymore). Si potrebbe pensare di trattare in modo diverso, a quel punto il constraint sull'ultimo nodo deve "espandersi" e creare vertex constraint su quel nodo fino a fine time horizon.

---

### Plan Result — `PlanResult`

`PlanResult` is the standardised output of all MAPF solvers. It stores the planned paths for each agent and provides metrics for solution analysis and solver comparison.
Among attributes: Success`: True` if planning succeeded for all agents 
- `failed_agent`: Id of the agent for which planning failed (`None` if success) 
- `paths` :` path in the TEG 
- `solver_stats`: solver statistics (runtime, expanded nodes, etc.)

**Some of the available metrics:**
-`makespan()`: Timestep at which the last agent reaches its goal 
-`path_length_time(agent_id)`: Number of timesteps for a single agent (including waits)
-`path_length_space(agent_id, G)`: Physical distance on the original graph (move steps only, no waits) 
-`compute_path_cost(agent_id, G_expanded, T)` | Sum of edge weights including wait steps: cost of solution in MAPF|
- `total_cost(G_expanded, T)` | Sum-of-costs across all agents |

*For details: `MAPF_algorithm/plan_result.py`*

---

### Environment — `MAPFEnvironment`

`MAPFEnvironment` is a simulation environment that executes a MAPF plan step by step, tracking the position of each agent over time.

It takes a `MAPFInstance` (static problem description) and a `PlanResult` (planned paths) as input, and produces a history of snapshots — one per timestep — each containing: `timestep` and `positions`.

Key methods:
- `run_plan(plan_result)`: executes the full plan and returns the history.
- `_reset()`: resets positions to start nodes and clears the history.

> **Choice of model**: agents that reach their goal before the last agent are assumed to remain on their goal node for all subsequent timesteps. This is consistent with the TEG model (the goal node acts as a sink).

*For details: `environment.py`*

---

## Shortest Path Algorithms — `A_star.py`

Shortest path algorithms find the minimum-cost path between a start and a goal node in a graph. In this project they serve as the low-level planner inside MAPF solvers.

### Standard A* — `a_star`

A* guides the search from start to goal using two components:
- `g(n)`: best known cost from start to node `n`.
- `h(n)`: admissible heuristic estimate of the cost from `n` to goal.
- `f(n) = g(n) + h(n)`: estimated total cost through `n`.

Available heuristics (passed via the `heuristic` parameter):
- `h_manhattan` (default): Manhattan distance — optimal for orthogonal grids with unit weights.
- `h_euclidean`: Euclidean distance — admissible only if edge weights equal spatial distances.
Note: the optimality of solution depends on the type of heuristic, graph and type of cost (for using Manhattan or Euclidean, cost of traversing an edge should be related to edge lenght).

### Focal A* — `a_star_with_focal_search`

A bounded-suboptimal variant that maintains two lists:
- **OPEN**: standard A* min-heap ordered by `f = g + h`.
- **FOCAL**: subset of OPEN with `f <= w * f_min`, ordered by `h_c` — the accumulated congestion cost along the path (number of other agents passing through each visited node).

At each iteration, the node with the lowest conflict cost is extracted from FOCAL instead of the node with the lowest `f`. This biases the search towards less conflicted paths while guaranteeing that the returned solution has cost at most `w * C*`.
Used as the low-level planner in BCBS (`w = w_l`).

*For details: `shortest_path_algorithm/A_star.py`*

---

## MAPF Solvers

MAPF solvers differ in three key properties:
- **Completeness**: Does the solver always find a solution if one exists? 
- **Optimality**: Does the solver always return a minimum-cost solution? 
- **Efficiency**: How fast does the solver run in practice? 

All solvers take a `MAPFInstance` (or its components) and return a `PlanResult`. Solver performance statistics are stored in `result.solver_stats`.

---

### Prioritized Planning — `PrioritizedPlanner`

Agents are planned one at a time in a fixed priority order (the insertion order in `fleet.agents` by default). Each planned path is converted into vertex and edge constraints for all subsequent agents, ensuring the later agents avoid the paths already committed. Prioritized planning is not complete neither optimal but it is fast.

```python
planner = PrioritizedPlanner(instance.graph, instance.T)
result  = planner.plan(instance.fleet)
```

*For details: `MAPF_algorithm/solvers/PP.py`*

---

### Conflict-Based Search — `CBSSolver`

CBS is a two-level algorithm:
**High level** — explores a **Constraint Tree (CT)** in best-first order (lowest sum-of-costs first):
1. The root node has no constraints; each agent gets its individual shortest path.
2. At each iteration, the lowest-cost CT node is expanded.
3. If the solution is conflict-free, it is returned (optimality guaranteed by best-first).
4. Otherwise, the first detected conflict is resolved by splitting the node into two children — one adding a constraint for each of the two conflicting agents.
**Low level** — for each agent, a new TEG is built with that agent's constraints and standard A* is run to find the shortest feasible path. CBS is optimal, complete but can suffer high computational cost.

```python
planner = CBSSolver(instance.graph, instance.T)
result  = planner.plan(instance.fleet)
```

*For details: `MAPF_algorithm/solvers/CBS.py`*

---

### Bounded-Suboptimal CBS — `BCBSSolver`

BCBS extends CBS by applying focal search at both levels, trading optimality for speed. It is parameterised by two suboptimality factors:
- `w_l >= 1`: low-level suboptimality — each individual path is explored via focal search.
- `w_h >= 1`: high-level suboptimality — the CT is explored via focal search.
BCBS(`w_l=1, w_h=1`) = Standard CBS
BCBS(`w_l=1, w_h>1`) = ECBS (Enhanced CBS)

**Cost guarantee**: sum-of-costs `<= w_h * w_l * C*`.


```python
planner = BCBSSolver(instance.graph, instance.T, w_l=3, w_h=1)  # ECBS-like
result  = planner.plan(instance.fleet)
```

*For details: `MAPF_algorithm/solvers/BCBS.py`*


## Project Structure

```
MAPF/
├── Network_graph.py                  # NetworkGraph class
├── fleet.py                          # Agent and Fleet classes
├── instance.py                       # MAPFInstance class
├── environment.py                    # MAPFEnvironment class
├── extended_time_graph.py            # TimeExpandedGraph class
│
├── graph_utils/
│   ├── grid_builder.py               # make_grid_graph
│   ├── graph_viz.py                  # plot_graph, animate_paths, print helpers
│   ├── graph_analysis.py             # graph_difference, precompute_shortest_paths
│   └── time_graph_builder.py         # time_expansion_graph_with_constr, build_teg_mappings
│
├── fleet_utils/
│   └── fleet_utils.py                # make_random_fleet
│
├── shortest_path_algorithm/
│   └── A_star.py                     # a_star, a_star_with_focal_search, heuristics
│
├── MAPF_algorithm/
│   ├── plan_result.py                # PlanResult class
│   └── solvers/
│       ├── PP.py                     # PrioritizedPlanner
│       ├── CBS.py                    # CBSSolver
│       └── BCBS.py                   # BCBSSolver
│
├── results_handler.py                # print_comparison, save_comparison
├── conflict_heuristics.py            # HEURISTICS_HIGH_L dict, h3_num_conflicts
└── main.py                           # entry point
```

### Bibliography:
Multi-Agent Path Finding – An Overview - Roni Stern
Suboptimal Variants of the Conflict-Based Search Algorithm for the Multi-Agent Pathfinding Problem - Max Barer, Guni Sharon, Roni Stern, Ariel Felner

