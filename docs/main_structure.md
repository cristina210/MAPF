# MAPF Project Documentation

## Overview

**Multi-Agent Path Finding (MAPF)** is the problem of finding collision-free paths for a set of agents moving on a graph, where each agent must reach its assigned goal node from its starting node.

Collisions between agents come in two forms:
- **Vertex conflict**: two agents occupy the same node at the same timestep.
- **Edge conflict (swap)**: two agents swap positions between two adjacent nodes in the same timestep interval.

Since agents move on a discrete graph, time is also treated as discrete: agents move one edge per timestep, or stay in place (wait action). From this point on, all references to time mean **timesteps**, not continuous time.

---

## Core Objects

### Graph — `NetworkGraph`

The graph is implemented as a directed graph using the `NetworkX` library (`nx.DiGraph`), wrapped in a custom `NetworkGraph` class. By default, each node stores:
- A unique integer `id`
- Spatial coordinates `x`, `y`

Edges store a `weight` attribute representing the traversal cost. In classic MAPF problem weight = 1 for all agents.

**Constraints on the graph (mandatory):**
- Node ids must be **consecutive integers starting from 0** (i.e., `0, 1, 2, ..., N-1`). This is required by the time-expanded graph builder and by the static conversion methods `compute_expanded_id` / `compute_original_id`. This requirement should not be limitation considering that name or other properties can be store as attributes thanks to the `NetworkX` graph structure.
- Edge weights must be **uniform** (same length for all edges). MAPF solvers in this project assume discrete, uniform timesteps — one edge traversal costs exactly one timestep. Graphs with variable edge lengths would violate this assumption. This can be a strong limitation: see future expansion

**Currently supported graph types:**
- Grid graphs via `make_grid_graph(rows, cols, step)`.
- External graph import is possible, provided the two constraints above are satisfied.

> **Future expansion**: implement a discretization method in `NetworkGraph` that approximates an arbitrary graph by rounding edge lengths to multiples of a base unit, then subdividing edges to produce a uniform-weight graph.

*For further details: `Network_graph.py`*

---

### Fleet and Agents — `Fleet`, `Agent`

An `Agent` represents a single mobile entity (e.g. an AMR) and is defined by:
- `id`: unique integer identifier.
- `start`: starting node id in the graph.
- `goal`: goal node id in the graph (`None` if the agent has no assigned goal — idle state).
- `state`: `"busy"` if a goal is assigned, `"idle"` otherwise.

A `Fleet` is a collection of agents stored in a dictionary `{agent_id: Agent}`. It provides methods to retrieve agents, update goals, and filter by state.
As in graph ids node agent ids should be assigned sequentially from `0` to `num_agents - 1`.

**Currently supported fleet creation:**
- Random fleet via `make_random_fleet(G, num_agents)`, which assigns random start/goal pairs ensuring no shared starting nodes and no shared goal nodes.

*For further details: `fleet.py`*

---

### Instance — `MAPFInstance`

`MAPFInstance` collects all the features that define a MAPF problem:

| Attribute | Description |
|-----------|-------------|
| `graph`   | The `NetworkGraph` on which agents move |
| `fleet`   | The `Fleet` of agents with their start/goal assignments |
| `T_min`   | Minimum feasible time horizon: the length of the longest individual shortest path across all agents (i.e., the makespan if all agents move independently without conflicts) |
| `T`       | Actual time horizon used by planners: `T_min` plus an overhead that accounts for congestion and conflict-avoidance detours |

**Why T matters**: the time-expanded graph (see below) is built up to timestep `T`. If `T` is too small, some agents may have no feasible path in the expanded graph and planning fails. If `T` is too large, memory and computation are wasted. The current formula for the overhead is a heuristic and may overestimate, see future expansions.

**Validation**: when an instance is created, the following checks are performed automatically:
- Every agent's `start` and `goal` node exist in the graph.
- No two agents share the same starting node.
- The number of agents does not exceed the number of graph nodes.
- A path from `start` to `goal` exists in the graph for every active agent.
- `T` is large enough for each agent to reach its goal (indipendently). This doesn't ensure to find the solution with MAPF algorithm.

```python
instance = MAPFInstance(graph=G, fleet=fleet)
```

*For further details: `instance.py`*

---

## Environment — `MAPFEnvironment`


---

## Shortest path search algorithm


---

## Solvers 


---

## Solvers 