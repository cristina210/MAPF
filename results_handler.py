from MAPF_algorithm.plan_result import PlanResult
from instance import MAPFInstance
from extended_time_graph import TimeExpandedGraph



def verify_solution(result: PlanResult) -> tuple[bool, list]:
    """
    Verifies that a MAPF solution is conflict-free.
    Agents disappear once they reach their goal:
    after time t >= len(path), they are no longer considered in the system.

    Checks:
    - vertex conflicts (same node at same timestep)
    - edge conflicts (swap between t and t+1), only if both agents exist at both times

    Args:
        result: PlanResult containing the planned paths in original node ids
    Returns:
        (True, [])                if no conflicts are found
        (False, conflict_list)    if conflicts are found.
    """
    paths = result.paths_original
    agent_ids = list(paths.keys())
    conflicts = []

    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            ai, aj = agent_ids[i], agent_ids[j]
            path_i, path_j = paths[ai], paths[aj]
            max_t = max(len(path_i), len(path_j))

            for t in range(max_t):

                # both agents must exist at time t
                if t >= len(path_i) or t >= len(path_j):
                    continue

                vi = path_i[t]
                vj = path_j[t]

                # vertex conflict
                if vi == vj:
                    conflicts.append({"type": "vertex","agents": (ai, aj), "timestep": t,"nodes":  (vi,) })

                # edge conflict: both agents must exist at t+1
                if (t + 1 < len(path_i)) and (t + 1 < len(path_j)):
                    vi_next = path_i[t + 1]
                    vj_next = path_j[t + 1]
                    if vi == vj_next and vj == vi_next:
                        conflicts.append({"type": "edge", "agents": (ai, aj), "timestep": t, "nodes":  (vi, vj)})

    return (len(conflicts) == 0), conflicts




def _build_per_agent_rows(agent, G_original, solvers: dict) -> tuple:
    headers = ["Algorithm", "Steps", "Distance", "Cost","Overhead steps", "Overhead cost", "Path"]
    rows    = []

    baseline_name  = next(iter(solvers))
    baseline_steps = solvers[baseline_name].path_length_time(agent.id)
    baseline_cost  = solvers[baseline_name].compute_path_cost(agent.id, G_original)

    for algo_name, result in solvers.items():
        steps = result.path_length_time(agent.id)
        dist  = result.path_length_space(agent.id, G_original)
        cost  = result.compute_path_cost(agent.id, G_original)
        path  = result.path_original_for(agent.id)

        if algo_name == baseline_name:
            overhead_steps = "—"
            overhead_cost  = "—"
        else:
            overhead_steps = (f"+{steps - baseline_steps}" if steps is not None and baseline_steps is not None else "—")
            overhead_cost  = (f"+{cost - baseline_cost:.2f}" if cost is not None and baseline_cost is not None else "—")

        rows.append([ algo_name, steps if steps is not None else "—",  f"{dist:.2f}"  if dist  is not None else "—", f"{cost:.2f}"  if cost  is not None else "—", overhead_steps,
            overhead_cost, path if path  is not None else "unreachable"])

    return headers, rows


def _build_aggregate_rows(G_original, solvers: dict) -> tuple:
    headers = ["Algorithm", "Makespan", "Total steps","Total distance", "Total cost"]
    rows    = []

    for algo_name, result in solvers.items():
        rows.append([algo_name,result.makespan(),result.cumulative_time_total(),f"{result.total_path_length_space(G_original):.2f}",f"{result.total_cost(G_original):.2f}" ])

    return headers, rows


def _write_comparison(instance, solvers: dict, write) -> None:
    validity = {name: verify_solution(result) for name, result in solvers.items()}

    # Per-agent results 
    write("\n" + "=" * 70 + "\n  PER-AGENT RESULTS\n" + "=" * 70 + "\n")
    for agent in instance.fleet.agents.values():
        write(f"\n  Agent {agent.id}  |  {agent.start} -> {agent.goal}\n")
        headers, rows = _build_per_agent_rows( agent, instance.graph, solvers)
        write(_format_table(headers, rows) + "\n")

    #  Aggregate results 
    write("\n" + "=" * 70 + "\n  AGGREGATE RESULTS\n" + "=" * 70 + "\n")
    headers, rows = _build_aggregate_rows(instance.graph, solvers)
    write(_format_table(headers, rows) + "\n")

    #  Solution validity 
    write("\n" + "=" * 70 + "\n  SOLUTION VALIDITY\n" + "=" * 70 + "\n")
    for algo_name, (is_valid, conflicts) in validity.items():
        if is_valid:
            write(f"\n  {algo_name:<20}: OK — no conflicts found\n")
        else:
            write(f"\n  {algo_name:<20}: FAIL — {len(conflicts)} conflict(s)\n")
            for c in conflicts:
                a1, a2 = c['agents']
                if c["type"] == "vertex":
                    write(f"[vertex] agents {a1},{a2} at node {c['nodes'][0]} t={c['timestep']}\n")
                else:
                    write(f"[edge]  agents {a1},{a2} swap {c['nodes'][0]}<->{c['nodes'][1]} t={c['timestep']}\n")

    # ── Solver statistics ──────────────────────────────────────────────────────
    write("\n" + "=" * 70 + "\n  SOLVER STATISTICS\n" + "=" * 70 + "\n")
    for algo_name, result in solvers.items():
        write(f"\n  {algo_name}:\n")
        for key, val in (result.solver_stats or {}).items():
            write(f" {key.replace('_',' ').capitalize():<35}: {val}\n")


def print_comparison(instance, solvers: dict) -> None:
    _write_comparison(instance, solvers, write=lambda s: print(s, end=""))


def save_comparison(instance, solvers: dict,
                    filepath: str = "results.txt") -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        _write_comparison(instance, solvers, write=f.write)



def save_instance(instance, filepath: str = "instance.txt"):
    with open(filepath, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("  INSTANCE: " + instance.name + "\n")
        f.write("=" * 70 + "\n")

        # graph
        f.write("\nGRAPH\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Nodes : {len(instance.graph.nodes)}\n")
        f.write(f"  Edges : {len(instance.graph.edges)}\n")
        f.write("\n  Node list (id | x | y):\n")
        for nid, attrs in instance.graph.nodes(data=True):
            f.write(f" {nid:>4} | x={attrs['x']:.2f} | y={attrs['y']:.2f}\n")
        f.write("\n  Edge list (src -> dst | weight):\n")
        for src, dst, attrs in instance.graph.edges(data=True):
            f.write(f" {src:>4} -> {dst:<4} | weight={attrs.get('weight', '—')}\n")

        # time horizon
        f.write("\nTIME\n")
        f.write("-" * 40 + "\n")
        f.write(f"  T_min (longest individual shortest path) : {instance.T_min}\n")
        f.write(f"  T  (time horizon used for planning)   : {instance.T}\n")

        # fleet
        f.write("\nFLEET\n")
        f.write("-" * 40 + "\n")
        f.write(f" Number of agents: {instance.fleet.num_agents()}\n\n")
        f.write(f"  {'ID':>4} | {'Start':>6} | {'Goal':>6} | {'Start (x,y)':>14} | {'Goal (x,y)':>14} | State\n")
        f.write("  " + "-" * 65 + "\n")
        for agent in instance.fleet.agents.values():
            start_attrs = instance.graph.nodes[agent.start]
            start_xy = f"({start_attrs['x']:.2f}, {start_attrs['y']:.2f})"
            if agent.goal is not None:
                goal_attrs = instance.graph.nodes[agent.goal]
                goal_xy  = f"({goal_attrs['x']:.2f}, {goal_attrs['y']:.2f})"
                goal_str = str(agent.goal)
            else:
                goal_xy = "—"
                goal_str = "—"
            f.write(f" {agent.id:>4} | {agent.start:>6} | {goal_str:>6} | {start_xy:>14} | {goal_xy:>14} | {agent.state}\n")


def save_simulation(history: list, filepath: str = "simulation_log.txt"):
    """
    Saves main characteristics of MAPF simulation in text file.
    """
    with open(filepath, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("  SIMULATION LOG\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Total timesteps: {len(history) - 1}\n\n")

        for snapshot in history:
            t  = snapshot["timestep"]
            done = snapshot["done"]
            pos  = snapshot["positions"]
            f.write(f"  t={t:>3}  {'[DONE]' if done else ''}\n")
            for agent_id, node in pos.items():
                f.write(f" Agent {agent_id:>3} -> node {node}\n")
            f.write("\n")



def _format_table(headers: list, rows: list) -> str:

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    def fmt_row(row):
        return "  " + "  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))

    separator = "  " + "  ".join("-" * w for w in col_widths)
    lines     = [fmt_row(headers), separator]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)