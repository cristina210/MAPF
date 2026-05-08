from Network_graph import NetworkGraph
from graph_utils.time_graph_builder import time_expansion_graph_with_constr


class TimeExpandedGraph:
    """
    An object of this class rapresent a time-expanded NetworkGraph.
    With attributes related to the original graph, the horizon of extension and the extended graph.
    Provides utility methods to map between original and expanded node ids and viceversa
    and to rebuild the extended graph taking consideration constraints of reservation of nodes/edges on it.

    Note: original node indices must be consecutive starting from 0.
    """
    def __init__(self, G: NetworkGraph, T: int, vertex_constraints: set = None, edge_constraints: set = None):
        """
        Args:
            G: original directed NetworkGraph
            T: number of timesteps in the time-expanded graph
            vertex_constraints: set of expanded node ids that cannot be visited in time extended graph (if no costraints -> empty set)
            edge_constraints: set of (src, dst) expanded edge pairs that cannot be used in time extended graph (if no costraints -> empty set)
        """
        self.G_original = G
        self.T = T
        self.vertex_constraints = vertex_constraints if vertex_constraints is not None else set()
        self.edge_constraints = edge_constraints if edge_constraints is not None else set()
        # build the time-expanded graph and store the mapping original_id -> list of expanded ids
        self.G_expanded, self.old_id_to_new = time_expansion_graph_with_constr(G, T, self.vertex_constraints, self.edge_constraints)
        # G is the resulting time extended graph and old_id_to_new is a list containing for each original id of G_original the corresponding expanded
        # nodes in G_expanded (also the ones involved in constraints)
        node_ids = sorted(G.nodes())
        assert node_ids == list(range(len(node_ids))), \
            "Node ids must be consecutive integers starting from 0"
    def get_expanded_id(self, original_id: int, t: int) -> int:
        """
        Return the expanded node id corresponding to original_id at timestep t.

        Args:
            original_id: node id in the original graph
            t: timestep
        Returns:
            expanded node id
        """
        return self.old_id_to_new[original_id][t]

    def get_original_id(self, expanded_id: int) -> int:
        """
        Return the original node id corresponding to an expanded node id.

        Args:
            expanded_id: node id in the time-expanded graph
        Returns:
            original node id
        """
        return self.G_expanded.nodes[expanded_id]["original_id"]  

    def add_vertex_constraint(self, *nodes: int) -> None:
        """
        Add one or more vertex constraints on the time-expanded graph.
        The nodes (identified by their expanded node ids) cannot be visited
        by subsequent agents during planning.

        Args:
            nodes: one or more expanded node ids to forbid
        Usage:
            teg.add_vertex_constraint(8)          # single node
            teg.add_vertex_constraint(8, 9, 10)   # multiple nodes
        """
        for node in nodes:
            self.vertex_constraints.add(node)

    def add_edge_constraint(self, *edges: tuple) -> None:
        """
        Add one or more edge constraints on the time-expanded graph.
        The edges (identified by expanded node ids) cannot be used
        by subsequent agents during planning in the time extended graph.

        Args:
            edges: one or more (src, dst) tuples to forbid
            teg.add_edge_constraint((14, 15))              # single edge
            teg.add_edge_constraint((14, 15), (16, 17))   # multiple edges
        """
        for edge in edges:
            self.edge_constraints.add(edge)

    def add_constraints_from_path(self, path: list) -> None:
        """
        Extract and add vertex and edge constraints from a planned path.
        All nodes and edges in the path are forbidden for subsequent agents.
        Note: path must be expressed in terms of expanded node ids
        (node ids in the time-expanded graph, not in the original graph).

        Args:
            path: list of expanded node ids representing the planned path
        """
        self.add_vertex_constraint(*path)
        for i in range(len(path) - 1):
            self.add_edge_constraint((path[i], path[i+1]))

    def rebuild_with_constraints(self, vertex_constraints = None, edge_constraints = None) -> None:
        """
        Rebuild the time-expanded graph eventually enforcing constraints.

        Args:
            vertex_constraints: set of expanded node ids that cannot be visited
            edge_constraints: set of (src, dst) expanded edge pairs that cannot be used
        """
        if vertex_constraints is not None:
            self.vertex_constraints = vertex_constraints
        if edge_constraints is not None:
            self.edge_constraints = edge_constraints

        # ensure they are initialized
        if self.vertex_constraints is None:
            self.vertex_constraints = set()
        if self.edge_constraints is None:
            self.edge_constraints = set()

        self.G_expanded, self.old_id_to_new = time_expansion_graph_with_constr(self.G_original, self.T, vertex_constraints, edge_constraints)
    
    def update_teg_with_adding_constraints(self, new_vertex_constr, new_edge_constr) -> None:
        """
        Incrementally updates the existing expanded graph by adding new constraints.
        Instead of rebuilding everything, it removes the newly forbidden nodes/edges 
        from the current graph structure. Efficient for sequential multi-agent planning.

        Args:
            new_vertex_constr: new expanded nodes to forbid and remove
            new_edge_constr: new expanded edges to forbid and remove
        """
        self.add_vertex_constraint(*new_vertex_constr)
        self.add_edge_constraint(*new_edge_constr)
        self.G_expanded = self._remove_nodes_edges_from_graph(self.G_expanded, new_vertex_constr, new_edge_constr)  

    def _remove_nodes_edges_from_graph(self, G, new_vertex_constr, new_edge_constr) -> NetworkGraph:
        G_new = NetworkGraph()
        for id_node, attrs in G.nodes(data=True):
            if id_node not in new_vertex_constr:
                G_new.add_node(id_node, **attrs)
        for src, dst, attrs in G.edges(data=True):
            if src not in new_vertex_constr and dst not in new_vertex_constr and (src, dst) not in new_edge_constr:
                G_new.add_edge(src, dst, **attrs)
        return G_new

    @staticmethod
    def compute_expanded_id(original_id: int, t: int, T: int) -> int:      # works only if id_expanded are built sequentially as in time_graph_builder
        return original_id * T + t

    @staticmethod  
    def compute_original_id(expanded_id: int, T: int) -> int:           # works only if id_expanded are built sequentially as in time_graph_builder
        return expanded_id // T
                

