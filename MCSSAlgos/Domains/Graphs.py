import networkx as nx

from MCSSAlgos.Domains.DomainBase import Domain


class Graph(Domain):
    def __init__(self, graph_object: nx.Graph, all_paths: bool = False):
        super(Graph, self).__init__(graph_object, graph_object.number_of_nodes())
        self.is_tree = nx.is_tree(graph_object)
        self.all_shortest_paths = {}
        if all_paths:
            self.all_shortest_paths = dict(nx.all_pairs_shortest_path_length(self.data_object))

    def get_elements(self) -> set:
        return set(self.data_object.nodes)
