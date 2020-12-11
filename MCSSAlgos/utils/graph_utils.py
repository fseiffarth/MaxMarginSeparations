import random

import matplotlib
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

"""Generate connected random graph of size n from random tree of size n"""


def random_graph(node_number, edge_density, seed=None, is_tree=False, plot=False):
    if seed is not None:
        graph = nx.random_tree(node_number, seed=seed)
    else:
        graph = nx.random_tree(node_number)
    if not is_tree:
        counter = 0
        if seed is not None:
            random.seed(seed)
        while 1 + counter / graph.number_of_nodes() < edge_density:
            v = random.randint(0, graph.number_of_nodes() - 1)
            w = random.randint(0, graph.number_of_nodes() - 1)
            if w != v and not graph.has_edge(v, w):
                graph.add_edge(v, w)
                counter += 1
    if plot:
        pos = graphviz_layout(graph, prog='neato')
        nx.draw(graph, pos)
        nx.draw_networkx_labels(graph, pos, nodelist=graph.nodes())
        matplotlib.pyplot.show()

    return graph
