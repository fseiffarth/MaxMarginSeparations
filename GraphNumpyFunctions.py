'''
Created on 07.11.2018

@author: florian
'''
import random
import numpy as np
import networkx as nx


def bfs_closure(graph, node, closure, new_closure_elements):
    lengths = nx.single_source_shortest_path_length(graph, node)
    current_elements = closure.copy()
    while len(np.nonzero(current_elements)[0]):
        for x in np.nonzero(current_elements)[0]:
            current_elements[x] = 0
            neighbors = graph.neighbors(x)
            for n in neighbors:
                if closure[n] == 0 and lengths[n] < lengths[x]:
                    current_elements[n] = 1
                    new_closure_elements[n] = 1
                    closure[n] = 1
    return closure, new_closure_elements


def graph_distance_linkage(graph, nodes, element):
    min_distance = graph.number_of_nodes()
    lengths = nx.single_source_shortest_path_length(graph, element)
    for node in np.nonzero(nodes)[0]:
        if lengths[node] < min_distance:
            min_distance = lengths[node]
    return min_distance


# calculates the closure of the nodes from node_list in the graph
def graph_closure(graph, nodes):
    closure = np.zeros(graph.number_of_nodes(), dtype=np.bool)
    np.put(closure, nodes, np.ones(len(nodes), dtype=np.bool))
    new_closure_elements = closure.copy()
    while len(np.nonzero(new_closure_elements)[0]):
        nodes = np.nonzero(new_closure_elements)[0]
        for node in nodes:
            new_closure_elements[node] = 0
            closure, new_closure_elements = bfs_closure(graph, node, closure, new_closure_elements)
    return closure


# the greedy algorithm from the papar, applied to graphs
def greedy_classifier_algorithm(Graph, start_nodes_red, start_nodes_green):
    closed_set_A = graph_closure(start_nodes_red)
    closed_set_B = graph_closure(start_nodes_green)
    if closed_set_A.intersection(closed_set_B):
        return False


def random_spanning_forest(Graph, seed=random.seed()):
    edge_list = list(Graph.edges())

    for edge in edge_list:
        w_edge = (edge[0], edge[1], random.uniform(0, 1))
        Graph.add_weighted_edges_from([w_edge])
    return nx.minimum_spanning_tree(Graph, algorithm="prim")
