'''
Created on 07.11.2018

@author: florian
'''
import random
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout



# from pygame.tests.test_utils.png import itertools


# calculates the closure of the nodes from node_list in the graph
def graph_closure(Graph, nodes):
    closure = set()
    node_pairs = set(combinations(nodes, 2))
    print(node_pairs)
    for pair in node_pairs:
        # there is a path from
        if nx.has_path(Graph, pair[0], pair[1]):
            gen = nx.all_shortest_paths(Graph, pair[0], pair[1])
            # print(list(gen))
            for path in gen:
                for elem in path:
                    closure.add(elem)
    return closure


# the greedy algorithm from the paper, applied to graphs
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


def random_graph(node_number, edge_density, seed=0, plot=False):
    T = nx.random_tree(node_number, seed=seed)
    counter = 0
    while 1 + counter / T.number_of_nodes() < edge_density:
        v = random.randint(0, T.number_of_nodes() - 1)
        w = random.randint(0, T.number_of_nodes() - 1)
        if w != v and not T.has_edge(v, w):
            T.add_edge(v, w)
            counter += 1
    if plot:
        pos = graphviz_layout(T, prog='neato')
        nx.draw(T, pos)
        nx.draw_networkx_labels(T, pos, nodelist=T.nodes())
        plt.show()

    return T
