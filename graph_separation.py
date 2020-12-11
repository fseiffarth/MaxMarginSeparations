import random
import time

from joblib import Parallel, delayed

import GraphNumpyFunctions
from DataToSQL.DataToSQL import DataToSQL
from Draw import draw_graph_with_labels, draw_graph_with_labels_training
from graph_generation import graph_generation
from Graph_Fuctions import random_graph
import MaximumMarginSeparation as mms
import numpy as np


def node_prediction_graphs(node_number, edge_density, seed, training_size, plot=False):
    """
    Runs maximum margin algorithm for node prediction on a random graph with specified node_number and edge density
    :param node_number: number of nodes of the graph
    :param edge_density: density of the edges
    :param training_size: size of the nodes where the node label is known
    :param seed: random seed
    :param plot: boolean variable if the graph should be plotted
    :return:
    """
    graph = random_graph(node_number=node_number, edge_density=edge_density, plot=False)
    points = random.sample(range(0, node_number), 2)
    graph_data = graph_generation(color_method="simultaneous_bfs", color_input=points, graph=graph, color_seed=seed)
    if plot:
        draw_graph_with_labels(graph_data)
    # graph_data = Graph_Generation(color_method="random_grow_coloring", color_input=[set(red), set(green)], graph=graph, color_seed=seed)
    # draw_graph_with_labels(graph_data)

    linkage_function = lambda nodes, element: GraphNumpyFunctions.graph_distance_linkage(graph, nodes, element)
    closure_operator = lambda x: GraphNumpyFunctions.graph_closure(graph, x)

    separation = mms.MaximumMarginSeparation(graph, closure_operator, linkage_function)

    target_set_a = list(graph_data.green_nodes)
    target_set_b = list(graph_data.red_nodes)


    start = time.time()
    classification = -1
    while classification == -1:
        set_a = random.sample(target_set_a, training_size)
        set_b = random.sample(target_set_b, training_size)

        graph_data.learning_green_nodes = set_a
        graph_data.learning_red_nodes = set_b
        graph_data.blue_nodes = [x for x in range(0, len(target_set_a) + len(target_set_b)) if
                                 x not in set_a and x not in set_b]
        if plot:
            draw_graph_with_labels_training(graph_data)
        classification = separation.max_margin_classification(set_a, set_b, target_set_a, target_set_b)

    graph_data.learning_green_nodes = set(np.nonzero(separation.closed_set_a)[0])
    graph_data.learning_red_nodes = set(np.nonzero(separation.closed_set_b)[0])

    print(separation.closed_set_a, separation.closed_set_b)
    print(np.logical_and(separation.closed_set_a, separation.closed_set_b))

    graph_data.blue_nodes = set(np.nonzero(np.logical_not(np.logical_or(separation.closed_set_b, separation.closed_set_a)))[0])
    if plot:
        draw_graph_with_labels_training(graph_data)
    print("Classification Time: ", time.time() - start)

    green_correct = classification[0]
    red_correct = classification[1]

    size_a = len(set_a)
    size_b = len(set_b)

    closed_set_a_size = len(np.nonzero(separation.closed_set_a)[0])
    closed_set_b_size = len(np.nonzero(separation.closed_set_b)[0])

    target_set_a_size = len(target_set_a)
    target_set_b_size = len(target_set_b)

    correct = green_correct + red_correct - size_a - size_b

    accuracy = (green_correct + red_correct) / (closed_set_a_size + closed_set_b_size)

    coverage = (closed_set_a_size + closed_set_b_size) / node_number

    unclassified = node_number - (closed_set_a_size + closed_set_b_size)

    entries = [accuracy, coverage, correct, unclassified, red_correct, green_correct, size_a, size_b, target_set_a_size,
               target_set_b_size, node_number, edge_density]

    return entries


def run_single_experiment(dataset, columns, column_types, num_nodes, edge_density, seed, train_set_size,
                              plot=False):
    entries = node_prediction_graphs(num_nodes, edge_density, seed, train_set_size)
    dataset.experiment_to_database("ECML2020", columns, [entries], column_types)
