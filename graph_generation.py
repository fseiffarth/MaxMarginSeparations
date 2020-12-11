import csv
import os
import random
import sys

import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import time
import networkx as nx
import itertools

from networkx.drawing.nx_agraph import graphviz_layout


class graph_generation(object):
    '''
    classdocs
    '''

    def __init__(self, red_nodes=set(), green_nodes=set(), graph_seed=2, random_coloring=True,
                 color_method="random_spread", color_input=[], color_seed=0, graph=nx.Graph(), layout=False):
        '''
        Constructor: Color and classify a graph
        red_nodes: manually color nodes as red
        green_nodes: manually color nodes as green
        graph_seed: seed can be saved
        random_coloring: randomly coloring the nodes with color_method
        color_method: method for coloring the nodes
        color_input: input for color_method to color nodes
        color_seed: can be saved
        graph: graph to be colored
        layout: has to be true if graph should be printed
        '''

        """Methods for coloring of the graph"""

        """randomly set E to red a do spreading across the neighbours, color all other nodes in green"""

        def random_spread(self, input=[], seed=0):
            random.seed(seed)
            red_nodes = set()
            green_nodes = set()

            # set initial nodes to red
            for x in color_input[0]:
                red_nodes.add(x)
            # nodes which are new added to red
            new_nodes = list(red_nodes)

            # test neighbours of new nodes to be red unless new_nodes is empty
            counter = 0
            k = 2
            while new_nodes and counter < k:
                current_nodes = set()
                for node in new_nodes:
                    for neighbor in nx.all_neighbors(self.Graph, node):
                        if neighbor not in red_nodes and neighbor not in green_nodes:
                            rand_number = random.random()
                            if rand_number < color_input[1][0]:
                                current_nodes.add(neighbor)
                                red_nodes.add(neighbor)
                            else:
                                green_nodes.add(neighbor)
                new_nodes = list(current_nodes)
                counter += 1

            for x in range(nx.number_of_nodes(self.Graph)):
                if x not in red_nodes:
                    green_nodes.add(x)

            return red_nodes, green_nodes

        """make simultaneous bfs from nodes of the graph"""

        def simultaneous_bfs(self, input, seed=0, extra=1 / 4):
            class_size = 0
            while class_size < extra:
                red_nodes = np.zeros(self.Graph.number_of_nodes(), dtype=np.bool)
                distances_red = np.full(self.Graph.number_of_nodes(), -1, dtype=np.int64)
                green_nodes = np.zeros(self.Graph.number_of_nodes(), dtype=np.bool)
                distances_green = np.full(self.Graph.number_of_nodes(), -1, dtype=np.int64)

                red_nodes[input[0]] = 1
                distances_red[input[0]] = 0
                green_nodes[input[1]] = 1
                distances_green[input[1]] = 0

                current_distance = 0

                while len(np.where(distances_green == current_distance)[0]) > 0 or len(
                        np.where(distances_red == current_distance)[0]) > 0:
                    for x in np.where(distances_green == current_distance)[0]:
                        for neighbor in nx.neighbors(graph, x):
                            if distances_green[neighbor] == -1:
                                if distances_red[neighbor] <= current_distance and distances_red[neighbor] >= 0:
                                    distances_green[neighbor] = -2
                                else:
                                    distances_green[neighbor] = current_distance + 1
                    for x in np.where(distances_red == current_distance)[0]:
                        for neighbor in nx.neighbors(graph, x):
                            if distances_red[neighbor] == -1:
                                if distances_green[neighbor] == current_distance + 1:
                                    flip_coin = random.randint(0, 1)
                                    if flip_coin:
                                        distances_red[neighbor] = current_distance + 1
                                        distances_green[neighbor] = -2
                                    else:
                                        distances_red[neighbor] = -2
                                elif distances_green[neighbor] <= current_distance and distances_green[neighbor] >= 0:
                                    distances_red[neighbor] = -2
                                else:
                                    distances_red[neighbor] = current_distance + 1
                                    distances_green[neighbor] = -2
                    current_distance += 1

                distances_red = np.where(distances_red < 0, sys.maxsize, distances_red)
                distances_green = np.where(distances_green < 0, sys.maxsize, distances_green)

                red_nodes = np.where(distances_red < distances_green, True, False)
                green_nodes = np.where(distances_green < distances_red, True, False)

                class_size = min(len(np.where(red_nodes)[0]),
                                 len(np.where(green_nodes)[0])) / self.Graph.number_of_nodes()

                input = random.sample(range(0, self.Graph.number_of_nodes()), 2)

            red_nodes_set = set([x for x in range(0, len(red_nodes)) if red_nodes[x]])
            green_nodes_set = set([x for x in range(0, len(green_nodes)) if green_nodes[x]])

            return red_nodes_set, green_nodes_set

        # Label a graph with number of classes different labels
        def label_graph(classes, max_biggest_class_size=0.75):
            start_nodes = random.sample([x for x in range(0, self.Graph.number_of_nodes())], classes)
            biggest_class = 1
            while biggest_class > max_biggest_class_size:
                fixed_labels = np.empty(self.Graph.number_of_nodes(), dtype=np.int8)
                fixed_labels.fill(-1)
                current_labels = np.empty(self.Graph.number_of_nodes(), dtype=np.int8)
                current_labels.fill(-1)
                current_nodes = start_nodes.copy()

                for i, node in enumerate(start_nodes):
                    fixed_labels[node] = i

                while len(current_nodes) > 0:
                    nodes_step = current_nodes.copy()
                    current_nodes.clear()
                    for node in nodes_step:
                        for neighbor in nx.all_neighbors(graph, node):
                            if fixed_labels[neighbor] == -1:
                                if current_labels == -1:
                                    current_labels[neighbor] = current_labels[node]
                                else:
                                    rand_num = random.randint(0, 1)
                                    if rand_num:
                                        current_labels[neighbor] = current_labels[node]
                                current_nodes.append(neighbor)
                    fixed_labels[np.where(current_labels != -1)[0]] = current_labels[np.where(current_labels != -1)[0]]
            return fixed_labels

        """randomly set E to red and green and do spreading across the neighbours"""

        def double_random_spread(self, input=[], seed=0):
            random.seed(seed)
            red_nodes = set()
            green_nodes = set()

            # set initial nodes to red and green
            red_nodes.add(color_input[0][0])
            green_nodes.add(color_input[0][1])
            # nodes which are new added to red
            new_nodes_red = list(red_nodes)
            new_nodes_green = list(green_nodes)

            # test neighbours of new nodes to be red unless new_nodes is empty
            alternating = 0
            while new_nodes_red and new_nodes_green:
                current_nodes = set()
                if alternating:
                    for node in new_nodes_red:
                        for neighbor in nx.all_neighbors(self.Graph, node):
                            if neighbor not in red_nodes and neighbor not in green_nodes:
                                rand_number = random.random()
                                if rand_number < color_input[1][0]:
                                    current_nodes.add(neighbor)
                                    red_nodes.add(neighbor)
                                else:
                                    green_nodes.add(neighbor)
                    new_nodes_red = list(current_nodes)
                    alternating = 0
                else:
                    for node in new_nodes_green:
                        for neighbor in nx.all_neighbors(self.Graph, node):
                            if neighbor not in red_nodes and neighbor not in green_nodes:
                                rand_number = random.random()
                                if rand_number < color_input[1][1]:
                                    current_nodes.add(neighbor)
                                    red_nodes.add(neighbor)
                                else:
                                    green_nodes.add(neighbor)
                    new_nodes_green = list(current_nodes)
                    alternating = 1

            for x in range(nx.number_of_nodes(self.Graph)):
                if x not in red_nodes:
                    green_nodes.add(x)

            return red_nodes, green_nodes

        """randomly set E to red and do closure in the graph of this E, color all other nodes in green"""

        def closure_init(self, input=[], seed=0):
            random.seed(seed)
            red_nodes = set(random.sample(self.nodes, len(self.nodes) // 4))
            red_nodes = self.graph_closure(red_nodes)
            green_nodes = self.nodes.difference(red_nodes)

            return red_nodes, green_nodes

        """get random spanning forest of graph and color nodes by two partitions in the tree"""

        def random_tree_coloring(self, input=[], seed=0, print_steps=False):
            T = self.random_spanning_forest(seed=seed)
            T_Graph = graph_generation(random_coloring=False, random_generate=False, Graph=T)
            # T_Graph.draw_graph()
            random.seed(seed)

            red_nodes = set()
            green_nodes = set()

            # bfs search
            bfs_search_red = [0] * T.number_of_nodes();
            bfs_search_green = [0] * T.number_of_nodes();

            # find red and green node
            red_node = random.randint(0, T.number_of_nodes() - 1)
            green_node = random.randint(0, T.number_of_nodes() - 1)
            while green_node == red_node:
                green_node = random.randint(0, T.number_of_nodes() - 1)

            # check if there is a path
            while not nx.has_path(T, green_node, red_node):

                # find new nodes with path
                red_node = random.randint(0, T.number_of_nodes() - 1)

                green_node = random.randint(0, T.number_of_nodes() - 1)
                while green_node == red_node:
                    green_node = random.randint(0, T.number_of_nodes() - 1)

            # print steps only for testing
            if print_steps:
                # add red start point
                bfs_search_red[red_node] = 1
                red_nodes.add(red_node)
                T_Graph.set_node_class(red_nodes, "red")
                T_Graph.draw_graph()

                # add green start point
                bfs_search_green[green_node] = 1
                green_nodes.add(green_node)
                T_Graph.set_node_class(green_nodes, "green")
                T_Graph.draw_graph()

                # find border between green and red by middle of shortest path
                flip = random.randint(0, 1)
                if flip:
                    path_red_green = nx.shortest_path(T, green_node, red_node)
                    red_border = path_red_green[len(path_red_green) // 2]
                    green_border = path_red_green[len(path_red_green) // 2 - 1]

                else:
                    path_red_green = nx.shortest_path(T, red_node, green_node)
                    red_border = path_red_green[len(path_red_green) // 2 - 1]
                    green_border = path_red_green[len(path_red_green) // 2]

                # bfs from green
                current_nodes = [green_node]
                while current_nodes:
                    x = current_nodes.pop()
                    neighbours = nx.neighbors(T, x)
                    for n in neighbours:
                        if bfs_search_red[n] == 0 and bfs_search_green[n] == 0 and n != red_border:
                            current_nodes.append(n)
                            bfs_search_green[n] = 1
                            green_nodes.add(n)
                            T_Graph.set_node_class(green_nodes, "green")
                            T_Graph.draw_graph()

                # bfs from red
                current_nodes = [red_node]
                while current_nodes:
                    x = current_nodes.pop()
                    neighbours = nx.neighbors(T, x)
                    for n in neighbours:
                        if bfs_search_green[n] == 0 and bfs_search_red[n] == 0 and n != green_border:
                            current_nodes.append(n)
                            bfs_search_red[n] = 1
                            red_nodes.add(n)
                            T_Graph.set_node_class(red_nodes, "red")
                            T_Graph.draw_graph()
            else:
                # add red start point
                bfs_search_red[red_node] = 1
                red_nodes.add(red_node)

                # add green start point
                bfs_search_green[green_node] = 1
                green_nodes.add(green_node)

                # find border between green and red by middle of shortest path
                flip = random.randint(0, 1)
                if flip:
                    path_red_green = nx.shortest_path(T, green_node, red_node)
                    red_border = path_red_green[len(path_red_green) // 2]
                    green_border = path_red_green[len(path_red_green) // 2 - 1]

                else:
                    path_red_green = nx.shortest_path(T, red_node, green_node)
                    red_border = path_red_green[len(path_red_green) // 2 - 1]
                    green_border = path_red_green[len(path_red_green) // 2]

                # bfs from green
                current_nodes = [green_node]
                while current_nodes:
                    x = current_nodes.pop()
                    neighbours = nx.neighbors(T, x)
                    for n in neighbours:
                        if bfs_search_red[n] == 0 and bfs_search_green[n] == 0 and n != red_border:
                            current_nodes.append(n)
                            bfs_search_green[n] = 1
                            green_nodes.add(n)

                # bfs from red
                current_nodes = [red_node]
                while current_nodes:
                    x = current_nodes.pop()
                    neighbours = nx.neighbors(T, x)
                    for n in neighbours:
                        if bfs_search_green[n] == 0 and bfs_search_red[n] == 0 and n != green_border:
                            current_nodes.append(n)
                            bfs_search_red[n] = 1
                            red_nodes.add(n)

            return red_nodes, green_nodes

        """get random spanning forest of graph and color nodes by input nodes into partitions in the tree"""

        def random_tree_partition_coloring(self, input=[{0}, {1}], seed=0, print_steps=False):
            training_elements = input[0].union(input[1])
            T = self.random_spanning_forest()
            T_Graph = graph_generation(random_coloring=False, Graph=T)

            red_nodes = input[1]
            green_nodes = input[0]

            random.seed(seed)
            # print steps only for testing
            if print_steps:
                # TreeL = Graph_Generation(red_nodes= self.learning_red_nodes, green_nodes=self.learning_green_nodes, graph = T)
                T_Graph.set_classified_node_class(input[0], "green")
                T_Graph.set_classified_node_class(input[1], "red")
                T_Graph.draw_classified_graph()
                # TreeL.draw_classified_graph()
                training = [0] * self.Graph.number_of_nodes()
                for x in input[0]:
                    training[x] = 1
                for x in input[1]:
                    training[x] = -1

                bfs_search = [0] * self.Graph.number_of_nodes()

                training_set = random.sample(training_elements, len(training_elements))

                while training_set:
                    x = training_set.pop()
                    if bfs_search[x] == 0:
                        bfs_search[x] = 1

                        # node is in green training
                        if training[x] == 1:
                            red = False
                            current_red_nodes = []
                            current_green_nodes = [x]
                            T_Graph.set_classified_node_class({x}, "green")
                            while current_green_nodes:
                                neighbors = nx.neighbors(T, current_green_nodes.pop())
                                for n in neighbors:
                                    if (red and training[n] == -1) or bfs_search[n] == 1:
                                        continue
                                    elif training[n] == -1:
                                        red = True
                                        current_red_nodes.append(n)
                                        bfs_search[n] = 1
                                        T_Graph.set_classified_node_class({n}, "red")
                                        T_Graph.draw_classified_graph()
                                    else:
                                        current_green_nodes.append(n)
                                        bfs_search[n] = 1
                                        T_Graph.set_classified_node_class({n}, "green")
                                        T_Graph.draw_classified_graph()
                            while current_red_nodes:
                                neighbors = nx.neighbors(T, current_red_nodes.pop())
                                for n in neighbors:
                                    if training[n] == 1 or bfs_search[n] == 1:
                                        continue
                                    else:
                                        current_red_nodes.append(n)
                                        bfs_search[n] = 1
                                        T_Graph.set_classified_node_class({n}, "red")
                                        T_Graph.draw_classified_graph()

                        # node is in red training
                        if training[x] == -1:
                            green = False
                            current_green_nodes = []
                            current_red_nodes = [x]
                            T_Graph.set_classified_node_class({x}, "red")
                            while current_red_nodes:
                                neighbors = nx.neighbors(T, current_red_nodes.pop())
                                for n in neighbors:
                                    if (green and training[n] == 1) or bfs_search[n] == 1:
                                        continue
                                    elif training[n] == 1:
                                        green = True
                                        current_green_nodes.append(n)
                                        bfs_search[n] = 1
                                        T_Graph.set_classified_node_class({n}, "green")
                                        T_Graph.draw_classified_graph()

                                    else:
                                        current_red_nodes.append(n)
                                        bfs_search[n] = 1
                                        T_Graph.set_classified_node_class({n}, "red")
                                        T_Graph.draw_classified_graph()
                            while current_green_nodes:
                                neighbors = nx.neighbors(T, current_green_nodes.pop())
                                for n in neighbors:
                                    if training[n] == -1 or bfs_search[n] == 1:
                                        continue
                                    else:
                                        current_green_nodes.append(n)
                                        bfs_search[n] = 1
                                        T_Graph.set_classified_node_class({n}, "green")
                                        T_Graph.draw_classified_graph()

            else:
                training = [0] * self.Graph.number_of_nodes()
                for x in input[0]:
                    training[x] = 1
                for x in input[1]:
                    training[x] = -1

                bfs_search = [0] * self.Graph.number_of_nodes()

                training_set = random.sample(training_elements, len(training_elements))

                while training_set:
                    x = training_set.pop()
                    if bfs_search[x] == 0:
                        bfs_search[x] = 1

                        # node is in green training
                        if training[x] == 1:
                            red = False
                            current_red_nodes = []
                            current_green_nodes = [x]
                            while current_green_nodes:
                                neighbors = nx.neighbors(T, current_green_nodes.pop())
                                for n in neighbors:
                                    if (red and training[n] == -1) or bfs_search[n] == 1:
                                        continue
                                    elif training[n] == -1:
                                        red = True
                                        current_red_nodes.append(n)
                                        bfs_search[n] = 1
                                        red_nodes.add(n)
                                    else:
                                        current_green_nodes.append(n)
                                        bfs_search[n] = 1
                                        green_nodes.add(n)
                            while current_red_nodes:
                                neighbors = nx.neighbors(T, current_red_nodes.pop())
                                for n in neighbors:
                                    if training[n] == 1 or bfs_search[n] == 1:
                                        continue
                                    else:
                                        current_red_nodes.append(n)
                                        bfs_search[n] = 1
                                        red_nodes.add(n)

                        # node is in red training
                        if training[x] == -1:
                            green = False
                            current_green_nodes = []
                            current_red_nodes = [x]

                            while current_red_nodes:
                                neighbors = nx.neighbors(T, current_red_nodes.pop())
                                for n in neighbors:
                                    if (green and training[n] == 1) or bfs_search[n] == 1:
                                        continue
                                    elif training[n] == 1:
                                        green = True
                                        current_green_nodes.append(n)
                                        bfs_search[n] = 1
                                        green_nodes.add(n)
                                    else:
                                        current_red_nodes.append(n)
                                        bfs_search[n] = 1
                                        red_nodes.add(n)

                            while current_green_nodes:
                                neighbors = nx.neighbors(T, current_green_nodes.pop())
                                for n in neighbors:
                                    if training[n] == -1 or bfs_search[n] == 1:
                                        continue
                                    else:
                                        current_green_nodes.append(n)
                                        bfs_search[n] = 1
                                        green_nodes.add(n)

            return red_nodes, green_nodes

        """get random spanning forest of graph and color nodes by input nodes into partitions in the tree by simultaneously BFS from all input nodes"""

        def random_grow_coloring(self, input=[{0}, {1}], seed=0, print_steps=False):
            start_points = list(input[0].union(input[1]))
            T = self.random_spanning_forest()

            classes = [0] * len(self.nodes)
            green_nodes = set()
            red_nodes = set()
            for x in input[0]:
                green_nodes.add(x)
                classes[x] = 1
            for x in input[1]:
                red_nodes.add(x)
                classes[x] = -1

            bfs_search = [0] * len(self.nodes)
            current_nodes = []
            for point in start_points:
                bfs_search[point] = 1
                current_nodes.append([point])

            finished = False

            while not finished:
                finished = True
                counter = 0
                for p in start_points:
                    if current_nodes[counter]:
                        finished = False
                        neighbors = nx.neighbors(T, current_nodes[counter].pop())
                        for n in neighbors:
                            if bfs_search[n] == 0:
                                if classes[p] == 1:
                                    green_nodes.add(n)
                                else:
                                    red_nodes.add(n)
                                current_nodes[counter].append(n)
                                bfs_search[n] = 1
                    counter += 1

            return red_nodes, green_nodes

        """use greedy classifier for coloring in red and green given start nodes"""

        def tree_halfspace(self, input=[], seed=0):
            red_nodes = set({0})
            green_nodes = set({1})
            self.greedy_classifier_algorithm2(red_nodes, green_nodes, seed=seed)
            red_nodes = self.classified_red_nodes
            green_nodes = self.classified_green_nodes
            return red_nodes, green_nodes

        """Initialize the coloring and set all the class variables"""
        self.color_method_dict = {"random_spread_coloring": random_spread,
                                  "simultaneous_bfs": simultaneous_bfs,
                                  "double_random_spread_coloring": double_random_spread,
                                  "random_grow_coloring": random_grow_coloring, "closure_init_coloring": closure_init,
                                  "tree_halfspace": tree_halfspace,
                                  "random_tree_coloring": random_tree_coloring,
                                  "random_tree_partition_coloring": random_tree_partition_coloring,
                                  "label_graph": label_graph}
        # set graph
        self.Graph = graph

        # set the set of all nodes
        self.nodes = set(self.Graph.nodes())

        # set random seed
        self.graph_seed = graph_seed
        self.color_seed = color_seed

        # classified class labels of the nodes
        self.classified_red_nodes = set()
        self.classified_green_nodes = set()
        self.classified_blue_nodes = self.nodes

        self.learning_red_nodes = set()
        self.learning_green_nodes = set()
        self.learning_blue_nodes = set()

        self.training_red_nodes = set()
        self.training_green_nodes = set()
        self.training_blue_nodes = set()

        # other inputs
        self.color_method = color_method

        # set predefined class labels of the nodes
        if not random_coloring or red_nodes or green_nodes:
            self.red_nodes = red_nodes
            self.green_nodes = green_nodes
            self.blue_nodes = self.nodes.difference(self.red_nodes.union(self.green_nodes))
        else:
            self.red_nodes, self.green_nodes = self.color_method_dict[color_method](self, input=color_input,
                                                                                    seed=self.color_seed)

        # check if red_nodes and green_nodes are valid
        for x in red_nodes:
            try:
                x >= self.Graph.number_of_nodes()
                x < 0
                x in green_nodes
            except ValueError:
                print("The node coloring is not valid.")

        # set all nodes which are not red or green as blue
        self.blue_nodes = self.nodes.difference(self.red_nodes.union(self.green_nodes))

        # layout of graph
        if layout:
            # self.pos = nx.nx_agraph.graphviz_layout(self.graph)
            self.pos = nx.planar_layout(self.Graph)

        # number of initial set size after closure calculation
        self.initial_green = 0
        self.initial_red = 0

    """Random spanning forest of the graph"""

    def random_spanning_forest(self, seed=random.seed()):
        # make reproduceble examples
        if seed:
            random.seed(seed)
        else:
            random.seed()
        New_Graph = self.Graph
        edge_list = list(self.Graph.edges())

        for edge in edge_list:
            w_edge = (edge[0], edge[1], random.uniform(0, 1))
            self.Graph.add_weighted_edges_from([w_edge])
        return nx.minimum_spanning_tree(self.Graph, algorithm="kruskal")

    """gets the nodes of one specific class node_class 1 is green -1 is red 0 is blue  """

    def get_nodes_of_class(self, node_class):
        if node_class == 1 or node_class == "green":
            return self.green_nodes
        elif node_class == -1 or node_class == "red":
            return self.red_nodes
        elif node_class == 0 or node_class == "blue":
            return self.blue_nodes

    """gets the classified nodes of one specific class node_class 1 is green -1 is red 0 is blue    """

    def get_classified_nodes_of_class(self, node_class):
        if node_class == 1 or node_class == "green":
            return self.classified_green_nodes
        elif node_class == -1 or node_class == "red":
            return self.classified_red_nodes
        elif node_class == 0 or node_class == "blue":
            return self.classified_blue_nodes

    """gets the learning nodes of one specific class node_class 1 is green -1 is red 0 is blue    """

    def get_learning_nodes_of_class(self, node_class):
        if node_class == 1 or node_class == "green":
            return self.learning_green_nodes
        elif node_class == -1 or node_class == "red":
            return self.learning_red_nodes
        elif node_class == 0 or node_class == "blue":
            return self.learning_blue_nodes

    """gets the training nodes of one specific class node_class 1 is green -1 is red 0 is blue   """

    def get_training_nodes_of_class(self, node_class):
        if node_class == 1 or node_class == "green":
            return self.training_green_nodes
        elif node_class == -1 or node_class == "red":
            return self.training_red_nodes
        elif node_class == 0 or node_class == "blue":
            return self.training_blue_nodes

    """print the classification of the nodes"""

    def print_classified_nodes(self):
        print("Green: {0}, Red: {1}, Blue: {2}".format(self.get_classified_nodes_of_class("green"),
                                                       self.get_classified_nodes_of_class("red"),
                                                       self.get_classified_nodes_of_class("blue")))

    """print the training nodes"""

    def print_learning_nodes(self):
        print("Green: {0}, Red: {1}, Blue: {2}".format(self.get_learning_nodes_of_class("green"),
                                                       self.get_learning_nodes_of_class("red"),
                                                       self.get_learning_nodes_of_class("blue")))

    """print the training nodes"""

    def print_training_nodes(self):
        print("Green: {0}, Red: {1}, Blue: {2}".format(self.get_training_nodes_of_class("green"),
                                                       self.get_training_nodes_of_class("red"),
                                                       self.get_training_nodes_of_class("blue")))

    """set nodes to one specific class node_class 1 is green -1 is red 0 is blue"""

    def set_node_class(self, nodes, node_class):
        if node_class == 1 or node_class == "green":
            self.green_nodes = self.green_nodes.union(nodes)
            self.red_nodes = self.red_nodes.difference(nodes)
            self.blue_nodes = self.blue_nodes.difference(nodes)
        elif node_class == -1 or node_class == "red":
            self.red_nodes = self.red_nodes.union(nodes)
            self.green_nodes = self.green_nodes.difference(nodes)
            self.blue_nodes = self.blue_nodes.difference(nodes)
        elif node_class == 0 or node_class == "blue":
            self.blue_nodes = self.blue_nodes.union(nodes)
            self.red_nodes = self.red_nodes.difference(nodes)
            self.green_nodes = self.green_nodes.difference(nodes)

    """set nodes to one specific class node_class 1 is green -1 is red 0 is blue"""

    def set_classified_node_class(self, nodes, node_class):
        if node_class == 1 or node_class == "green":
            self.classified_green_nodes = self.classified_green_nodes.union(nodes)
            self.classified_red_nodes = self.classified_red_nodes.difference(nodes)
            self.classified_blue_nodes = self.classified_blue_nodes.difference(nodes)
        elif node_class == -1 or node_class == "red":
            self.classified_red_nodes = self.classified_red_nodes.union(nodes)
            self.classified_green_nodes = self.classified_green_nodes.difference(nodes)
            self.classified_blue_nodes = self.classified_blue_nodes.difference(nodes)
        elif node_class == 0 or node_class == "blue":
            self.classified_blue_nodes = self.classified_blue_nodes.union(nodes)
            self.classified_red_nodes = self.classified_red_nodes.difference(nodes)
            self.classified_green_nodes = self.classified_green_nodes.difference(nodes)

    """set nodes to one specific class node_class 1 is green -1 is red 0 is blue"""

    def set_learning_node_class(self, nodes, node_class):
        if node_class == 1 or node_class == "green":
            self.learning_green_nodes = self.learning_green_nodes.union(nodes)
            self.learning_red_nodes = self.learning_red_nodes.difference(nodes)
            self.learning_blue_nodes = self.learning_blue_nodes.difference(nodes)
        elif node_class == -1 or node_class == "red":
            self.learning_red_nodes = self.learning_red_nodes.union(nodes)
            self.learning_green_nodes = self.learning_green_nodes.difference(nodes)
            self.learning_blue_nodes = self.learning_blue_nodes.difference(nodes)
        elif node_class == 0 or node_class == "blue":
            self.learning_blue_nodes = self.learning_blue_nodes.union(nodes)
            self.learning_red_nodes = self.learning_red_nodes.difference(nodes)
            self.learning_green_nodes = self.learning_green_nodes.difference(nodes)

    """set nodes to one specific class node_class 1 is green -1 is red 0 is blue"""

    def set_training_node_class(self, nodes, node_class):
        if node_class == 1 or node_class == "green":
            self.training_green_nodes = self.training_green_nodes.union(nodes)
            self.training_red_nodes = self.training_red_nodes.difference(nodes)
            self.training_blue_nodes = self.training_blue_nodes.difference(nodes)
        elif node_class == -1 or node_class == "red":
            self.training_red_nodes = self.training_red_nodes.union(nodes)
            self.training_green_nodes = self.training_green_nodes.difference(nodes)
            self.training_blue_nodes = self.training_blue_nodes.difference(nodes)
        elif node_class == 0 or node_class == "blue":
            self.training_blue_nodes = self.training_blue_nodes.union(nodes)
            self.training_red_nodes = self.training_red_nodes.difference(nodes)
            self.training_green_nodes = self.training_green_nodes.difference(nodes)

    """choose randomly a training set of nodes  given the number of red and green training nodes"""

    def random_choose_learning_set(self, n_red=0, n_green=0, seed=0):
        # make reproduceble examples
        if seed:
            random.seed(seed)
        else:
            random.seed()

        self.set_learning_node_class(set(random.sample(self.green_nodes, n_green)), "green")
        self.set_learning_node_class(set(random.sample(self.red_nodes, n_red)), "red")
        self.learning_blue_nodes = self.nodes.difference(self.learning_green_nodes.union(self.learning_red_nodes))

    def random_choose_training_samples(self, n_train_samples, seed=0):
        # make reproduceble examples
        if seed:
            random.seed(seed)
        else:
            random.seed()

        train_a = random.sample(self.green_nodes, n_train_samples)
        train_b = random.sample(self.red_nodes, n_train_samples)

        for x in train_a:
            self.set_learning_node_class({x}, "green")
        for x in train_b:
            self.set_learning_node_class({x}, "red")

        self.learning_blue_nodes = self.nodes.difference(self.learning_green_nodes.union(self.learning_red_nodes))
        return len(self.learning_green_nodes), len(self.learning_red_nodes)

    """choose randomly a number of training nodes of all nodes"""

    def random_choose_training_set(self, n_train_examples, seed=0):
        # make reproduceble examples
        if seed:
            random.seed(seed)
        else:
            random.seed()

        train_nodes = random.sample(self.green_nodes.union(self.red_nodes), n_train_examples)
        for x in train_nodes:
            if x in self.green_nodes:
                self.set_learning_node_class({x}, "green")
            else:
                self.set_learning_node_class({x}, "red")

        self.learning_blue_nodes = self.nodes.difference(self.learning_green_nodes.union(self.learning_red_nodes))
        return len(self.learning_green_nodes), len(self.learning_red_nodes)

    """select a random number of nodes of a class 1 is green, -1 is red and 0 is blue  """

    def select_random_nodes_of_class(self, p=0.1, n=1, node_class=1, absolute=True, seed=random.seed()):
        # make reproduceble examples
        if seed:
            random.seed(seed)
        else:
            random.seed()
        if absolute:
            return random.sample(self.return_class(node_class), n)
        else:
            return random.sample(self.return_class(node_class), int(p * len(self.return_class(node_class))))

    def tree_closure(self, nodes):
        closure = nodes.copy()
        directed_tree = np.full(self.Graph.number_of_nodes(), -1)
        e = next(iter(closure))
        directed_tree[e] = e
        current_elements = [e]

        while current_elements:
            current_elem = current_elements.pop(0)
            neighbors = nx.neighbors(self.Graph, current_elem)
            for x in neighbors:
                if directed_tree[x] == -1:
                    current_elements.append(x)
                    directed_tree[x] = current_elem

        for node in nodes:
            x = node
            while directed_tree[x] not in closure:
                closure.add(x)
                x = directed_tree[x]

        return closure

    """calculates the closure of the nodes restricted to the set restriction given by the all shortest path closure(DOES ONLY WORK ON TREES)"""

    def graph_closure(self, nodes, restriction=set()):
        closure = nodes.copy()
        node_pairs = set(itertools.combinations(nodes, 2))

        if len(restriction):
            # print(node_pairs)
            for pair in node_pairs:
                # there is a path from
                if nx.has_path(self.Graph, pair[0], pair[1]):
                    gen = nx.all_shortest_paths(self.Graph, pair[0], pair[1])
                    # print(list(gen))
                    for path in gen:
                        for elem in path:
                            if elem in restriction:
                                closure.add(elem)
        else:
            # print(node_pairs)
            for pair in node_pairs:
                # there is a path from
                if nx.has_path(self.Graph, pair[0], pair[1]):
                    gen = nx.all_shortest_paths(self.Graph, pair[0], pair[1])
                    # print(list(gen))
                    for path in gen:
                        for elem in path:
                            closure.add(elem)
        return closure

    """the greedy algorithm from the paper, applied to graphs(DOES ONLY WORK ON TREES)"""

    def greedy_classifier_algorithm(self, start_nodes_red, start_nodes_green, restriction=set(), seed=random.seed()):
        # make reproduceble examples
        if seed:
            random.seed(seed)
        else:
            random.seed()

        closed_set_A = self.graph_closure(start_nodes_red, restriction)
        closed_set_B = self.graph_closure(start_nodes_green, restriction)
        if closed_set_A.intersection(closed_set_B):
            return False
        else:
            unclassified_nodes = list(self.nodes.difference(self.graph_closure(start_nodes_red, restriction).union(
                self.graph_closure(start_nodes_green, restriction))))
            while unclassified_nodes:
                closed_set_A.add(unclassified_nodes[0])
                closure = self.graph_closure(closed_set_A, restriction)
                if closure.intersection(closed_set_B):
                    closed_set_A.remove(unclassified_nodes[0])
                    closed_set_B.add(unclassified_nodes[0])
                    closure = self.graph_closure(closed_set_B, restriction)
                    if closure.intersection(closed_set_A):
                        closed_set_B.remove(unclassified_nodes[0])
                        unclassified_nodes.remove(unclassified_nodes[0])
                        continue
                    else:
                        closed_set_B = closure
                        for elem in closed_set_B:
                            if elem in unclassified_nodes:
                                unclassified_nodes.remove(elem)
                else:
                    closed_set_A = closure
                    for elem in closed_set_A:
                        if elem in unclassified_nodes:
                            unclassified_nodes.remove(elem)
        self.set_classified_node_class(closed_set_A, "red")
        self.set_classified_node_class(closed_set_B, "green")
        self.set_classified_node_class(self.nodes.difference(closed_set_A.union(closed_set_B)), "blue")

    "variant of the greedy algorithm from the paper, applied to graphs (DOES ONLY WORK FOR TREES)"""

    def greedy_classifier_algorithm2(self, start_nodes_red, start_nodes_green, restriction=set(), seed=random.seed()):
        # make reproduceble examples
        if seed:
            random.seed(seed)
        else:
            random.seed()

        closed_set_A = self.graph_closure(start_nodes_red, restriction)
        closed_set_B = self.graph_closure(start_nodes_green, restriction)
        if closed_set_A.intersection(closed_set_B):
            return False
        else:
            node_list = list(self.nodes.difference(self.graph_closure(start_nodes_red, restriction).union(
                self.graph_closure(start_nodes_green, restriction))))
            unclassified_nodes = random.sample(node_list, len(node_list))
            alter = 1
            while unclassified_nodes:

                # try to add alternately to setA or setB
                if alter % 2:
                    closed_set_A.add(unclassified_nodes.pop())
                    closure = self.graph_closure(closed_set_A, restriction)
                    if closure.intersection(closed_set_B):
                        closed_set_A.remove(unclassified_nodes[0])
                        closed_set_B.add(unclassified_nodes[0])
                        closure = self.graph_closure(closed_set_B, restriction)
                        if closure.intersection(closed_set_A):
                            closed_set_B.remove(unclassified_nodes[0])
                            continue
                        else:
                            closed_set_B = closure
                            for elem in closed_set_B:
                                if elem in unclassified_nodes:
                                    unclassified_nodes.remove(elem)
                    else:
                        closed_set_A = closure
                        for elem in closed_set_A:
                            if elem in unclassified_nodes:
                                unclassified_nodes.remove(elem)
                    alter = 0
                else:
                    closed_set_B.add(unclassified_nodes.pop())
                    closure = self.graph_closure(closed_set_B, restriction)
                    if closure.intersection(closed_set_A):
                        closed_set_B.remove(unclassified_nodes[0])
                        closed_set_A.add(unclassified_nodes[0])
                        closure = self.graph_closure(closed_set_A, restriction)
                        if closure.intersection(closed_set_B):
                            closed_set_A.remove(unclassified_nodes[0])
                            continue
                        else:
                            closed_set_A = closure
                            for elem in closed_set_A:
                                if elem in unclassified_nodes:
                                    unclassified_nodes.remove(elem)
                    else:
                        closed_set_B = closure
                        for elem in closed_set_B:
                            if elem in unclassified_nodes:
                                unclassified_nodes.remove(elem)
                    alter = 1

        self.set_classified_node_class(closed_set_A, "red")
        self.set_classified_node_class(closed_set_B, "green")
        self.set_classified_node_class(self.nodes.difference(closed_set_A.union(closed_set_B)), "blue")

    "majority vote classification per random spanning forests"""

    def majority_vote_classification(self, number_of_trees, restricted=set()):
        classification = {}
        for i in range(len(self.nodes)):
            classification[i] = [0, 0, 0]

        number_of_valid_trees = 0
        for i in range(number_of_trees):
            # print("{0}/{1}".format(i, number_of_trees))
            T = self.random_spanning_forest()
            T_Graph = graph_generation(random_generate=False, Graph=T)
            if T_Graph.greedy_classifier_algorithm2(self.learning_red_nodes, self.learning_green_nodes,
                                                    restricted):
                number_of_valid_trees += 1
                for x in T_Graph.classified_green_nodes:
                    classification[x][0] += 1
                for x in T_Graph.classified_red_nodes:
                    classification[x][1] += 1
                for x in T_Graph.classified_blue_nodes:
                    classification[x][2] += 1

        return classification

    """gives the precision of a classification"""

    def precision_of_classification(self):
        correct = 0
        wrong = 0
        unclassified = len(self.nodes)

        for x in self.classified_red_nodes:
            if x in self.red_nodes:
                correct += 1
                unclassified -= 1
            else:
                wrong += 1
                unclassified -= 1
        for x in self.classified_green_nodes:
            if x in self.green_nodes:
                correct += 1
                unclassified -= 1
            else:
                wrong += 1
                unclassified -= 1
        return correct, wrong, unclassified

    """returns all the classification error which occur"""

    def full_classification_error(self):
        red_correct = 0
        red_false = 0
        red_unclassified = 0
        green_correct = 0
        green_false = 0
        green_unclassified = 0

        for x in self.classified_red_nodes:
            if x in self.red_nodes:
                red_correct += 1
            elif x in self.green_nodes:
                red_false += 1
            else:
                red_unclassified += 1

        for x in self.classified_green_nodes:
            if x in self.green_nodes:
                green_correct += 1
            elif x in self.red_nodes:
                green_false += 1
            else:
                green_unclassified += 1

        return red_correct, red_false, red_unclassified, green_correct, green_false, green_unclassified

    """returns all nodes of a class"""

    def return_class(self, node_class):
        if node_class == 1 or node_class == "green":
            return self.green_nodes
        if node_class == -1 or node_class == "red":
            return self.red_nodes
        if node_class == 0 or node_class == "blue":
            return self.blue_nodes

    """print all labels of nodes of the graph"""

    def print_classes(self):
        print("Red nodes: {0}, Green nodes: {1}, Blue nodes: {2}".format(self.red_nodes, self.green_nodes,
                                                                         self.blue_nodes))

    """prints the training error"""

    def print_training_error(self):
        train_error = 0
        for x in self.learning_green_nodes:
            if x in self.classified_red_nodes:
                train_error += 1
        for x in self.learning_red_nodes:
            if x in self.classified_green_nodes:
                train_error += 1
        print("Absolute train error: {0}, relative train error: {1}".format(train_error, train_error / (
                len(self.learning_green_nodes) + len(self.learning_red_nodes))))

    """prints the test error"""

    def print_test_error(self):
        test_error = 0
        for x in self.classified_green_nodes:
            if x in self.red_nodes and x not in self.learning_green_nodes:
                test_error += 1
        for x in self.classified_red_nodes:
            if x in self.green_nodes and x not in self.learning_green_nodes:
                test_error += 1
        print("Absolute test error: {0}, relative test error: {1}".format(test_error, test_error / (
                len(self.classified_green_nodes.difference(self.learning_green_nodes)) + len(
            self.classified_red_nodes.difference(self.learning_red_nodes)))))

    """classifies the nodes given a classification dictionary"""

    def majority_vote_to_classification(self, classification):
        self.set_classified_node_class(self.nodes, "blue")
        for Key, Value in classification.items():
            if Value[0] > Value[1] and Value[0] > Value[2]:
                self.classified_green_nodes.add(Key)
            elif Value[0] < Value[1] and Value[1] > Value[2]:
                self.classified_red_nodes.add(Key)
            else:
                self.classified_blue_nodes.add(Key)

        self.classified_blue_nodes = self.nodes.difference(self.classified_green_nodes.union(self.classified_red_nodes))

    """classifies the nodes given a classification dictionary and a treshold"""

    def majority_vote_to_classification_red_green(self, classification, threshold=1):
        self.set_classified_node_class(self.nodes, "blue")
        for Key, Value in classification.items():
            if Value[0] > threshold * (Value[0] + Value[1] + Value[2]):
                self.classified_green_nodes.add(Key)
            elif Value[1] > threshold * (Value[0] + Value[1] + Value[2]):
                self.classified_red_nodes.add(Key)
            else:
                self.classified_blue_nodes.add(Key)

        self.classified_blue_nodes = self.nodes.difference(self.classified_green_nodes.union(self.classified_red_nodes))

    """use majority vote to define training samples"""

    def majority_vote_to_training(self, classification):
        self.set_classified_node_class(self.nodes, "blue")
        for Key, Value in classification.items():
            if Value[0] > Value[1] and Value[0] > Value[2]:
                self.training_green_nodes.add(Key)
            elif Value[0] < Value[1] and Value[1] > Value[2]:
                self.training_red_nodes.add(Key)
            else:
                self.training_blue_nodes.add(Key)

        self.training_blue_nodes = self.nodes.difference(self.training_green_nodes.union(self.training_red_nodes))

    """makes training samples from learning samples"""

    def make_training_samples(self, number_of_trees):
        restricted = self.learning_red_nodes.union(self.learning_green_nodes)
        start_nodes_red = set(self.learning_red_nodes[0])
        start_nodes_green = set(self.learning_green_nodes[0])
        return self.majority_vote_classification(number_of_trees, start_nodes_red, start_nodes_green, restricted)

    """CLASSIFICATION ALGORITHMS"""
    """only for trees which are Kakutani separable"""

    def fast_greedy_classification(self, number_of_steps, print_steps=False):
        green_learning = self.tree_closure(self.learning_green_nodes)
        red_learning = self.tree_closure(self.learning_red_nodes)
        self.initial_green = len(green_learning)
        self.initial_red = len(red_learning)
        if green_learning.intersection(red_learning):
            return False
        else:
            # print(len(green_learning) + len(red_learning))
            self.learning_green_nodes = green_learning
            self.learning_red_nodes = red_learning
            return self.greedy_tree_partition(number_of_steps, print_steps=print_steps)

    def opt_tree_classification(self):
        start_opt = time.time()
        green_learning = self.tree_closure(self.learning_green_nodes)
        red_learning = self.tree_closure(self.learning_red_nodes)
        self.initial_green = len(green_learning)
        self.initial_red = len(red_learning)
        middle_time = time.time()
        if green_learning.intersection(red_learning):
            return False
        else:
            self.learning_green_nodes = green_learning
            self.learning_red_nodes = red_learning
            training = np.zeros(self.Graph.number_of_nodes())
            for x in self.learning_green_nodes:
                training[x] = 1
            for x in self.learning_red_nodes:
                training[x] = -1

            directed_tree = np.full(self.Graph.number_of_nodes(), -1)
            e = next(iter(self.learning_green_nodes))
            directed_tree[e] = e
            current_elements = [e]

            while current_elements:
                current_elem = current_elements.pop(0)
                neighbors = nx.neighbors(self.Graph, current_elem)
                for x in neighbors:
                    if directed_tree[x] == -1:
                        current_elements.append(x)
                        directed_tree[x] = current_elem
                        if x in self.learning_red_nodes:
                            red_green_path = [x]
                            while red_green_path[-1] not in self.learning_green_nodes:
                                red_green_path.append(directed_tree[red_green_path[-1]])
                            for i, y in enumerate(red_green_path, 0):
                                if i < len(red_green_path) // 2:
                                    self.learning_red_nodes.add(y)
                                else:
                                    self.learning_green_nodes.add(y)
                            end_time = time.time()
                            return self.fast_greedy_classification(1)

    """transductive learning"""

    def transductive_learning_classification(self, number_of_steps, restriction=set()):  #
        start = time.time()
        learning_elements = self.learning_red_nodes.union(self.learning_green_nodes)
        classification = {}
        for i in range(len(self.nodes)):
            classification[i] = [0, 0, 0]

        for i in range(number_of_steps):
            # print("{0}/{1}".format(i, number_of_steps))
            red_training = set()
            green_training = set()
            T = self.random_spanning_forest()
            T_Graph = Graph_Generation(random_generate=False, Graph=T)
            list = random.sample(learning_elements, len(learning_elements))
            while list:
                red_training.add(list[0])
                closure = T_Graph.graph_closure(red_training, restriction)
                if list[0] in self.learning_red_nodes and not closure.intersection(self.learning_green_nodes):
                    red_training = closure
                    for elem in red_training:
                        if elem in list:
                            list.remove(elem)
                else:
                    red_training.remove(list[0])
                    green_training.add(list[0])
                    closure = T_Graph.graph_closure(green_training, restriction)
                    if list[0] in self.learning_green_nodes and not closure.intersection(self.learning_red_nodes):
                        green_training = closure
                        for elem in green_training:
                            if elem in list:
                                list.remove(elem)
                    else:
                        green_training.remove(list[0])
                        list.remove(list[0])

            for x in green_training:
                classification[x][0] += 1
            for x in red_training:
                classification[x][1] += 1
            for x in T_Graph.nodes.difference(red_training.union(green_training)):
                classification[x][2] += 1

        print(time.time() - start)
        return classification

    """uses iteratively the greedy algorithm from the paper to classify the nodes"""

    def greedy_tree_partition(self, number_of_steps=1, restriction=set(), print_steps=False):
        training_elements = self.learning_red_nodes.union(self.learning_green_nodes)
        classification = {}

        for i in range(len(self.nodes)):
            classification[i] = [0, 0, 0]

        # print steps only for testing
        if print_steps:
            for t in range(number_of_steps):
                # if  not t % 50:
                # print("{0}/{1}".format(t, number_of_steps))
                random.seed()
                T = self.random_spanning_forest()
                T_Graph = graph_generation(red_nodes=self.red_nodes, green_nodes=self.green_nodes, Graph=T)
                # TreeL = Graph_Generation(red_nodes= self.learning_red_nodes, green_nodes=self.learning_green_nodes, graph = T)
                T_Graph.set_classified_node_class(self.learning_green_nodes, "green")
                T_Graph.set_classified_node_class(self.learning_red_nodes, "red")
                T_Graph.draw_classified_graph()
                # TreeL.draw_classified_graph()
                training = [0] * self.Graph.number_of_nodes()
                for x in self.learning_green_nodes:
                    training[x] = 1
                for x in self.learning_red_nodes:
                    training[x] = -1

                bfs_search = [0] * self.Graph.number_of_nodes()

                random.seed()
                training_set = random.sample(training_elements, len(training_elements))

                while training_set:
                    x = training_set.pop()
                    if bfs_search[x] == 0:
                        bfs_search[x] = 1

                        # node is in green training
                        if training[x] == 1:
                            red = False
                            current_red_nodes = []
                            current_green_nodes = [x]
                            classification[x][0] += 1
                            T_Graph.set_classified_node_class({x}, "green")
                            while current_green_nodes:
                                neighbors = nx.neighbors(T, current_green_nodes.pop())
                                for n in neighbors:
                                    if (red and training[n] == -1) or bfs_search[n] == 1:
                                        continue
                                    elif training[n] == -1:
                                        red = True
                                        current_red_nodes.append(n)
                                        bfs_search[n] = 1
                                        classification[n][1] += 1
                                        T_Graph.set_classified_node_class({n}, "red")
                                        T_Graph.draw_classified_graph()
                                    else:
                                        current_green_nodes.append(n)
                                        bfs_search[n] = 1
                                        classification[n][0] += 1
                                        T_Graph.set_classified_node_class({n}, "green")
                                        T_Graph.draw_classified_graph()
                            while current_red_nodes:
                                neighbors = nx.neighbors(T, current_red_nodes.pop())
                                for n in neighbors:
                                    if training[n] == 1 or bfs_search[n] == 1:
                                        continue
                                    else:
                                        current_red_nodes.append(n)
                                        bfs_search[n] = 1
                                        classification[n][1] += 1
                                        T_Graph.set_classified_node_class({n}, "red")
                                        T_Graph.draw_classified_graph()

                        # node is in red training
                        if training[x] == -1:
                            green = False
                            current_green_nodes = []
                            current_red_nodes = [x]
                            classification[x][1] += 1
                            T_Graph.set_classified_node_class({x}, "red")
                            while current_red_nodes:
                                neighbors = nx.neighbors(T, current_red_nodes.pop())
                                for n in neighbors:
                                    if (green and training[n] == 1) or bfs_search[n] == 1:
                                        continue
                                    elif training[n] == 1:
                                        green = True
                                        current_green_nodes.append(n)
                                        bfs_search[n] = 1
                                        classification[n][0] += 1
                                        T_Graph.set_classified_node_class({n}, "green")
                                        T_Graph.draw_classified_graph()

                                    else:
                                        current_red_nodes.append(n)
                                        bfs_search[n] = 1
                                        classification[n][1] += 1
                                        T_Graph.set_classified_node_class({n}, "red")
                                        T_Graph.draw_classified_graph()
                            while current_green_nodes:
                                neighbors = nx.neighbors(T, current_green_nodes.pop())
                                for n in neighbors:
                                    if training[n] == -1 or bfs_search[n] == 1:
                                        continue
                                    else:
                                        current_green_nodes.append(n)
                                        bfs_search[n] = 1
                                        classification[n][0] += 1
                                        T_Graph.set_classified_node_class({n}, "green")
                                        T_Graph.draw_classified_graph()

        else:
            for t in range(number_of_steps):
                # if  not t % 50:
                # print("{0}/{1}".format(t, number_of_steps))
                random.seed()
                T = self.random_spanning_forest()
                training = [0] * self.Graph.number_of_nodes()
                for x in self.learning_green_nodes:
                    training[x] = 1
                for x in self.learning_red_nodes:
                    training[x] = -1

                bfs_search = [0] * self.Graph.number_of_nodes()

                random.seed()
                training_set = random.sample(training_elements, len(training_elements))

                while training_set:
                    x = training_set.pop()
                    if bfs_search[x] == 0:
                        bfs_search[x] = 1

                        # node is in green training
                        if training[x] == 1:
                            red = False
                            current_red_nodes = []
                            current_green_nodes = [x]
                            classification[x][0] += 1
                            while current_green_nodes:
                                neighbors = nx.neighbors(T, current_green_nodes.pop())
                                for n in neighbors:
                                    if (red and training[n] == -1) or bfs_search[n] == 1:
                                        continue
                                    elif training[n] == -1:
                                        red = True
                                        current_red_nodes.append(n)
                                        bfs_search[n] = 1
                                        classification[n][1] += 1
                                    else:
                                        current_green_nodes.append(n)
                                        bfs_search[n] = 1
                                        classification[n][0] += 1
                            while current_red_nodes:
                                neighbors = nx.neighbors(T, current_red_nodes.pop())
                                for n in neighbors:
                                    if training[n] == 1 or bfs_search[n] == 1:
                                        continue
                                    else:
                                        current_red_nodes.append(n)
                                        bfs_search[n] = 1
                                        classification[n][1] += 1

                                        # node is in red training
                        if training[x] == -1:
                            green = False
                            current_green_nodes = []
                            current_red_nodes = [x]
                            classification[x][1] += 1

                            while current_red_nodes:
                                neighbors = nx.neighbors(T, current_red_nodes.pop())
                                for n in neighbors:
                                    if (green and training[n] == 1) or bfs_search[n] == 1:
                                        continue
                                    elif training[n] == 1:
                                        green = True
                                        current_green_nodes.append(n)
                                        bfs_search[n] = 1
                                        classification[n][0] += 1
                                    else:
                                        current_red_nodes.append(n)
                                        bfs_search[n] = 1
                                        classification[n][1] += 1

                            while current_green_nodes:
                                neighbors = nx.neighbors(T, current_green_nodes.pop())
                                for n in neighbors:
                                    if training[n] == -1 or bfs_search[n] == 1:
                                        continue
                                    else:
                                        current_green_nodes.append(n)
                                        bfs_search[n] = 1
                                        classification[n][0] += 1

        return classification

    """uses iteravely the greedy tree partition to classify the nodes of the graph"""

    def iterative_greedy_tree_partition(self, tree_number, threshold=0.51):
        num_classified = len(self.learning_red_nodes.union(self.learning_green_nodes))
        num_new_classified = num_classified + 1
        classification = {}

        while num_new_classified > num_classified:
            num_classified = len(self.classified_green_nodes.union(self.classified_red_nodes))
            self.classified_green_nodes = set()
            self.classified_red_nodes = set()
            classification = self.greedy_tree_partition(tree_number, print_steps=False)
            self.majority_vote_to_classification_red_green(classification, threshold)
            self.learning_green_nodes = self.classified_green_nodes
            self.learning_red_nodes = self.classified_red_nodes
            num_new_classified = len(self.classified_green_nodes.union(self.classified_red_nodes))

        return classification

    """DRAW FUNCIONS"""

    """draw the graph with colors"""

    def draw_graph(self):
        # Draw the graph
        nx.draw_networkx_nodes(self.Graph, self.pos, nodelist=self.green_nodes, node_color="green")
        nx.draw_networkx_nodes(self.Graph, self.pos, nodelist=self.red_nodes, node_color="red")
        nx.draw_networkx_nodes(self.Graph, self.pos, nodelist=self.blue_nodes, node_color="blue")
        nx.draw_networkx_labels(self.Graph, self.pos, nodelist=self.Graph.nodes())
        nx.draw_networkx_edges(self.Graph, self.pos, self.Graph.edges(), edge_color="black")
        plt.axis('off')
        # tikz_save("/home/florian/Dokumente/Forschung/EigeneForschung/SpringerLatex/" + "TreePlot" + ".tex", wrap = False)
        plt.show()

    """draw the graph with classified colors"""

    def draw_classified_graph(self):
        # Draw the graph
        nx.draw_networkx_nodes(self.Graph, self.pos, nodelist=self.classified_green_nodes, node_color="green")
        nx.draw_networkx_nodes(self.Graph, self.pos, nodelist=self.classified_red_nodes, node_color="red")
        nx.draw_networkx_nodes(self.Graph, self.pos, nodelist=self.classified_blue_nodes, node_color="blue")
        nx.draw_networkx_labels(self.Graph, self.pos, nodelist=self.Graph.nodes())
        nx.draw_networkx_edges(self.Graph, self.pos, self.Graph.edges(), edge_color="black")
        plt.axis('off')
        plt.show()

        """draw the graph with training colors"""

    def draw_learning_graph(self):
        # Draw the graph
        nx.draw_networkx_nodes(self.Graph, self.pos, nodelist=self.learning_green_nodes, node_color="green")
        nx.draw_networkx_nodes(self.Graph, self.pos, nodelist=self.learning_red_nodes, node_color="red")
        nx.draw_networkx_nodes(self.Graph, self.pos, nodelist=self.learning_blue_nodes, node_color="blue")
        nx.draw_networkx_labels(self.Graph, self.pos, nodelist=self.Graph.nodes())
        nx.draw_networkx_edges(self.Graph, self.pos, self.Graph.edges(), edge_color="black")
        plt.axis('off')
        plt.show()

        "draw the graph with training labels"""

    def draw_training_graph(self):
        # Draw the graph
        nx.draw_networkx_nodes(self.Graph, self.pos, nodelist=self.training_green_nodes, node_color="green")
        nx.draw_networkx_nodes(self.Graph, self.pos, nodelist=self.training_red_nodes, node_color="red")
        nx.draw_networkx_nodes(self.Graph, self.pos, nodelist=self.training_blue_nodes, node_color="blue")
        nx.draw_networkx_labels(self.Graph, self.pos, nodelist=self.Graph.nodes())
        nx.draw_networkx_edges(self.Graph, self.pos, self.Graph.edges(), edge_color="black")
        plt.axis('off')
        plt.show()

    """TODO"""

    def print_to_file(self, filename="data.csv"):
        return ""

    """Add a classification to a file"""

    def add_row_to_file(self, filename, classification, threshold, learning_seed, train_method, tree_number,
                        classification_method="majority"):
        if classification_method == "red_green":
            self.majority_vote_to_classification_red_green(classification, threshold)
        elif classification_method == "majority":
            self.majority_vote_to_classification(classification)

        correct, wrong, unclassified = self.precision_of_classification()
        red_correct, red_false, red_unclassified, green_correct, green_false, green_unclassified = self.full_classification_error()
        precision = (correct + wrong - len(self.learning_green_nodes) - len(self.learning_red_nodes)) / (
                correct + wrong - len(self.learning_green_nodes) - len(self.learning_red_nodes) + unclassified)
        recall = (correct - len(self.learning_green_nodes) - len(self.learning_red_nodes)) / (
                correct + wrong - len(self.learning_green_nodes) - len(self.learning_red_nodes))
        with open(filename, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([recall, precision, correct, wrong, unclassified, red_correct, red_false, red_unclassified,
                             green_correct, green_false, green_unclassified, self.Graph.number_of_nodes(),
                             self.Graph.number_of_edges(), self.graph_seed, self.color_method, self.color_seed,
                             len(self.red_nodes), len(self.green_nodes), len(self.blue_nodes),
                             len(self.learning_red_nodes), len(self.learning_red_nodes), learning_seed, train_method,
                             tree_number, threshold])

    """If saving a classification result to a sqlite check if db exists otherwise create"""

    def create_table(self, db_name, table_name, column_list):
        if not os.path.isfile(db_name):
            con = sqlite3.connect(db_name)
            cur = con.cursor()

            string = ""
            for entry in column_list:
                string += entry
                string += ","
            string = string[:-1]
            cur.execute("CREATE TABLE " + table_name + " (" + string + ");")

    def generate_db_entries(self, classification, threshold, num_learning_green, num_learning_red, learning_seed,
                            train_method, number_of_steps, classification_method="red_green"):
        if classification_method == "red_green":
            self.majority_vote_to_classification_red_green(classification, threshold)
        elif classification_method == "majority":
            self.majority_vote_to_classification(classification)

        correct, wrong, unclassified = self.precision_of_classification()
        red_correct, red_false, red_unclassified, green_correct, green_false, green_unclassified = self.full_classification_error()

        recall = (correct + wrong - num_learning_green - num_learning_red) / (
                correct + wrong - num_learning_green - num_learning_red + unclassified)

        if correct + wrong - num_learning_green - num_learning_red == 0:
            accuracy = 0
        else:
            accuracy = (correct - num_learning_green - num_learning_red) / (
                    correct + wrong - num_learning_green - num_learning_red)
        if correct + wrong - self.initial_green - self.initial_red + unclassified == 0:
            adjusted_accuracy = 0
        else:
            adjusted_accuracy = (correct - self.initial_green - self.initial_red) / (
                    correct + wrong - self.initial_green - self.initial_red + unclassified)

        return [time.time(), round(float(accuracy), 2),
                max(len(self.red_nodes), len(self.green_nodes)) / len(self.nodes),
                round(float(recall), 2), int(correct), int(wrong), int(unclassified), int(red_correct),
                int(red_false), int(red_unclassified), int(green_correct), int(green_false), int(green_unclassified),
                int(self.Graph.number_of_nodes()), int(self.Graph.number_of_edges()), int(self.graph_seed),
                str(self.color_method), int(self.color_seed), int(len(self.red_nodes)), int(len(self.green_nodes)),
                int(len(self.blue_nodes)), int(num_learning_red), int(num_learning_green), int(learning_seed),
                str(train_method), int(number_of_steps), float(threshold), self.initial_green, self.initial_red,
                adjusted_accuracy]

    """Calculate number of spanning trees for the graph (DOES NOT WORK)"""

    def number_of_spanning_trees(self):
        laplace = nx.laplacian_matrix(self.Graph).to_numpy_array()
        tree_mat = np.delete(np.delete(laplace, 0, 0))
