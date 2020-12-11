import itertools
from abc import ABC
from collections import OrderedDict

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import random
import time
from joblib import Parallel, delayed

import GraphNumpyFunctions
from DataStructure import DataStructure


class GraphData(DataStructure, ABC):
    '''
    classdocs
    '''

    def __init__(self, graph: nx.Graph, class_num: int = 2, max_class_size: float = 0.75, min_class_size: float = 0.5, plot: bool = False):
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

        """Initialize the coloring and set all the class variables"""
        # set graph
        self.class_num = class_num

        self.Graph = graph
        self.size = self.Graph.number_of_nodes()

        self.linkage_function = lambda nodes, element: GraphNumpyFunctions.graph_distance_linkage(graph, nodes, element)
        self.closure_operator = lambda x: GraphNumpyFunctions.graph_closure(self.Graph, x)

        if plot:
            self.plot_data()


        # other inputs
        self.labels = self.label_data(max_class_size, min_class_size / class_num)
        self.training_samples = np.zeros(0)

        # plot the graph
        if plot:
            self.pos = nx.spring_layout(self.Graph)
            # self.pos = nx.nx_agraph.graphviz_layout(self.graph)
            self.draw_graph()

    """Methods for coloring of the graph"""

    # Label a graph with number of classes different labels
    def label_data(self, max_biggest_class_size=0.75, min_smallest_class=0.1, start_nodes=None):
        biggest_class = 1
        smallest_class = 0
        counter = 0
        return_labels = np.empty(self.Graph.number_of_nodes(), dtype=np.int8)
        return_labels.fill(-1)
        current_nodes = []
        while (biggest_class >= max_biggest_class_size or smallest_class <= min_smallest_class) and counter <= 1000:
            counter += 1
            # print(counter)
            if start_nodes is None:
                start_nodes = random.sample([[x] for x in range(0, self.Graph.number_of_nodes())], self.class_num)
            fixed_labels = np.empty(self.Graph.number_of_nodes(), dtype=np.int8)
            fixed_labels.fill(-1)
            current_nodes.clear()

            for i, nodes in enumerate(start_nodes):
                for node in nodes:
                    fixed_labels[node] = i
                    current_nodes.append(node)

            while len(current_nodes) > 0:
                nodes_step = current_nodes.copy()
                current_nodes.clear()
                current_labels = fixed_labels.copy()
                for node in nodes_step:
                    for neighbor in nx.all_neighbors(self.Graph, node):
                        if fixed_labels[neighbor] == -1:
                            if current_labels[neighbor] == -1:
                                current_labels[neighbor] = current_labels[node]
                                current_nodes.append(neighbor)
                            else:
                                rand_num = random.randint(0, 1)
                                if rand_num:
                                    current_labels[neighbor] = current_labels[node]
                fixed_labels = current_labels.copy()
            start_nodes = None
            for i in range(0, self.class_num):
                if i == 0:
                    biggest_class = len(np.where(fixed_labels == i)[0]) / float(self.Graph.number_of_nodes())
                    smallest_class = len(np.where(fixed_labels == i)[0]) / float(self.Graph.number_of_nodes())
                else:
                    biggest_class = max(biggest_class,
                                        len(np.where(fixed_labels == i)[0]) / float(self.Graph.number_of_nodes()))
                    smallest_class = min(smallest_class,
                                         len(np.where(fixed_labels == i)[0]) / float(self.Graph.number_of_nodes()))

            if counter == 1:
                return_labels = fixed_labels.copy()
            else:
                smallest_return_class = 0
                for i in range(0, self.class_num):
                    if i == 0:
                        smallest_return_class = len(np.where(return_labels == i)[0]) / float(
                            self.Graph.number_of_nodes())
                    else:
                        smallest_return_class = min(smallest_return_class,
                                                    len(np.where(return_labels == i)[0]) / float(
                                                        self.Graph.number_of_nodes()))
                if smallest_return_class < smallest_class:
                    return_labels = fixed_labels.copy()
        # print(return_labels)
        return return_labels

    def random_choose_training_samples(self, n_train_samples, seed=0):
        # make reproduceble examples
        if seed:
            random.seed(seed)
        else:
            random.seed()

        self.training_samples = np.zeros((self.class_num, n_train_samples), dtype=np.int32)
        for i in range(self.class_num):
            y = list(np.where(self.labels == i)[0])
            x = random.sample(y, n_train_samples)
            self.training_samples[i] = np.asarray(x)

        return n_train_samples, n_train_samples

    def intersect(self, A, B):
        result = len(set(A).intersection(B))
        return result != 0

    def training_sets_disjoint(self, training_sets):
        for pair in itertools.combinations(range(self.class_num), r=2):
            if self.intersect(training_sets[pair[0]], training_sets[pair[1]]):
                return False
        return True

    """CLASSIFICATION ALGORITHMS"""

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
                closure.append(x)
                x = directed_tree[x]

        return closure

    def tree_greedy(self, number_of_steps=1):
        closures = []
        for i in range(self.class_num):
            closures.append(self.tree_closure(list(self.training_samples[i])))
        if not self.training_sets_disjoint(closures):
            return False
        else:
            prediction = np.empty(self.Graph.number_of_nodes(), dtype=np.int8)
            prediction.fill(-1)
            current_nodes = []

            for i, nodes in enumerate(closures):
                for node in nodes:
                    prediction[node] = i

            for i in range(self.class_num):
                current_nodes.clear()
                for node in closures[i]:
                    current_nodes.append(node)
                while len(current_nodes) > 0:
                    node = current_nodes.pop()
                    for neighbor in nx.all_neighbors(self.Graph, node):
                        if prediction[neighbor] == -1:
                            prediction[neighbor] = prediction[node]
                            current_nodes.append(neighbor)
            return prediction

    def tree_opt(self, number_of_steps=1):
        closures = []
        prediction = np.empty(self.Graph.number_of_nodes())
        prediction.fill(-1)
        for i in range(self.class_num):
            closures.append(self.tree_closure(list(self.training_samples[i])))
        if not self.training_sets_disjoint(closures):
            return False
        else:
            return self.label_data(1, 0, closures)

    """DRAW FUNCIONS"""

    """draw the graph with colors"""
    def plot_data(self):
        self.draw_graph_prediction(np.zeros(self.Graph.number_of_nodes()))


    def draw_graph(self):
        # Draw the graph
        pos = nx.spring_layout(self.Graph)
        cmap = matplotlib.cm.get_cmap('viridis')
        for i in range(self.class_num):
            color = cmap(float(i) / (self.class_num - 1))
            # print(self.class_num, self.labels)
            nx.draw_networkx_nodes(self.Graph, pos, nodelist=np.where(self.labels == i)[0], node_color=color)
        nx.draw_networkx_labels(self.Graph, pos, nodelist=self.Graph.nodes())
        nx.draw_networkx_edges(self.Graph, pos, self.Graph.edges(), edge_color="black")
        plt.axis('off')
        # tikz_save("/home/florian/Dokumente/Forschung/EigeneForschung/SpringerLatex/" + "TreePlot" + ".tex", wrap = False)
        plt.show()

    def draw_graph_prediction(self, prediction=[]):
        # Draw the graph
        pos = nx.spring_layout(self.Graph)
        cmap = matplotlib.cm.get_cmap('viridis')
        for i in range(self.class_num):
            color = cmap(float(i) / (self.class_num - 1))
            # print(self.class_num, prediction)
            nx.draw_networkx_nodes(self.Graph, pos, nodelist=np.where(prediction == i)[0], node_color=color)
        nx.draw_networkx_labels(self.Graph, pos, nodelist=self.Graph.nodes())
        nx.draw_networkx_edges(self.Graph, pos, self.Graph.edges(), edge_color="black")
        plt.axis('off')
        # tikz_save("/home/florian/Dokumente/Forschung/EigeneForschung/SpringerLatex/" + "TreePlot" + ".tex", wrap = False)
        plt.show()

    def set_bfs(self, _set, distances):
        current_nodes = []
        start_nodes = _set
        visited_nodes = np.empty(self.Graph.number_of_nodes(), dtype=np.bool)
        visited_nodes.fill(False)
        distance = 0
        visited_nodes[start_nodes] = True
        for node in start_nodes:
            current_nodes.append(node)

        while len(current_nodes) > 0:
            nodes_step = current_nodes.copy()
            distance += 1
            current_nodes.clear()
            for node in nodes_step:
                for neighbor in nx.all_neighbors(self.Graph, node):
                    if not visited_nodes[neighbor]:
                        visited_nodes[neighbor] = True
                        distances[neighbor] = distance
                        current_nodes.append(neighbor)

    def linkage_pre_computation(self):
        distances = []
        for _ in range(self.class_num):
            distances.append({})

        for i, closure in enumerate(self.training_samples):
            self.set_bfs(closure, distances[i])
            distances[i] = OrderedDict(
                sorted(distances[i].items(), key=lambda l: l[1], reverse=True))

        pre_sorting = OrderedDict()
        S = []
        for _set in self.training_samples:
            S.append(self.closure_operator(_set))
        # Set unlabeled elements
        F = list(set(range(self.size)) - set([x for y in S for x in y]))
        for x in F:
            dist_list = []
            class_list = range(0, self.class_num)
            for i in range(self.class_num):
                dist_list.append(distances[i][x])
            class_list = [x for _, x in sorted(zip(dist_list, class_list))]
            pre_sorting[x] = (min(dist_list), class_list)
        return OrderedDict(sorted(pre_sorting.items(), key=lambda l: l[1][0], reverse=True))


class Evaluation:
    def __init__(self, labels, class_num):
        self.runtime = []
        self.accuracy = []
        self.w_accuracy = []
        self.coverage = []
        self.w_coverage = []
        self.training_size = 0
        self.algo = []
        self.density = []
        self.node_num = []
        self.edge_num = []
        self.labels = labels
        self.class_num = class_num
        self.size = len(labels)
        self.label_distribution = {}
        for i in range(self.class_num):
            self.label_distribution[i] = 0
        for i in labels:
            self.label_distribution[i] += 1

    def set_values(self, graph, algo, training, labels, prediction, runtime):
        prediction_dist = {}
        for x in prediction:
            if x in prediction_dist.keys():
                prediction_dist[x] += 1
            else:
                prediction_dist[x] = 1

        acc = 0
        covered = 0
        acc_w = 0
        cover_w = 0
        for key, value in prediction_dist.items():
            key = int(key)
            if key != -1:
                c_i = np.where(np.asarray(prediction) == key)[0]
                c_i_hat = np.setdiff1d(c_i, training[key])
                covered += len(c_i_hat)
                label_indices = np.where(labels == key)[0]
                correct = np.intersect1d(c_i_hat, label_indices)
                acc += len(correct)
                if len(c_i_hat > 0):
                    acc_w += len(correct) / len(c_i_hat)
                else:
                    acc_w += 0
                cover_w += len(correct) / (self.label_distribution[key] - len(training[key]))
        self.training_size = sum(len(row) for row in training)
        if covered == 0:
            self.accuracy.append(0)
        else:
            self.accuracy.append(acc / covered)
        self.w_accuracy.append(acc_w / self.class_num)
        self.coverage.append(covered / (self.size - self.training_size))
        self.w_coverage.append(cover_w / self.class_num)
        self.runtime.append(runtime)
        self.algo.append(algo)
        self.node_num.append(graph.number_of_nodes())
        self.edge_num.append(graph.number_of_edges())
        self.density.append(self.edge_num[-1] / self.node_num[-1])

    def evaluation_to_database(self, database, type):
        columns = ['Timestamp', 'Algo', 'Accuracy', 'AccuracyW', 'Coverage', 'CoverageW', 'Runtime',
                   'TrainingSize', 'Classes', 'Density', 'Graph Size', 'Edge Size']
        column_types = ["FLOAT" for x in columns]
        column_types[1] = "TEXT"
        table_name = "Data_" + str(type)
        for i in range(len(self.accuracy)):
            database.experiment_to_database(table_name, columns, [
                [time.time(), self.algo[i], self.accuracy[i], self.w_accuracy[i], self.coverage[i], self.w_coverage[i],
                 self.runtime[i], self.training_size, self.class_num,
                 self.density[i], self.node_num[i], self.edge_num[i]]], column_types)


class MaximumMarginSeparation:
    def __init__(self, data: DataStructure, is_greedy: bool = False, print_runtimes: bool = False):
        self.S = []
        self.is_greedy = is_greedy
        self.print_runtimes = print_runtimes
        self.DataObject = data
        for _set in data.training_samples:
            self.S.append(data.closure_operator(_set))
        # Set unlabeled elements
        self.F = list(set(range(self.DataObject.size)) - set([x for y in self.S for x in y]))
        if not self.is_greedy:
            self.F_sorted = data.linkage_pre_computation()

    def generalized_algorithm(self):
        # Check if initial sets are disjoint
        if set.intersection(*[set(x) for x in self.S]):
            prediction = np.empty((self.DataObject.size,), dtype=np.int)
            prediction.fill(-1)
            for i, class_data in enumerate(self.DataObject.training_sets):
                for elem in class_data:
                    prediction[elem] = i
            return 0, prediction, False
        # If sets are disjoint continue
        oracle_calls = 0
        if not self.is_greedy:
            F = np.asarray([x for x, _ in self.F_sorted.items()]).astype(dtype=np.int)
        else:
            F = [x for x in self.F]
            random.shuffle(F)
            F = np.asarray(F).astype(dtype=np.int)

        arange_indices = np.arange(0, len(F))
        F_indices = np.empty((self.DataObject.size,), dtype=np.int)
        F_indices[F] = arange_indices
        elements_remaining = len(F)
        prediction = np.empty((self.DataObject.size,), dtype=np.int)
        prediction.fill(-1)
        for i, class_data in enumerate(self.S):
            for elem in class_data:
                prediction[elem] = i

        while elements_remaining > 0:
            element_start = time.time()
            print(elements_remaining)
            next_point = F[np.max(np.where(F != -1)[0])]
            F[F_indices[next_point]] = -1
            elements_remaining -= 1
            for i in range(self.DataObject.class_num):
                if not self.is_greedy:
                    current_class = self.F_sorted[next_point][1][i]
                else:
                    current_class = i

                current_class_elements = np.where(prediction == current_class)[0]
                new_closed = self.DataObject.closure_operator(list(current_class_elements) + [next_point])
                oracle_calls += 1
                start = time.time()
                if not self.DataObject.intersect(new_closed, np.setdiff1d(np.where((prediction >= 0))[0],
                                                                          current_class_elements)):
                    F[F_indices[np.setdiff1d(new_closed, current_class_elements)]] = -1
                    elements_remaining = len(np.where(F != -1)[0])
                    prediction[new_closed] = current_class
                    if self.print_runtimes:
                        print("Intersection", time.time() - start)

                    break
                if self.print_runtimes:
                    print("No Intersection", time.time() - start)
            if self.print_runtimes:
                print("Element Added Total Time", time.time() - element_start)
        return oracle_calls, prediction, True


def run_single_example(database, algos, graph_sizes, densities=[1], class_numbers=[2], training_sizes=[2], number=1,
                       steps=1, is_tree=False, plot=False):
    """set the training samples"""
    for size in graph_sizes:
        for classes in class_numbers:
            for density in densities:
                RandomGraph = GraphNumpyFunctions.random_graph(node_number=size, edge_density=density, is_tree=is_tree, plot=plot)
                Graph = GraphData(RandomGraph, classes, plot=plot)
                for training_size in training_sizes:
                    Graph.random_choose_training_samples(training_size, random.seed())
                    eval = Evaluation(Graph.labels, Graph.class_num)
                    for algo in algos:
                        if is_tree:
                            if algo == "greedy":
                                start = time.time()
                                prediction = Graph.tree_greedy()
                                new_time = time.time()
                                if plot:
                                    Graph.draw_graph_prediction(prediction)
                            if algo == "max_margin":
                                prediction = Graph.tree_opt()
                                new_time = time.time()
                                if plot:
                                    Graph.draw_graph_prediction(prediction)
                        else:
                            mms = MaximumMarginSeparation(Graph)
                            mms.is_greedy = (algo == "greedy")
                            start = time.time()
                            _, prediction, _ = mms.generalized_algorithm()
                            new_time = time.time()
                            if plot:
                                Graph.draw_graph_prediction(prediction)
                        eval.set_values(Graph.Graph, algo, Graph.training_samples, Graph.labels, prediction,
                                        new_time - start)
    if is_tree:
        eval.evaluation_to_database(database, "Tree")
    else:
        eval.evaluation_to_database(database, "Graph")
    print("Experiment {} finished".format(number))


def run_graph(database, algos, graph_sizes, class_numbers, graph_densities, train_sizes, number=1, is_tree=False,
              print_time=False, plot=False):
    Parallel(n_jobs=1)(
        delayed(run_single_example)(database, algos, graph_sizes, graph_densities, class_numbers, train_sizes, i,
                                    is_tree=is_tree,
                                    plot=plot) for i in
        range(number))
