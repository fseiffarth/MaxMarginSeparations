import random
import time

from DataToSQL import DataToSQL
import networkx as nx
from joblib import Parallel, delayed
from graph_generation import graph_generation

class TreeSeparation(object):
    def __init__(self, dataset, classifiers=["greedy", "opt"], tree_sizes=[1000], training_sizes=[3, 4, 5, 10, 25, 50],
                 number_of_examples=1000,
                 step_number=100, class_balance=1 / 3, threshold=0.5, plotting=False, job_num=16):
        self.number_of_examples = number_of_examples
        self.threshold = threshold
        self.class_balance = class_balance
        self.classifiers = classifiers
        self.training_sizes_job_num = min(job_num, len(training_sizes))
        self.plotting = plotting
        self.dataset = dataset
        self.tree_sizes = tree_sizes
        self.step_number = step_number
        self.training_sizes = training_sizes

        self.tree_init_time = 0
        self.greedy_classifier_time = 0
        self.opt_classifier_time = 0

    def init_tree(self, size, classes=2, seed=0, color_seed=0):
        start = time.time()
        counter = 1
        while counter:
            counter += 1
            tree = nx.random_tree(size, random.seed())
            start_nodes = random.sample([x for x in range(0, size)], classes)
            red = set(start_nodes[:len(start_nodes) // 2])
            green = set(start_nodes[len(start_nodes) // 2:])
            """set the training examples"""
            tree_generated = graph_generation(color_method="random_grow_coloring", color_input=[set(red), set(green)],
                                              color_seed=color_seed, graph=tree, graph_seed=seed + color_seed,
                                              layout=False)
            if len(tree_generated.red_nodes) / size > self.class_balance and len(
                    tree_generated.green_nodes) / size > self.class_balance:
                counter = 0
        self.tree_init_time = time.time() - start
        return tree_generated

    def run_single_example(self, tree_generated, training_size, learning_seed=0):
        """set the training examples"""
        tree_generated.learning_red_nodes = set()
        tree_generated.learning_green_nodes = set()
        train_a, train_b = tree_generated.random_choose_training_samples(training_size, random.seed())
        #draw_graph_labels(Tree.graph, [Tree.class_a, Tree.class_b, Tree.learning_red_nodes, Tree.learning_green_nodes], ["red", "green", "black", "blue"])
        #draw_graph_with_labels_training(tree)

        for classifier in self.classifiers:
            if classifier == "greedy":
                start = time.time()
                classification = tree_generated.fast_greedy_classification(self.step_number)
                self.greedy_classifier_time = time.time() - start
                print("Size: {} Greedy: {}".format(len(tree_generated.nodes), self.greedy_classifier_time))
            if classifier == "opt":
                start = time.time()
                classification = tree_generated.opt_tree_classification()
                self.opt_classifier_time = time.time() - start
                print("Size: {} Opt: {}".format(len(tree_generated.nodes),self.opt_classifier_time))
            columns = ['Timestamp', 'Accuracy', 'DefaultVal', 'Coverage', 'Correct', 'Wrong', 'Unclassified',
                       'RedCorrect', 'RedFalse', 'RedUnclassified', 'GreenCorrect', 'GreenFalse',
                       'GreenUnclassified', 'NumNodes', 'NumEdges', 'GraphSeed', 'ColorMethod', 'ColorSeed',
                       'NumberRed', 'NumberGreen', 'NumberBlue', 'NumTrainingRed', 'NumTrainingGreen',
                       'TrainingSeed', 'TrainMethod', 'TreeNumber', 'Threshold', 'InitialGreen', 'InitialRed', 'AdjustedAccuracy']
            column_types = ["FLOAT" for x in columns]
            entries = tree_generated.generate_db_entries(classification, threshold=self.threshold, learning_seed=learning_seed,
                                               train_method=classifier, number_of_steps=self.step_number,
                                               num_learning_green=train_b,
                                               num_learning_red=train_a, classification_method="red_green")
            # print(train_a, train_b, classifier, classification, len(classification))
            self.dataset.experiment_to_database("ECML2020" + classifier, columns, [entries], column_types)

    def run_tree_experiment(self):
        for i in range(self.number_of_examples):
            for tree_size in self.tree_sizes:
                tree_generated = self.init_tree(tree_size, seed=(i + 1) * tree_size, color_seed=(i + 1) * tree_size)
                print("Example Num: {}, Tree size: {} Time to init: {}".format(i, tree_size, self.tree_init_time))
                Parallel(n_jobs=self.training_sizes_job_num)(delayed(self.run_single_example)(tree_generated, training_size,
                                                                                              learning_seed=training_size)
                                                             for
                                                             training_size in self.training_sizes if
                                                             tree_size // training_size >= 10)
