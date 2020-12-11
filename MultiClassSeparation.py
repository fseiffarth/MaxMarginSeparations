import sys
import time
import math
from collections import OrderedDict
import random
import copy

import matplotlib
from scipy.spatial import ConvexHull
import numpy as np
import itertools
import sklearn
import matplotlib.pyplot as plt

####Class for the point set in which separation is done
from Algos import get_points_inside_convex_hull, intersect, get_points_inside_convex_hull_linprog


def joggle_points(points, indices, prec=10e-8):
    points[indices] += np.random.rand(len(indices), len(points[0])) * prec


def add_random_neighbours(points, point, number=1, prec=10e-6):
    max_vals = np.amax(np.abs(points), axis=0)
    random_np = np.random.rand(number, len(points[0])) * prec * max_vals
    new_points = np.repeat(np.reshape(points[point], (1, -1)), number, axis=0) + random_np
    return new_points


class MultiClassSeparation(object):
    def __init__(self, E=[], training_sets=[], labels=[], openMLData=None, is_manifold_true=False, is_greedy=False, linprog_calc=None, print_runtimes=None):
        self.field_size = math.inf
        self.openMLData = None
        self.is_greedy = is_greedy
        self.linprog_calc = linprog_calc
        self.print_runtimes = print_runtimes

        if openMLData == None:
            self.E = E
            self.labels = labels
        else:
            self.openMLData = openMLData
            self.E = openMLData.data_X
            self.labels = openMLData.data_y

        self.training_sets = training_sets
        self.size_E = len(self.E)

        self.dimension_E = len(self.E[0])
        self.class_number = len(self.training_sets)

        # if training set is to small generate random training points nearby the given training points
        self.virtual_training_points = []
        for _ in range(self.class_number):
            self.virtual_training_points.append([])

        # point distances
        self.distances = []
        self.pre_sorting = []
        self.confidences = [[0 for _ in range(self.class_number)] for _ in range(self.size_E)]

        self.initial_hulls = []
        self.current_hulls = []
        self.current_vertices = []
        self.S = []
        self.colors_E = []

        self.outside_points = np.ones(self.size_E).astype(np.int)

        for i, train_set in enumerate(self.training_sets):
            # if the train set is not fully dimensional add points near to the given points to get full dimension
            if len(train_set) <= self.dimension_E:
                self.virtual_training_points[i] = add_random_neighbours(self.E, train_set[0],
                                                                        self.dimension_E + 1 - len(train_set))
            # Joggle the points to achieve full dimension
            joggle_points(self.E, train_set)
            if self.virtual_training_points[i] != []:
                hull = ConvexHull(np.concatenate((self.E[train_set], self.virtual_training_points[i])), 1)
                self.initial_hulls.append(hull)
                self.current_hulls.append(hull)
            else:
                hull = ConvexHull(self.E[train_set], 1)
                self.initial_hulls.append(hull)
                self.current_hulls.append(hull)
            hull_points, added = get_points_inside_convex_hull(self.E, self.initial_hulls[-1], train_set,
                                                               [x for x in range(self.size_E) if x not in train_set])
            self.S.append(hull_points)

            for x in hull_points:
                if x < self.size_E:
                    self.confidences[x][i] = sys.maxsize / (self.class_number + 1)

        # Set unlabeled elements
        self.F = list(set(range(self.size_E)) - set([x for y in self.S for x in y]))

        # Pre-computation of monotone linkages for all classes
        start = time.time()
        if not self.is_greedy:
            self.linkage_pre_computation()
        # print("Preprocessing {}".format(time.time()-start))

    def linkage_pre_computation(self):
        for _ in range(self.class_number):
            self.distances.append({})
        for i in range(self.class_number):
            t = sklearn.metrics.pairwise_distances(X=self.E[self.F], Y=self.E[self.S[i]], metric="euclidean").min(
                axis=1)
            for j, _ in enumerate(self.F):
                self.distances[i][self.F[j]] = t[j]
            self.distances[i] = OrderedDict(
                sorted(self.distances[i].items(), key=lambda l: l[1], reverse=True))

        self.pre_sorting = OrderedDict()
        for x in self.F:
            dist_list = []
            class_list = range(0, self.class_number)
            for i in range(self.class_number):
                dist_list.append(self.distances[i][x])
            class_list = [x for _, x in sorted(zip(dist_list, class_list))]
            self.pre_sorting[x] = (min(dist_list), class_list)
        self.pre_sorting = OrderedDict(sorted(self.pre_sorting.items(), key=lambda l: l[1][0], reverse=True))

    def set_confidence(self, classes, classification, confidence_measure=None):
        if confidence_measure is None:
            for i, x in enumerate(classification):
                if x == 0:
                    self.confidences[i][classes[0]] += 1
                elif x == 1:
                    self.confidences[i][classes[1]] += 1
        elif confidence_measure == "linkage":
            for i, x in enumerate(classification):
                if x == 0:
                    if i in self.F:
                        distance = self.distances[classes[0]][i]
                        if distance == 0:
                            self.confidences[i][classes[0]] += sys.maxsize / self.class_number ** 2
                        else:
                            self.confidences[i][classes[0]] += 1. / distance
                elif x == 1:
                    if i in self.F:
                        distance = self.distances[classes[1]][i]
                        if distance == 0:
                            self.confidences[i][classes[1]] += sys.maxsize / self.class_number ** 2
                        else:
                            self.confidences[i][classes[1]] += 1. / distance

        elif confidence_measure == "one_vs_all":
            for i, x in enumerate(classification):
                if x == 0:
                    if i in self.F:
                        distance = self.distances[classes][i]
                        if distance == 0:
                            self.confidences[i][classes] += sys.maxsize / self.class_number ** 2
                        else:
                            self.confidences[i][classes] += 1. / distance

    def prediction_from_confidences(self):
        predicted_labels = np.zeros(shape=(self.size_E))
        for i, x in enumerate(self.confidences):
            value = x.index(max(x))
            predicted_labels[i] = value
        return predicted_labels

    def one_vs_one(self, confidence_measure=None):
        total_calls = 0
        for pair in itertools.combinations(range(self.class_number), r=2):
            classification_setup = MultiClassSeparation(self.E, [self.S[pair[0]], self.S[pair[1]]])
            (calls, prediction, separable) = classification_setup.generalized_algorithm()
            if separable:
                total_calls += calls
                self.set_confidence(pair, prediction, confidence_measure)
        return total_calls, self.prediction_from_confidences(), True

    def one_vs_all(self):
        total_calls = 0
        for i in range(self.class_number):
            classification_setup = MultiClassSeparation(self.E, [self.S[i],
                                                                 [x for j in range(self.class_number) if j != i for x in
                                                                  self.S[j]]])
            (calls, prediction, separable) = classification_setup.generalized_algorithm()
            if separable:
                total_calls += calls
                self.set_confidence(i, prediction, "one_vs_all")
        return total_calls, self.prediction_from_confidences(), True

    def generalized_algorithm(self):
        #Check if initial sets are disjoint
        if set.intersection(*[set(x) for x in self.S]):
            prediction = np.empty((self.size_E,), dtype=np.int)
            prediction.fill(-1)
            for i, class_data in enumerate(self.training_sets):
                for elem in class_data:
                    prediction[elem] = i
            return 0, prediction, False
        #If sets are disjoint continue
        oracle_calls = 0
        if not self.is_greedy:
            F = np.asarray([x for x, _ in self.pre_sorting.items()]).astype(dtype=np.int)
        else:
            F = [x for x in self.F]
            random.shuffle(F)
            F = np.asarray(F).astype(dtype=np.int)

        arange_indices = np.arange(0, len(F))
        F_indices = np.empty((self.size_E,), dtype=np.int)
        F_indices[F] = arange_indices
        elements_remaining = len(F)
        prediction = np.empty((self.size_E,), dtype=np.int)
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
            for i in range(self.class_number):
                if not self.is_greedy:
                    current_class = self.pre_sorting[next_point][1][i]
                else:
                    current_class = i
                current_class_elements = np.where(prediction == current_class)[0]
                current_class_outside = np.where(prediction != current_class)[0]
                start = time.time()
                if self.linprog_calc is None:
                    hull = copy.copy(self.current_hulls[current_class])
                    hull.add_points([self.E[next_point]], 1)
                    if self.print_runtimes:
                        print("Hull add point", time.time()-start)
                    start = time.time()
                    new_closed, added = get_points_inside_convex_hull(self.E, hull, current_class_elements, current_class_outside)
                    if self.print_runtimes:
                        print("Get Inside points", time.time() - start)
                else:
                    new_closed, added = get_points_inside_convex_hull_linprog(self.E, np.append(current_class_elements, next_point),
                                                                      current_class_outside, next_point)
                    if self.print_runtimes:
                        print("Linprog", time.time() - start)
                oracle_calls += 1
                start = time.time()
                if not intersect(new_closed, np.setdiff1d(np.where((prediction >= 0))[0],
                                                          current_class_elements)):
                    F[F_indices[added]] = -1
                    elements_remaining = len(np.where(F != -1)[0])
                    prediction[added] = current_class
                    if not self.linprog_calc:
                        self.current_hulls[current_class] = hull
                    if self.print_runtimes:
                        print("Intersection", time.time() - start)

                    break
                if self.print_runtimes:
                    print("No Intersection", time.time() - start)
            if self.print_runtimes:
                print("Element Added Total Time", time.time() - element_start)
        return oracle_calls, prediction, True

    def plot_prediction(self, prediction, disjoint_sets):
        cmap = matplotlib.cm.get_cmap('viridis')
        for i in range(self.openMLData.class_number):
            prediction_indices = np.where(prediction == i)[0]
            class_indices = np.where(self.openMLData.data_y == i)[0]
            points = self.openMLData.data_X[prediction_indices]
            hull = ConvexHull(points)

            plt.plot(self.openMLData.data_X[prediction_indices[hull.vertices], 0],
                     self.openMLData.data_X[prediction_indices[hull.vertices], 1]
                     , linestyle="--", lw=2)
            plt.scatter(self.openMLData.data_X[class_indices, 0], self.openMLData.data_X[class_indices, 1],
                        c=[cmap(float(i) / (self.openMLData.class_number - 1)) for _ in range(len(class_indices))])
            plt.scatter(self.openMLData.data_X[prediction_indices[hull.vertices], 0],
                        self.openMLData.data_X[prediction_indices[hull.vertices], 1])
        plt.show()
        plt.savefig("D:\EigeneDokumente\Forschung\Code\MaxMarginSeparations\LongVersion\Images/test.png")
