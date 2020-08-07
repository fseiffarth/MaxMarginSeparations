import sys
import time
import random
import copy
from copy import deepcopy
from math import sqrt
import math
from collections import OrderedDict
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tkinter.constants import INSIDE
from scipy.optimize import linprog
import sqlite3
import os
from sklearn.datasets.base import load_diabetes, load_breast_cancer
from sklearn.datasets.openml import fetch_openml
from scipy.io import arff
import tikzplotlib
from PyGEL3D import gel
import itertools
import seaborn as sb

####Class for the point set in which separation is done
from Algos import get_points_inside_convex_hull, dist, intersect, time_step
from BinarySeparation import BinarySeparation


class MultiClassSeparation(object):
    def __init__(self, E=[], training_sets=[], labels=[], openMLData=None, is_manifold_true=False):
        self.field_size = math.inf
        self.openMLData = None

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

        # point distances
        self.distances = []
        self.pre_sorting = []
        self.confidences = [[0 for _ in range(self.class_number)] for _ in range(self.size_E)]

        self.hulls = []
        self.S = []
        self.colors_E = []

        for i, train_set in enumerate(self.training_sets):
            self.hulls.append(ConvexHull(self.E[train_set], 1))
            hull_points, added = get_points_inside_convex_hull(self.E, self.hulls[-1], train_set,
                                                               [x for x in range(self.size_E) if x not in train_set])
            self.S.append(hull_points)

            for x in hull_points:
                self.confidences[x][i] = sys.maxsize

        if is_manifold_true:
            # manifolds from hull
            self.m1 = gel.Manifold()
            for s in self.hull_C_A.simplices:
                self.m1.add_face(self.hull_C_A.points[s])

            self.m1dist = gel.MeshDistance(self.m1)
            self.m2 = gel.Manifold()
            for s in self.hull_C_B.simplices:
                self.m2.add_face(self.hull_C_B.points[s])

            self.m2dist = gel.MeshDistance(self.m2)

        # Set test elements
        self.F = list(set(range(self.size_E)) - set([x for y in self.S for x in y]))

        # Pre-computation of monotone linkages for all classes
        self.linkage_pre_computation()

    def linkage_pre_computation(self):
        for _ in range(self.class_number):
            self.distances.append({})
        for x in self.F:
            for i, hull_points in enumerate(self.S, 0):
                min_dist = sys.maxsize
                for point in hull_points:
                    point_dist = dist(self.E[point], self.E[x])
                    if point_dist < min_dist:
                        min_dist = point_dist
                self.distances[i][x] = min_dist

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
        if confidence_measure == None:
            for i, x in enumerate(classification):
                if x == 0:
                    self.confidences[i][classes[0]] += 1
                elif x == 1:
                    self.confidences[i][classes[1]] += 1
        elif confidence_measure == "linkage":
            for i, x in enumerate(classification):
                if x == 0:
                    if i in self.F:
                        self.confidences[i][classes[0]] += 1. / self.distances[classes[0]][i]
                elif x == 1:
                    if i in self.F:
                        self.confidences[i][classes[1]] += 1. / self.distances[classes[1]][i]

        elif confidence_measure == "one_vs_all":
            for i, x in enumerate(classification):
                if x == 0:
                    if i in self.F:
                        self.confidences[i][classes] += 1. / self.distances[classes][i]

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
        time_point = time.time()
        oracle_calls = 0
        F = [x for x, _ in self.pre_sorting.items()]
        disjoint_closed = self.S.copy()
        outside_points = []
        for _ in range(self.class_number):
            outside_points.append(list(range(self.size_E)))
        prediction = [-1 for _ in range(self.size_E)]
        for i, closed_set in enumerate(disjoint_closed):
            for x in closed_set:
                prediction[x] = i
                outside_points[i].remove(x)

        for pair in itertools.combinations(range(self.class_number), r=2):
            if intersect(disjoint_closed[pair[0]], disjoint_closed[pair[1]]):
                return [], [], False
        while len(F) > 0:
            #print(len(F))
            next_point = F.pop()
            for i in range(self.class_number):
                current_class = self.pre_sorting[next_point][1][i]
                hull = ConvexHull(self.E[disjoint_closed[current_class]], 1)
                hull.add_points([self.E[next_point]], 1)
                time_point = time_step("Adding Convex Hull E 1:", time_point)
                new_closed, added = get_points_inside_convex_hull(self.E, hull, disjoint_closed[current_class],
                                                                  outside_points[current_class])
                oracle_calls += 1
                intersection = False
                for j in range(self.class_number):
                    if j != current_class and intersect(new_closed, disjoint_closed[j]):
                        intersection = True
                        break
                if not intersection:
                    disjoint_closed[current_class] = new_closed
                    for x in added:
                        if x in F:
                            F.remove(x)
                        outside_points[current_class].remove(x)
                        prediction[x] = current_class
                    time_point = time_step("Update arrays:", time_point)
                    break
        return oracle_calls, prediction, True
