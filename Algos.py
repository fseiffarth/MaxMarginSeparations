'''
Created on 10.08.2018

@author: florian
'''
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


####Class for the point set in which separation is done
class ClassificationPointSet(object):
    def __init__(self, E, A, B, is_manifold_true=False):
        self.field_size = math.inf
        self.size_E = len(E)
        self.dimension_E = len(E[0])
        self.colors_E = []

        self.A = A
        self.B = B
        self.size_A = len(A)
        self.size_B = len(B)

        # set colors
        for i in range(self.size_E):
            if i in self.A:
                self.colors_E.append("red")
            elif i in self.B:
                self.colors_E.append("green")
            else:
                self.colors_E.append("blue")

        # point distances
        self.convex_A_distances = {}
        self.convex_B_distances = {}
        self.convex_A_hull_distances = {}
        self.convex_B_hull_distances = {}

        self.E = E
        self.hull_C_A = ConvexHull(self.E[self.A], 1)
        self.C_A, added = get_points_inside_convex_hull(self.E, self.hull_C_A, self.A, [x for x in range(self.size_E) if x not in self.A])
        for x in added:
            self.colors_E[x] = 'orange'

        self.hull_C_B = ConvexHull(self.E[self.B], 1)
        self.C_B, added = get_points_inside_convex_hull(self.E, self.hull_C_B, self.B, [x for x in range(self.size_E) if x not in self.B])
        for x in added:
            self.colors_E[x] = 'violet'

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

        # Set unlabeled elements
        self.F = []
        for i in range(0, self.size_E):
            if i not in self.C_A and i not in self.C_B:
                self.F.append(i)

        # Pre-computation of monotone linkages
        self.convex_a_neighbors()
        self.convex_b_neighbors()

    def plot_2d_classification(self, name="Test", colorlist=[]):

        # initialize first half space
        PointsH_1 = np.ndarray(shape=(len(self.C_A), self.dimension_E))
        counter = 0
        for i in self.C_A:
            PointsH_1[counter] = self.get_point(i)
            counter += 1
        self.C_A = ConvexHull(PointsH_1, 1)

        # Initialize second half space
        PointsH_2 = np.ndarray(shape=(len(self.C_B), self.dimension_E))
        counter = 0
        for i in self.C_B:
            PointsH_2[counter] = self.get_point(i)
            counter += 1
        self.C_B = ConvexHull(PointsH_2, 1)

        # Draw convex hulls of disjoint convex sets
        for simplex in self.C_A.simplices:
            plt.plot(self.C_A.points[simplex, 0], self.C_A.points[simplex, 1], 'k-')
        for simplex in self.C_B.simplices:
            plt.plot(self.C_B.points[simplex, 0], self.C_B.points[simplex, 1], 'k-')

        x_val_dict = {}
        y_val_dict = {}

        if colorlist == []:
            colorlist = self.colors_E

        for i, x in enumerate(colorlist, 0):
            if x not in x_val_dict:
                x_val_dict[x] = [self.E[i][0]]
                y_val_dict[x] = [self.E[i][1]]
            else:
                x_val_dict[x].append(self.E[i][0])
                y_val_dict[x].append(self.E[i][1])

        for key, value in x_val_dict.items():
            plt.scatter(value, y_val_dict[key], c=key)

        tikzplotlib.save(name + ".tex")
        plt.show()

    def plot_3d_classification(self):

        # initialize first half space
        PointsH_1 = np.ndarray(shape=(len(self.C_A), self.dimension_E))
        counter = 0
        for i in self.C_A:
            PointsH_1[counter] = self.get_point(i)
            counter += 1

        self.C_A = ConvexHull(PointsH_1, 1)

        # Initialize second half space
        PointsH_2 = np.ndarray(shape=(len(self.C_B), self.dimension_E))
        counter = 0
        for i in self.C_B:
            PointsH_2[counter] = self.get_point(i)
            counter += 1

        self.C_B = ConvexHull(PointsH_2, 1)

        # print(self.ConvexHull1.equations)

        ax = plt.axes(projection='3d')

        for simplex in self.C_A.simplices:
            ax.plot3D(self.C_A.points[simplex, 0], self.C_A.points[simplex, 1],
                      self.C_A.points[simplex, 2], 'k-')
        for simplex in self.C_B.simplices:
            ax.plot3D(self.C_B.points[simplex, 0], self.C_B.points[simplex, 1],
                      self.C_B.points[simplex, 2], 'k-')

        X_coord = np.ones(self.size_E)
        Y_coord = np.ones(self.size_E)
        Z_coord = np.ones(self.size_E)
        for i in range(self.size_E):
            X_coord[i] = self.E[i][0]
            Y_coord[i] = self.E[i][1]
            Z_coord[i] = self.E[i][2]

        ax.scatter3D(X_coord, Y_coord, Z_coord, c=self.colors_E)
        plt.show()

    def get_point(self, n):
        return self.E[n]

    def convex_a_neighbors(self, is_manifold=False):
        for n in self.F:
            min_dist = sys.maxsize
            for x in self.C_A:
                point_dist = dist(self.get_point(n), self.get_point(x))
                if point_dist < min_dist:
                    min_dist = point_dist
            self.convex_A_distances[n] = min_dist

            if is_manifold:
                d = self.m1dist.signed_distance(self.get_point(n))
                self.convex_A_hull_distances[n] = d * np.sign(d)
        self.convex_A_distances = OrderedDict(sorted(self.convex_A_distances.items(), key=lambda x: x[1], reverse=True))

        if is_manifold:
            self.convex_A_hull_distances = OrderedDict(
                sorted(self.convex_A_hull_distances.items(), key=lambda x: x[1], reverse=True))

    def convex_b_neighbors(self, is_manifold=False):
        for n in self.F:
            min_dist = sys.maxsize
            for x in self.C_B:
                point_dist = dist(self.get_point(n), self.get_point(x))
                if point_dist < min_dist:
                    min_dist = point_dist
            self.convex_B_distances[n] = min_dist

            if is_manifold:
                d = self.m2dist.signed_distance(self.get_point(n))
                self.convex_A_hull_distances[n] = d * np.sign(d)
        self.convex_B_distances = OrderedDict(sorted(self.convex_B_distances.items(), key=lambda x: x[1], reverse=True))

        if is_manifold:
            self.convex_B_hull_distances = OrderedDict(
                sorted(self.convex_B_distances.items(), key=lambda x: x[1], reverse=True))

    def decide_nearest(self):
        len1 = len(self.convex_A_distances)
        len2 = len(self.convex_B_distances)

        if len1 == 0:
            return 0
        elif len2 == 0:
            return 1
        elif next(reversed(self.convex_A_distances.values())) <= next(reversed(self.convex_B_distances.values())):
            return 1
        else:
            return 0

    def decide_nearest_hull(self):
        len1 = len(self.convex_A_hull_distances)
        len2 = len(self.convex_B_hull_distances)

        if len1 == 0:
            return 0
        elif len2 == 0:
            return 1
        elif next(reversed(self.convex_A_hull_distances.values())) <= next(
                reversed(self.convex_B_hull_distances.values())):
            return 1
        else:
            return 0

    def decide_farthest(self):
        len1 = len(self.convex_A_distances)
        len2 = len(self.convex_B_distances)

        if len1 == 0:
            return 0
        elif len2 == 0:
            return 1
        elif (next(iter(self.convex_A_distances.values())) <= next(iter(self.convex_B_distances.values()))):
            return 1
        else:
            return 0

    def add_C_A_nearest(self):
        point = list(self.convex_A_distances.items())[0][0]
        self.C_A.append(point)
        self.colors_E[point] = "orange"
        self.F.remove(point)
        del self.convex_A_distances[point]
        del self.convex_B_distances[point]

    def add_C_B_nearest(self):
        point = list(self.convex_B_distances.items())[0][0]
        self.C_B.append(point)
        self.colors_E[point] = "violet"
        self.F.remove(point)
        del self.convex_B_distances[point]
        del self.convex_A_distances[point]

    # SeCos-Algorithm from paper applied to convex hulls in R^d
    def greedy_alg(self):
        end = False
        oracle_calls = 0

        # take an arbitrary element out of F
        random.shuffle(self.F)
        # label vector of the elements (binary classification {-1, 1}
        labels = -1 * np.ones(shape=(self.size_E, 1))

        # E labels of initial labelled data
        for i in self.C_A:
            labels[i] = 1
        for i in self.C_B:
            labels[i] = 0

        if not intersect(self.C_A, self.C_B):
            while len(self.F) > 0 and end == False:
                # print(len(self.F))
                next_point = self.F.pop()
                new_hull_C_A = ConvexHull(self.E[self.C_A], 1)
                new_hull_C_A.add_points([self.get_point(next_point)], 1)
                new_C_A, added = get_points_inside_convex_hull(self.E, new_hull_C_A, self.C_A, self.F + self.C_B,
                                                               next_point)
                oracle_calls += 1

                if not intersect(new_C_A, self.C_B):
                    self.hull_C_A = new_hull_C_A
                    self.C_A = new_C_A
                    self.F = list(set(self.F) - set(added))

                    for x in added:
                        labels[x] = 1
                        self.colors_E[x] = "orange"

                else:
                    new_hull_C_B = ConvexHull(self.E[self.C_B], 1)
                    new_hull_C_B.add_points([self.get_point(next_point)], 1)
                    new_C_B, added = get_points_inside_convex_hull(self.E, new_hull_C_B, self.C_B, self.F + self.C_A,
                                                                   next_point)
                    oracle_calls += 1

                    if not intersect(self.C_A, new_C_B):
                        self.hull_C_B = new_hull_C_B
                        self.C_B = new_C_B
                        self.F = list(set(self.F) - set(added))
                        for x in added:
                            labels[x] = 0
                            self.colors_E[x] = "violet"
        else:
            return [], [], False

        return oracle_calls, labels, True

    def greedy_fast_alg(self):
        end = False
        oracle_calls = 0
        random.shuffle(self.F)

        # label vector of the elements (binary classification {1, 0} -1 are unclassified
        labels = -1 * np.ones(shape=(self.size_E, 1))
        outside_1 = self.F + self.C_B
        outside_2 = self.F + self.C_A

        # E labels of initial labelled data
        for i in self.C_A:
            labels[i] = 1
        for i in self.C_B:
            labels[i] = 0

        # E initial hulls
        inside1, added, intersection = get_inside_points(self.E, self.C_A, self.F, CheckSet=self.C_B)
        oracle_calls += 1

        if not intersection:
            for x in added:
                labels[x] = 1
                self.colors_E[x] = "orange"
            self.C_A = inside1

        # E initial hulls
        inside2, added, intersection = get_inside_points(self.E, self.C_B, self.F, CheckSet=self.C_A)
        oracle_calls += 1

        if not intersection:
            for x in added:
                labels[x] = 0
                self.colors_E[x] = "violet"
            self.C_B = inside2

        if not intersect(inside1, inside2):
            for x in self.C_A:
                if x in self.F:
                    self.F.remove(x)
            for x in self.C_B:
                if x in self.F:
                    self.F.remove(x)

            while len(self.F) > 0 and end == False:
                # print(len(self.F))
                next_point = self.F.pop()
                inside1, added, intersection = get_inside_points(self.E, self.C_A, self.F, next_point,
                                                                 self.C_B)
                oracle_calls += 1

                if not intersection:
                    for x in added:
                        labels[x] = 1
                        self.colors_E[x] = "orange"
                        if x in self.F:
                            self.F.remove(x)
                    self.C_A = inside1

                else:
                    inside2, added, intersection = get_inside_points(self.E, self.C_B, self.F, next_point,
                                                                     self.C_A)
                    oracle_calls += 1

                    if not intersection:
                        for x in added:
                            labels[x] = 0
                            self.colors_E[x] = "violet"
                            if x in self.F:
                                self.F.remove(x)
                        self.C_B = inside2
        else:
            return ([], [], False)

        return (oracle_calls, labels, True)

    def greedy_alg2(self):
        end = False
        oracle_calls = 0
        random.shuffle(self.F)

        # label vector of the elements (binary classification {-1, 1}
        labels = -1 * np.ones(shape=(self.size_E, 1))
        outside_1 = self.F + self.C_B
        outside_2 = self.F + self.C_A

        # E labels of initial labelled data
        for i in self.C_A:
            labels[i] = 1
        for i in self.C_B:
            labels[i] = 0

        # E initial hulls
        Hull1 = self.C_A
        inside1, added = get_points_inside_convex_hull(self.E, Hull1, self.C_A, self.F + self.C_B)
        oracle_calls += 1

        if not intersect(inside1, self.C_B):
            for x in added:
                labels[x] = 1
                self.colors_E[x] = "orange"
            self.C_A = inside1

        # E initial hulls
        Hull2 = self.C_B
        inside2, added = get_points_inside_convex_hull(self.E, Hull2, self.C_B, self.F + self.C_A)
        oracle_calls += 1

        if not intersect(inside2, self.C_A):
            for x in added:
                labels[x] = 0
                self.colors_E[x] = "violet"
            self.C_B = inside2

        if not intersect(inside1, inside2):
            for x in self.C_A:
                if x in self.F:
                    self.F.remove(x)
            for x in self.C_B:
                if x in self.F:
                    self.F.remove(x)

            while len(self.F) > 0 and end == False:
                # print(len(self.F))
                next_point = self.F.pop()

                if (random.randint(0, 1)):
                    Hull1 = self.C_A
                    Hull1.add_points([self.get_point(next_point)], 1)
                    inside1, added = get_points_inside_convex_hull(self.E, Hull1, self.C_A, self.F + self.C_B,
                                                                   next_point)
                    oracle_calls += 1

                    if not intersect(inside1, self.C_B):
                        self.C_A.add_points([self.get_point(next_point)], 1)
                        for x in added:
                            labels[x] = 1
                            self.colors_E[x] = "orange"
                            if x in self.F:
                                self.F.remove(x)
                        self.C_A = inside1

                    else:
                        # initialize first half space
                        PointsH_1 = np.ndarray(shape=(len(self.C_A), self.dimension_E))
                        counter = 0
                        for i in self.C_A:
                            PointsH_1[counter] = self.get_point(i)
                            counter += 1

                        self.C_A = ConvexHull(PointsH_1, 1)

                        Hull2 = self.C_B
                        Hull2.add_points([self.get_point(next_point)], 1)
                        inside2, added = get_points_inside_convex_hull(self.E, Hull2, self.C_B,
                                                                       self.F + self.C_A, next_point)
                        oracle_calls += 1

                        if not intersect(self.C_A, inside2):
                            self.C_B.add_points([self.get_point(next_point)], 1)
                            for x in added:
                                labels[x] = 0
                                self.colors_E[x] = "violet"
                                if x in self.F:
                                    self.F.remove(x)
                            self.C_B = inside2
                        else:
                            # initialize second half space
                            PointsH_2 = np.ndarray(shape=(len(self.C_B), self.dimension_E))
                            counter = 0
                            for i in self.C_B:
                                PointsH_2[counter] = self.get_point(i)
                                counter += 1

                            self.C_B = ConvexHull(PointsH_2, 1)

                else:
                    Hull2 = self.C_B
                    Hull2.add_points([self.get_point(next_point)], 1)
                    inside2, added = get_points_inside_convex_hull(self.E, Hull2, self.C_B, self.F + self.C_A,
                                                                   next_point)
                    oracle_calls += 1

                    if not intersect(inside2, self.C_A):
                        self.C_B.add_points([self.get_point(next_point)], 1)
                        for x in added:
                            labels[x] = 0
                            self.colors_E[x] = "violet"
                            if x in self.F:
                                self.F.remove(x)
                        self.C_B = inside2

                    else:
                        # initialize first half space
                        PointsH_2 = np.ndarray(shape=(len(self.C_B), self.dimension_E))
                        counter = 0
                        for i in self.C_B:
                            PointsH_2[counter] = self.get_point(i)
                            counter += 1

                        self.C_B = ConvexHull(PointsH_2, 1)

                        Hull1 = self.C_A
                        Hull1.add_points([self.get_point(next_point)], 1)
                        inside1, added = get_points_inside_convex_hull(self.E, Hull1, self.C_A,
                                                                       self.F + self.C_B, next_point)
                        oracle_calls += 1

                        if not intersect(self.C_B, inside1):
                            self.C_A.add_points([self.get_point(next_point)], 1)
                            for x in added:
                                labels[x] = 1
                                self.colors_E[x] = "orange"
                                if x in self.F:
                                    self.F.remove(x)
                            self.C_A = inside1
                        else:
                            # initialize second half space
                            PointsH_1 = np.ndarray(shape=(len(self.C_A), self.dimension_E))
                            counter = 0
                            for i in self.C_A:
                                PointsH_1[counter] = self.get_point(i)
                                counter += 1

                            self.C_A = ConvexHull(PointsH_1, 1)
        else:
            return ([], [], False)

        return (oracle_calls, labels, True)

    def optimal_alg(self):
        time_point = time.time()
        oracle_calls = 0
        counter = 0
        labels = -1 * np.ones(shape=(self.size_E, 1))
        outside_points_1 = [x for x in self.convex_A_distances.keys()] + self.C_B
        outside_points_2 = [x for x in self.convex_B_distances.keys()] + self.C_A

        # add labels
        for i in self.C_A:
            labels[i] = 1
        for i in self.C_B:
            labels[i] = 0

        if not intersect(self.C_A, self.C_B):
            while (len(self.convex_A_distances) > 0 or len(self.convex_B_distances) > 0):
                # print(len(self.Set1Distances), len(self.Set2Distances))
                added = []

                # First set is nearer to nearest not classified point
                if self.decide_nearest():
                    time_point = time_step("Find Neighbour:", time_point)
                    if len(self.convex_A_distances) > 0:
                        next_point = self.convex_A_distances.popitem()[0]
                        new_hull_C_A = ConvexHull(self.E[self.C_A], 1)
                        new_hull_C_A.add_points([self.get_point(next_point)], 1)
                        time_point = time_step("Adding Convex Hull E 1:", time_point)
                        new_C_A, added = get_points_inside_convex_hull(self.E, new_hull_C_A, self.C_A,
                                                                       outside_points_1)
                        oracle_calls += 1
                        time_point = time_step("Getting inside E:", time_point)

                        # if there is no intersection the point can be added to the first convex set
                        if not intersect(new_C_A, self.C_B):
                            time_point = time_step("Intersection Test:", time_point)
                            self.C_A = new_C_A
                            self.hull_C_A = new_hull_C_A
                            for x in added:
                                # add to labels
                                labels[x] = 1
                                self.colors_E[x] = "orange"
                                if x in self.convex_A_distances.keys():
                                    del self.convex_A_distances[x]
                                if x in self.convex_B_distances.keys():
                                    del self.convex_B_distances[x]
                            outside_points_1 = list(set(outside_points_1) - set(added))
                            time_point = time_step("Update arrays:", time_point)


                        # if there is an intersection we have to check if it can be added to the second set
                        else:
                            # Test second half space
                            new_hull_C_B = ConvexHull(self.E[self.C_B], 1)
                            new_hull_C_B.add_points([self.get_point(next_point)], 1)
                            new_C_B, added = get_points_inside_convex_hull(self.E, new_hull_C_B, self.C_B,
                                                                           outside_points_2)
                            oracle_calls += 1

                            # the point can be added to the second set,
                            # if we reach this point the first time all the other E which are classified did not change the optimal margin
                            if not intersect(self.C_A, new_C_B):
                                self.C_B = new_C_B
                                self.hull_C_B = new_hull_C_B
                                for x in added:
                                    # add to labels
                                    labels[x] = 0
                                    self.colors_E[x] = "violet"
                                    if x in self.convex_A_distances.keys():
                                        del self.convex_A_distances[x]
                                    if x in self.convex_B_distances.keys():
                                        del self.convex_B_distances[x]
                                outside_points_2 = list(set(outside_points_2) - set(added))
                            # the point cannot be added to any set
                            else:
                                if next_point in outside_points_1:
                                    outside_points_1.remove(next_point)
                                if next_point in outside_points_2:
                                    outside_points_2.remove(next_point)

                    time_point = time_step("Point add Hull:", time_point)

                # Second set is nearer to nearest not classified point
                else:
                    time_point = time_step("Find Neighbour:", time_point)
                    if len(self.convex_B_distances) > 0:
                        next_point = self.convex_B_distances.popitem()[0]
                        new_hull_C_B = ConvexHull(self.E[self.C_B], 1)
                        new_hull_C_B.add_points([self.get_point(next_point)], 1)
                        new_C_B, added = get_points_inside_convex_hull(self.E, new_hull_C_B, self.C_B,
                                                                       outside_points_2)
                        oracle_calls += 1

                        # we can add the new point to the second, the nearer set
                        if not intersect(new_C_B, self.C_A):
                            self.C_B = new_C_B
                            self.hull_C_B = new_hull_C_B
                            for x in added:
                                # add to labels
                                labels[x] = 0
                                self.colors_E[x] = "violet"
                                if x in self.convex_A_distances.keys():
                                    del self.convex_A_distances[x]
                                if x in self.convex_B_distances.keys():
                                    del self.convex_B_distances[x]
                            outside_points_2 = list(set(outside_points_2) - set(added))

                        # we check if we can add the point to the first set
                        # if we reach this point the first time all the other E which are classified did not change the optimal margin
                        else:
                            # Test first half space
                            new_hull_C_A = ConvexHull(self.E[self.C_A], 1)
                            new_hull_C_A.add_points([self.get_point(next_point)], 1)
                            new_C_A, added = get_points_inside_convex_hull(self.E, new_hull_C_A, self.C_A,
                                                                           outside_points_1)
                            oracle_calls += 1

                            # the point can be classified to the second set
                            if not intersect(self.C_B, new_C_A):
                                self.hull_C_A = new_hull_C_A
                                self.C_A = new_C_A
                                for x in added:
                                    # add to labels
                                    labels[x] = 1
                                    self.colors_E[x] = "orange"
                                    if x in self.convex_A_distances.keys():
                                        del self.convex_A_distances[x]
                                    if x in self.convex_B_distances.keys():
                                        del self.convex_B_distances[x]
                                outside_points_1 = list(set(outside_points_1) - set(added))

                            # we cannot classify the point
                            else:
                                if next_point in outside_points_1:
                                    outside_points_1.remove(next_point)
                                if next_point in outside_points_2:
                                    outside_points_2.remove(next_point)

                time_point = time_step("Point add Hull:", time_point)
        else:
            return [], [], False

        return oracle_calls, labels, True

    def optimal_hull_alg(self):
        time_point = time.time()
        oracle_calls = 0
        counter = 0
        labels = -1 * np.ones(shape=(self.size_E, 1))
        outside_points_1 = [x for x in self.convex_A_hull_distances.keys()] + self.C_B
        outside_points_2 = [x for x in self.convex_B_hull_distances.keys()] + self.C_A

        # add labels
        for i in self.C_A:
            labels[i] = 1
        for i in self.C_B:
            labels[i] = 0

        # check if hulls are intersecting
        Hull1 = self.C_A
        inside1, added = get_points_inside_convex_hull(self.E, Hull1, self.C_A, outside_points_1)
        self.C_A = inside1
        Hull2 = self.C_B
        inside2, added = get_points_inside_convex_hull(self.E, Hull2, self.C_B, outside_points_2)
        self.C_B = inside2

        if not intersect(inside1, inside2):
            while (len(self.convex_A_hull_distances) > 0 or len(self.convex_B_hull_distances) > 0):
                # print(len(self.Set1HullDistances), len(self.Set2HullDistances))
                added = []

                # First set is nearer to nearest not classified point
                if self.decide_nearest_hull():

                    time_point = time_step("Find Neighbour:", time_point)

                    if len(self.convex_A_hull_distances) > 0:
                        next_point = self.convex_A_hull_distances.popitem()[0]

                        Hull1 = self.C_A
                        Hull1.add_points([self.get_point(next_point)], 1)
                        time_point = time_step("Adding Convex Hull E 1:", time_point)

                        inside1, added = get_points_inside_convex_hull(self.E, Hull1, self.C_A,
                                                                       outside_points_1)
                        oracle_calls += 1
                        time_point = time_step("Getting inside E:", time_point)

                        # if there is no intersection the point can be added to the first convex set
                        if not intersect(inside1, self.C_B):
                            time_point = time_step("Intersection Test:", time_point)
                            self.C_A = Hull1
                            time_point = time_step("Adding Convex Hull E:", time_point)

                            for x in added:
                                # add to labels
                                labels[x] = 1
                                self.colors_E[x] = "orange"
                                if x in self.convex_A_hull_distances.keys():
                                    del self.convex_A_hull_distances[x]
                                if x in self.convex_B_hull_distances.keys():
                                    del self.convex_B_hull_distances[x]

                                outside_points_1.remove(x)
                            self.C_A = inside1

                            time_point = time_step("Update arrays:", time_point)


                        # if there is an intersection we have to check if it can be added to the second set
                        else:
                            time_point = time_step("Intersection Test:", time_point)
                            # Renew first half space
                            PointsH_1 = np.ndarray(shape=(len(self.C_A), self.dimension_E))
                            counter = 0
                            for i in self.C_A:
                                PointsH_1[counter] = self.get_point(i)
                                counter += 1
                            self.C_A = ConvexHull(PointsH_1, 1)

                            # Test second half space
                            Hull2 = self.C_B
                            Hull2.add_points([self.get_point(next_point)], 1)
                            inside2, added = get_points_inside_convex_hull(self.E, Hull2, self.C_B,
                                                                           outside_points_2)
                            oracle_calls += 1

                            # the point can be added to the second set,
                            # if we reach this point the first time all the other E which are classified did not change the optimal margin
                            if not intersect(self.C_A, inside2):
                                self.C_B = Hull2
                                for x in added:
                                    # add to labels
                                    labels[x] = 0
                                    self.colors_E[x] = "violet"
                                    if x in self.convex_A_hull_distances.keys():
                                        del self.convex_A_hull_distances[x]
                                    if x in self.convex_B_hull_distances.keys():
                                        del self.convex_B_hull_distances[x]
                                    outside_points_2.remove(x)
                                self.C_B = inside2


                            # the point cannot be added to any set
                            else:
                                # Renew second half space
                                PointsH_2 = np.ndarray(shape=(len(self.C_B), self.dimension_E))
                                counter = 0
                                for i in self.C_B:
                                    PointsH_2[counter] = self.get_point(i)
                                    counter += 1
                                self.C_B = ConvexHull(PointsH_2, 1)
                                if next_point in outside_points_1:
                                    outside_points_1.remove(next_point)
                                if next_point in outside_points_2:
                                    outside_points_2.remove(next_point)

                    time_point = time_step("Point add Hull:", time_point)


                # Second set is nearer to nearest not classified point
                else:

                    time_point = time_step("Find Neighbour:", time_point)

                    if len(self.convex_B_hull_distances) > 0:
                        next_point = self.convex_B_hull_distances.popitem()[0]
                        Hull2 = self.C_B
                        Hull2.add_points([self.get_point(next_point)], 1)
                        inside2, added = get_points_inside_convex_hull(self.E, Hull2, self.C_B,
                                                                       outside_points_2)
                        oracle_calls += 1

                        # we can add the new point to the second, the nearer set
                        if not intersect(inside2, self.C_A):
                            self.C_B = Hull2
                            for x in added:
                                # add to labels
                                labels[x] = 0
                                self.colors_E[x] = "violet"
                                if x in self.convex_A_hull_distances.keys():
                                    del self.convex_A_hull_distances[x]
                                if x in self.convex_B_hull_distances.keys():
                                    del self.convex_B_hull_distances[x]
                                outside_points_2.remove(x)
                            self.C_B = inside2



                        # we check if we can add the point to the first set
                        # if we reach this point the first time all the other E which are classified did not change the optimal margin
                        else:
                            # Renew second half space
                            PointsH_2 = np.ndarray(shape=(len(self.C_B), self.dimension_E))
                            counter = 0
                            for i in self.C_B:
                                PointsH_2[counter] = self.get_point(i)
                                counter += 1
                            self.C_B = ConvexHull(PointsH_2, 1)

                            # Test first half space
                            Hull1 = self.C_A
                            Hull1.add_points([self.get_point(next_point)], 1)
                            inside1, added = get_points_inside_convex_hull(self.E, Hull1, self.C_A,
                                                                           outside_points_1)
                            oracle_calls += 1

                            # the point can be classified to the second set
                            if not intersect(self.C_B, inside1):
                                self.C_A = Hull1
                                for x in added:
                                    # add to labels
                                    labels[x] = 1
                                    self.colors_E[x] = "orange"
                                    if x in self.convex_A_hull_distances.keys():
                                        del self.convex_A_hull_distances[x]
                                    if x in self.convex_B_hull_distances.keys():
                                        del self.convex_B_hull_distances[x]
                                    outside_points_1.remove(x)
                                self.C_A = inside1




                            # we cannot classify the point
                            else:
                                # Renew first half space
                                PointsH_1 = np.ndarray(shape=(len(self.C_A), self.dimension_E))
                                counter = 0
                                for i in self.C_A:
                                    PointsH_1[counter] = self.get_point(i)
                                    counter += 1
                                self.C_A = ConvexHull(PointsH_1, 1)
                                if next_point in outside_points_1:
                                    outside_points_1.remove(next_point)
                                if next_point in outside_points_2:
                                    outside_points_2.remove(next_point)

                time_point = time_step("Point add Hull:", time_point)
        else:
            return ([], [], False)

        return (oracle_calls, labels, True)

    def optimal_runtime_alg(self):
        oracle_calls = 0
        end = False
        counter = 0
        while len(self.F) > 0 and end == False:
            # print(len(self.F))
            if not self.decide_farthest():
                items = list(self.convex_A_distances.items())
                if len(items) > 0:
                    next_point = items[len(items) - 1][0]

                    Hull1 = self.C_A
                    Hull1.add_points([self.get_point(next_point)], 1)
                    inside1 = get_points_inside_convex_hull(self.E, Hull1)
                    oracle_calls += 1
                    if not intersect(inside1, self.C_B):
                        self.C_A.add_points([self.get_point(next_point)], 1)
                        for x in inside1:
                            if x not in self.C_A:
                                self.colors_E[x] = "orange"
                                self.F.remove(x)
                                del self.convex_A_distances[x]
                                del self.convex_B_distances[x]
                        self.C_A = inside1

                    else:
                        # Renew first half space
                        PointsH_1 = np.ndarray(shape=(len(self.C_A), 2))
                        counter = 0
                        for i in self.C_A:
                            PointsH_1[counter] = self.get_point(i)
                            counter += 1
                        self.C_A = ConvexHull(PointsH_1, 1)

                        # Test second half space
                        Hull2 = self.C_B
                        Hull2.add_points([self.get_point(next_point)], 1)
                        inside2 = get_points_inside_convex_hull(self.E, Hull2)
                        oracle_calls += 1

                        if not intersect(self.C_A, inside2):
                            self.C_B.add_points([self.get_point(next_point)], 1)
                            for x in inside2:
                                if x not in self.C_B:
                                    self.colors_E[x] = "violet"
                                    self.F.remove(x)
                                    del self.convex_A_distances[x]
                                    del self.convex_B_distances[x]
                            self.C_B = inside2
                        else:
                            # Renew second half space
                            PointsH_2 = np.ndarray(shape=(len(self.C_B), 2))
                            counter = 0
                            for i in self.C_B:
                                PointsH_2[counter] = self.get_point(i)
                                counter += 1
                            self.C_B = ConvexHull(PointsH_2, 1)
                            self.F.remove(next_point)
                            del self.convex_A_distances[next_point]
                            del self.convex_B_distances[next_point]

            else:
                items = list(self.convex_B_distances.items())
                if len(items) > 0:
                    next_point = items[len(items) - 1][0]
                    Hull2 = self.C_B
                    Hull2.add_points([self.get_point(next_point)], 1)
                    inside2 = get_points_inside_convex_hull(self.E, Hull2, self.C_B)
                    oracle_calls += 1

                    if not intersect(inside2, self.C_A):
                        self.C_B.add_points([self.get_point(next_point)], 1)
                        for x in inside2:
                            if x not in self.C_B:
                                self.colors_E[x] = "violet"
                                self.F.remove(x)
                                del self.convex_A_distances[x]
                                del self.convex_B_distances[x]
                        self.C_B = inside2

                    else:
                        # Renew second half space
                        PointsH_2 = np.ndarray(shape=(len(self.C_B), 2))
                        counter = 0
                        for i in self.C_B:
                            PointsH_2[counter] = self.get_point(i)
                            counter += 1
                        self.C_B = ConvexHull(PointsH_2, 1)

                        # Test first half space
                        Hull1 = self.C_A
                        Hull1.add_points([self.get_point(next_point)], 1)
                        inside1 = get_points_inside_convex_hull(self.E, Hull1)
                        oracle_calls += 1

                        if not intersect(self.C_B, inside1):
                            self.C_A.add_points([self.get_point(next_point)], 1)
                            for x in inside1:
                                if x not in self.C_A:
                                    self.colors_E[x] = "orange"
                                    self.F.remove(x)
                                    del self.convex_A_distances[x]
                                    del self.convex_B_distances[x]
                            self.C_A = inside1
                        else:
                            # Renew first half space
                            PointsH_1 = np.ndarray(shape=(len(self.C_A), 2))
                            counter = 0
                            for i in self.C_A:
                                PointsH_1[counter] = self.get_point(i)
                                counter += 1
                            self.C_A = ConvexHull(PointsH_1, 1)
                            self.F.remove(next_point)
                            del self.convex_A_distances[next_point]
                            del self.convex_B_distances[next_point]

        return oracle_calls


def dist(x, y):
    return sqrt(np.dot(x - y, x - y))


def intersect(Hull1, Hull2):
    for i in Hull1:
        if i in Hull2:
            return True
    return False


def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b, method="simplex")
    return lp.success


def get_inside_points(Points, X, Outside, add_point="", CheckSet=""):
    inside_points = X.copy()
    added_points = []
    if add_point != "":
        inside_points.append(add_point)
        added_points.append(add_point)

    inside_points_array = np.zeros(shape=(len(inside_points), len(Points[0])))
    counter = 0
    for i in inside_points:
        for j in range(0, len(Points[i])):
            inside_points_array[counter][j] = Points[i][j]
        counter += 1

    if CheckSet != "":
        for i in CheckSet:
            if in_hull(inside_points_array, Points[i]):
                return [], [], True

    for i in Outside:
        if in_hull(inside_points_array, Points[i]):
            inside_points.append(i)
            added_points.append(i)
    return inside_points, added_points, False


def get_points_inside_convex_hull(E, convex_hull, inside_p, outside_points, added_point=""):
    inside_points = inside_p.copy()
    added_points = []
    for i in outside_points:
        point = E[i]
        inside = True
        for face in convex_hull.equations:
            c = np.dot(point, face[:-1]) + face[-1]  # point[0]*face[0] + point[1]*face[1] + face[2]
            # print(i, point, c)
            if 1e-14 > c > 0:  # 0.0000000000000005
                c = 0
            if c > 0:
                inside = False
                break
        if inside == True:
            inside_points.append(i)
            added_points.append(i)

    if added_point != "":
        inside_points.append(added_point)
        added_points.append(added_point)

    return inside_points, added_points


def random_point_set(number, dimension, start_points1_number, start_points2_number, field_size=1):
    correct = False
    Point_List = []

    while correct == False:
        correct = True
        X_coord = field_size * np.random.rand(number // 2)
        Y_coord = field_size * np.random.rand(number // 2)
        if dimension == 3:
            Z_coord = field_size * np.random.rand(number // 2)

        X_coord = np.append(X_coord, 0 + field_size * np.random.rand(number // 2))
        Y_coord = np.append(Y_coord, 0 + field_size * np.random.rand(number // 2))
        if dimension == 3:
            Z_coord = np.append(Z_coord, 0 + field_size * np.random.rand(number // 2))

        Point_List = np.ndarray(shape=(number, dimension))

        if dimension == 3:
            for i in range(0, number):
                Point_List[i] = [X_coord[i], Y_coord[i], Z_coord[i]]
        else:
            for i in range(0, number):
                Point_List[i] = [X_coord[i], Y_coord[i]]

        # start point dense
        # set the start E 0 and number//2 and find neighbours
        start_pointsA = [0]
        start_pointsB = [number // 2]
        dist_points = 0.2
        while (len(start_pointsA) != start_points1_number) or (len(start_pointsB) != start_points2_number):
            dist_points += 0.1
            for i in range(number):
                if i not in start_pointsA and i not in start_pointsB:
                    if len(start_pointsA) < start_points1_number:
                        if dist(Point_List[i], Point_List[0]) < dist_points:
                            start_pointsA.append(i)
                    if len(start_pointsB) < start_points2_number:
                        if dist(Point_List[i], Point_List[number // 2]) < dist_points:
                            start_pointsB.append(i)

        # print(len(start_pointsA), len(start_pointsB))

        # initialize first half space
        HullPoints1 = start_pointsA
        PointsH_1 = np.ndarray(shape=(len(HullPoints1), dimension))
        counter = 0
        outside_points = [x for x in range(number) if x not in HullPoints1]
        for i in HullPoints1:
            PointsH_1[counter] = Point_List[i]
            counter += 1

        ConvexHull1 = ConvexHull(PointsH_1, 1)
        insideH_1, added = get_points_inside_convex_hull(Point_List, ConvexHull1, HullPoints1, outside_points)
        HullPoints1 = insideH_1

        # Initialize second half space
        HullPoints2 = start_pointsB
        PointsH_2 = np.ndarray(shape=(len(HullPoints2), dimension))
        counter = 0
        outside_points = [x for x in range(number) if x not in HullPoints2]
        for i in HullPoints2:
            PointsH_2[counter] = Point_List[i]
            counter += 1

        ConvexHull2 = ConvexHull(PointsH_2, 1)

        insideH_2, added = get_points_inside_convex_hull(Point_List, ConvexHull2, HullPoints2, outside_points)
        HullPoints2 = insideH_2

        # Only allow those cases where there is no intersection of the convex hull at the beginning
        if intersect(HullPoints1, HullPoints2) == True:
            correct = False
            continue

    return (Point_List, HullPoints1, HullPoints2)


def generate_start_points(class_label, labels, number):
    pos_labels = [x for x in range(len(labels)) if labels[x] == class_label]
    return random.sample(pos_labels, number)


def time_step(string_name, time_point, level=1):
    time_dur = time.time() - time_point
    time_point = time.time()
    tabs = ""
    for i in range(level):
        tabs += "\t"
    # print(tabs, string_name, time_dur, "s")
    return time_point


def plot_svm(ax, model):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
    # plot support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')


def plot(Point_List, Color_List, dim=3, name="Test", model=""):
    if dim == 3:
        X_coord = []
        Y_coord = []
        Z_coord = []

        ax = plt.axes(projection='3d')

        for i in range(len(Point_List)):
            X_coord.append(Point_List[i][0])
            Y_coord.append(Point_List[i][1])
            Z_coord.append(Point_List[i][2])

        ax.scatter3D(X_coord, Y_coord, Z_coord, c=Color_List)
    else:
        ax = plt.axes()

        x_val_dict = {}
        y_val_dict = {}

        for i, x in enumerate(Color_List, 0):
            if x not in x_val_dict:
                x_val_dict[x] = [Point_List[i][0]]
                y_val_dict[x] = [Point_List[i][1]]
            else:
                x_val_dict[x].append(Point_List[i][0])
                y_val_dict[x].append(Point_List[i][1])

        for key, value in x_val_dict.items():
            plt.scatter(value, y_val_dict[key], c=key)

        if model:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # create grid to evaluate model
            xx = np.linspace(xlim[0], xlim[1], 30)
            yy = np.linspace(ylim[0], ylim[1], 30)
            YY, XX = np.meshgrid(yy, xx)
            xy = np.vstack([XX.ravel(), YY.ravel()]).T
            Z = model.decision_function(xy).reshape(XX.shape)

            # plot decision boundary and margins
            plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=1,
                        linestyles=['--', '-', '--'])
            # plot support vectors
            plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                        linewidth=1, facecolors='none', edgecolors='k')

        tikzplotlib.save(name + ".tex")
        plt.show()

    # tikz_save("/home/florian/Dokumente/Forschung/EigeneForschung/SpringerLatex/" + "PlotOneClass" + ".tex", wrap = False)


def color_list(Label_List):
    color_list = []
    for x in Label_List:
        if x == 0:
            color_list.append("green")
        elif x == 1:
            color_list.append("red")

    return color_list


def color_list_testing(labels, train_a, train_b):
    colorlist = np.where(labels == 0, "orange", "violet")
    colorlist[train_a] = ["blue"]
    colorlist[train_b] = ["red"]
    return colorlist


def color_list_result(Node_List, Label_List, pos_points, neg_points):
    color_list = []
    for x in range(len(Node_List)):
        if x in pos_points:
            color_list.append("red")
        elif x in neg_points:
            color_list.append("green")
        else:
            if Label_List[x] == 0:
                color_list.append("violet")
            elif Label_List[x] == 1:
                color_list.append("orange")
            elif Label_List[x] == -1:
                color_list.append("blue")

    return color_list


def color_list_training(Node_List, pos_points, neg_points):
    color_list = []
    for x in range(len(Node_List)):
        if x in pos_points:
            color_list.append("red")
        elif x in neg_points:
            color_list.append("green")
        else:
            color_list.append("blue")
    return color_list


def print_error_evaluation(name, X, y, classification, pos_points, neg_points):
    num_training = (len(pos_points) + len(neg_points))
    num = len(X)
    num_test = (num - num_training)
    red = 0
    green = 0

    # Fehlerauswertung
    error = 0
    error_red = 0
    error_green = 0
    unclassified_red = 0
    unclassified_green = 0
    correct_red = 0
    correct_green = 0

    counter = 0
    for x in classification.flatten():
        label = y.flatten()[counter]

        if label == 0:
            green += 1
        else:
            red += 1

        # errors
        if x == -1:
            if label == 1:
                unclassified_red += 1
            else:
                unclassified_green += 1
        elif x == 1:
            if label == 1:
                correct_red += 1
            else:
                error_red += 1
        elif x == 0:
            if label == 0:
                correct_green += 1
            else:
                error_green += 1
        counter += 1

    correct_green -= len(neg_points)
    correct_red -= len(pos_points)

    error = error_green + error_red
    correct = correct_green + correct_red
    accuracy = correct / (correct + error)
    unclassified = unclassified_green + unclassified_red
    recall = (correct + error) / num_test
    print(name)
    print("Accuracy", correct / (error + correct), "Correct:", correct, "Error: ", error, "Unclassified: ",
          unclassified, "Training: ", len(pos_points) + len(neg_points))


"""If saving a classification result to a sqlite check if db exists otherwise create"""


def create_table(db_name, table_name, column_list):
    if not os.path.isfile(db_name):
        con = sqlite3.connect(db_name)
        cur = con.cursor()

        string = ""
        for entry in column_list:
            string += entry
            string += ","
        string = string[:-1]
        cur.execute("CREATE TABLE " + table_name + " (" + string + ");")


"""Saving classification result to database"""


def get_data_values(X, y, classification, pos_points, neg_points):
    num_training = (len(pos_points) + len(neg_points))
    num = len(X)
    num_test = (num - num_training)
    red = 0
    green = 0

    # Fehlerauswertung
    error = 0
    error_red = 0
    error_green = 0
    unclassified_red = 0
    unclassified_green = 0
    correct_red = 0
    correct_green = 0

    counter = 0
    for x in classification.flatten():
        label = y.flatten()[counter]

        if label == 0:
            green += 1
        else:
            red += 1

        # errors
        if x == -1:
            if label == 1:
                unclassified_red += 1
            else:
                unclassified_green += 1
        elif x == 1:
            if label == 1:
                correct_red += 1
            else:
                error_red += 1
        elif x == 0:
            if label == 0:
                correct_green += 1
            else:
                error_green += 1
        counter += 1

    correct_green -= len(neg_points)
    correct_red -= len(pos_points)

    error = error_green + error_red
    correct = correct_green + correct_red
    accuracy = correct / (correct + error)
    unclassified = unclassified_green + unclassified_red
    recall = (correct + error) / num_test

    # ['Timestamp','Accuracy','DefaultVal','Recall','Correct','Wrong','Unclassified','RedCorrect','RedFalse','RedUnclassified','GreenCorrect','GreenFalse','GreenUnclassified','Num','NumberRed','NumberGreen','NumberBlue','NumTrainingRed','NumTrainingGreen']

    values = [[time.time(), round(float(accuracy), 2), max(red, green) / num, round(float(recall), 2), int(correct),
               int(error), int(unclassified), int(correct_red), int(error_red), int(unclassified_red),
               int(correct_green), int(error_green), int(unclassified_green), int(num), int(red), int(green),
               int(unclassified), int(len(pos_points)), int(len(neg_points))]]
    return values


"""Saving classification result to database"""


def add_row_to_database(databasename, table_name, column_list, X, y, classification, pos_points, neg_points):
    con = sqlite3.connect(databasename)
    cur = con.cursor()

    string = ""
    for entry in column_list:
        string += entry
        string += ","

    num_training = (len(pos_points) + len(neg_points))
    num = len(X)
    num_test = (num - num_training)
    red = 0
    green = 0

    # Fehlerauswertung
    error = 0
    error_red = 0
    error_green = 0
    unclassified_red = 0
    unclassified_green = 0
    correct_red = 0
    correct_green = 0

    counter = 0
    for x in classification.flatten():
        label = y.flatten()[counter]

        if label == 0:
            green += 1
        else:
            red += 1

        # errors
        if x == -1:
            if label == 1:
                unclassified_red += 1
            else:
                unclassified_green += 1
        elif x == 1:
            if label == 1:
                correct_red += 1
            else:
                error_red += 1
        elif x == 0:
            if label == 0:
                correct_green += 1
            else:
                error_green += 1
        counter += 1

    correct_green -= len(neg_points)
    correct_red -= len(pos_points)

    error = error_green + error_red
    correct = correct_green + correct_red
    accuracy = correct / (correct + error)
    unclassified = unclassified_green + unclassified_red
    recall = (correct + error) / num_test

    # ['Timestamp','Accuracy','DefaultVal','Recall','Correct','Wrong','Unclassified','RedCorrect','RedFalse','RedUnclassified','GreenCorrect','GreenFalse','GreenUnclassified','Num','NumberRed','NumberGreen','NumberBlue','NumTrainingRed','NumTrainingGreen']

    to_db = [[time.time(), round(float(accuracy), 2), max(red, green) / num, round(float(recall), 2), int(correct),
              int(error), int(unclassified), int(correct_red), int(error_red), int(unclassified_red),
              int(correct_green), int(error_green), int(unclassified_green), int(num), int(red), int(green),
              int(unclassified), int(len(pos_points)), int(len(neg_points))]]

    cur.executemany(
        "INSERT INTO " + table_name + " (" + string[:-1] + ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);", to_db)
    con.commit()
    con.close()


def set_training_testing(E, E_labels, A_size, B_size, seed=0):
    # Generate samples
    A_elements = []
    B_elements = []

    A_B_vectors = np.zeros(shape=(A_size + B_size, E.shape[1]))
    test_points = np.zeros(shape=(len(E) - A_size + B_size, E.shape[1]))
    A_B_labels = np.zeros(shape=(A_size + B_size, 1))
    test_labels = np.zeros(shape=(len(E) - A_size + B_size, 1))

    counter = 0
    random_training = np.zeros(shape=(len(E), 1))
    while counter != len(E) or (len(A_elements) < A_size and len(B_elements) < B_size):
        # random.seed(seed)
        elem = random.randint(0, len(E) - 1)
        if random_training[elem] == 0:
            if E_labels[elem] == 1 and len(A_elements) < A_size:
                A_elements.append(elem)
            elif E_labels[elem] == 0 and len(B_elements) < B_size:
                B_elements.append(elem)
            random_training[elem] = 1
            counter += 1

    # E training and test sets
    counter = 0
    counter1 = 0
    counter2 = 0
    for x in E:
        if counter in A_elements:
            A_B_vectors[counter1] = E[counter]
            A_B_labels[counter1] = 1
            counter1 += 1
        elif counter in B_elements:
            A_B_vectors[counter1] = E[counter]
            A_B_labels[counter1] = 0
            counter1 += 1
        else:
            test_points[counter2] = E[counter]
            test_labels[counter2] = E_labels[counter]
            counter2 += 1
        counter += 1

    return A_elements, B_elements, A_B_vectors, A_B_labels, test_points, test_labels


def load_data(file_path, max_number, n_features=3, max_labels=1):
    number_of_points = 5000
    n_features = 5
    n_labels = 1

    data, meta = arff.loadarff("/home/florian/scikit_learn_data/mozilla4.arff")
    print(data)
    X = np.zeros(shape=(min(len(range(number_of_points)), len(data)), n_features))
    y = np.zeros(shape=(min(len(range(number_of_points)), len(data)), n_labels))
    counter = 0
    label_name = ""
    for row in data:
        if counter < number_of_points:
            for i in range(n_features):
                X[counter][i] = row[i]
            if counter == 0:
                label_name = row[n_features]
            if row[n_features] == label_name:
                y[counter][0] = 0
            else:
                y[counter][0] = 1
        counter += 1
    # E, E_labels = _convert_arff_data(data, 3, 1)
    # E, E_labels = load_breast_cancer(True)
