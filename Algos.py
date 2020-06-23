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
from sklearn.datasets.openml import fetch_openml, _download_data_arff, \
    _convert_arff_data
from scipy.io import arff
import tikzplotlib
from PyGEL3D import gel


####Class for the point set in which separation is done
class PointSet(object):
    def __init__(self, Set, Hull1, Hull2, isManifoldTrue=False):
        self.field_size = math.inf
        self.N = len(Set)
        self.dim = len(Set[0])
        self.color = []

        # convex hulls
        self.Set1 = Hull1  # convex hull 1
        self.Set2 = Hull2  # convex hull 2
        self.Set1_size = len(Hull1)
        self.Set2_size = len(Hull2)

        # set colors
        for i in range(self.N):
            if i in self.Set1:
                self.color.append("red")
            elif i in self.Set2:
                self.color.append("green")
            else:
                self.color.append("b")

        # point distances
        self.Set1Distances = {}
        self.Set2Distances = {}
        self.Set1HullDistances = {}
        self.Set2HullDistances = {}

        self.Point_List = Set

        self.H_1 = self.Set1
        PointsH_1 = np.ndarray(shape=(len(self.H_1), self.dim))
        counter = 0
        for i in self.H_1:
            PointsH_1[counter] = self.get_point(i)
            counter += 1

        self.ConvexHull1 = ConvexHull(PointsH_1, 1)

        self.H_2 = self.Set2
        PointsH_2 = np.ndarray(shape=(len(self.H_2), self.dim))
        counter = 0
        for i in self.H_2:
            PointsH_2[counter] = self.get_point(i)
            counter += 1

        self.ConvexHull2 = ConvexHull(PointsH_2, 1)

        if isManifoldTrue:
            # manifolds from hull
            self.m1 = gel.Manifold()
            for s in self.ConvexHull1.simplices:
                self.m1.add_face(self.ConvexHull1.points[s])

            self.m1dist = gel.MeshDistance(self.m1)
            self.m2 = gel.Manifold()
            for s in self.ConvexHull2.simplices:
                self.m2.add_face(self.ConvexHull1.points[s])

            self.m2dist = gel.MeshDistance(self.m2)

        self.F = []

        for i in range(0, self.N):
            if i not in self.H_1 and i not in self.H_2:
                self.F.append(i)

        self.set_set1_neighbors()
        self.set_set2_neighbors()

    def plot_2d_classification(self, name="Test", colorlist = []):

        # initialize first half space
        PointsH_1 = np.ndarray(shape=(len(self.H_1), self.dim))
        counter = 0
        for i in self.H_1:
            PointsH_1[counter] = self.get_point(i)
            counter += 1
        self.ConvexHull1 = ConvexHull(PointsH_1, 1)

        # Initialize second half space
        PointsH_2 = np.ndarray(shape=(len(self.H_2), self.dim))
        counter = 0
        for i in self.H_2:
            PointsH_2[counter] = self.get_point(i)
            counter += 1
        self.ConvexHull2 = ConvexHull(PointsH_2, 1)

        # Draw convex hulls of disjoint convex sets
        for simplex in self.ConvexHull1.simplices:
            plt.plot(self.ConvexHull1.points[simplex, 0], self.ConvexHull1.points[simplex, 1], 'k-')
        for simplex in self.ConvexHull2.simplices:
            plt.plot(self.ConvexHull2.points[simplex, 0], self.ConvexHull2.points[simplex, 1], 'k-')

        x_val_dict = {}
        y_val_dict = {}

        if colorlist == []:
            colorlist = self.color

        for i, x in enumerate(colorlist, 0):
            if x not in x_val_dict:
                x_val_dict[x] = [self.Point_List[i][0]]
                y_val_dict[x] = [self.Point_List[i][1]]
            else:
                x_val_dict[x].append(self.Point_List[i][0])
                y_val_dict[x].append(self.Point_List[i][1])

        for key, value in x_val_dict.items():
            plt.scatter(value, y_val_dict[key], c=key)

        tikzplotlib.save(name + ".tex")
        plt.show()

    def plot_3d_classification(self):

        # initialize first half space
        PointsH_1 = np.ndarray(shape=(len(self.H_1), self.dim))
        counter = 0
        for i in self.H_1:
            PointsH_1[counter] = self.get_point(i)
            counter += 1

        self.ConvexHull1 = ConvexHull(PointsH_1, 1)

        # Initialize second half space
        PointsH_2 = np.ndarray(shape=(len(self.H_2), self.dim))
        counter = 0
        for i in self.H_2:
            PointsH_2[counter] = self.get_point(i)
            counter += 1

        self.ConvexHull2 = ConvexHull(PointsH_2, 1)

        # print(self.ConvexHull1.equations)

        ax = plt.axes(projection='3d')

        for simplex in self.ConvexHull1.simplices:
            ax.plot3D(self.ConvexHull1.points[simplex, 0], self.ConvexHull1.points[simplex, 1],
                      self.ConvexHull1.points[simplex, 2], 'k-')
        for simplex in self.ConvexHull2.simplices:
            ax.plot3D(self.ConvexHull2.points[simplex, 0], self.ConvexHull2.points[simplex, 1],
                      self.ConvexHull2.points[simplex, 2], 'k-')

        X_coord = np.ones(self.N)
        Y_coord = np.ones(self.N)
        Z_coord = np.ones(self.N)
        for i in range(self.N):
            X_coord[i] = self.Point_List[i][0]
            Y_coord[i] = self.Point_List[i][1]
            Z_coord[i] = self.Point_List[i][2]

        ax.scatter3D(X_coord, Y_coord, Z_coord, c=self.color)
        plt.show()

    def get_point(self, n):
        return self.Point_List[n]

    def set_set1_neighbors(self, is_manifold=False):
        for n in self.F:
            min_dist = sys.maxsize
            for x in self.Set1:
                point_dist = dist(self.get_point(n), self.get_point(x))
                if point_dist < min_dist:
                    min_dist = point_dist
            self.Set1Distances[n] = min_dist

            if is_manifold:
                d = self.m1dist.signed_distance(self.get_point(n))
                self.Set1HullDistances[n] = d * np.sign(d)
        self.Set1Distances = OrderedDict(sorted(self.Set1Distances.items(), key=lambda x: x[1], reverse=True))

        if is_manifold:
            self.Set1HullDistances = OrderedDict(
                sorted(self.Set1HullDistances.items(), key=lambda x: x[1], reverse=True))

    def set_set2_neighbors(self, is_manifold=False):
        for n in self.F:
            min_dist = self.field_size * self.field_size
            for x in self.Set2:
                point_dist = dist(self.get_point(n), self.get_point(x))
                if (point_dist < min_dist):
                    min_dist = point_dist
            self.Set2Distances[n] = min_dist

            if is_manifold:
                d = self.m2dist.signed_distance(self.get_point(n))
                self.Set1HullDistances[n] = d * np.sign(d)
        self.Set2Distances = OrderedDict(sorted(self.Set2Distances.items(), key=lambda x: x[1], reverse=True))

        if is_manifold:
            self.Set2HullDistances = OrderedDict(sorted(self.Set2Distances.items(), key=lambda x: x[1], reverse=True))

    def decide_nearest(self):
        len1 = len(self.Set1Distances)
        len2 = len(self.Set2Distances)

        if len1 == 0:
            return 0
        elif len2 == 0:
            return 1
        elif next(reversed(self.Set1Distances.values())) <= next(reversed(self.Set2Distances.values())):
            return 1
        else:
            return 0

    def decide_nearest_hull(self):
        len1 = len(self.Set1HullDistances)
        len2 = len(self.Set2HullDistances)

        if len1 == 0:
            return 0
        elif len2 == 0:
            return 1
        elif next(reversed(self.Set1HullDistances.values())) <= next(reversed(self.Set2HullDistances.values())):
            return 1
        else:
            return 0

    def decide_farthest(self):
        len1 = len(self.Set1Distances)
        len2 = len(self.Set2Distances)

        if len1 == 0:
            return 0
        elif len2 == 0:
            return 1
        elif (next(iter(self.Set1Distances.values())) <= next(iter(self.Set2Distances.values()))):
            return 1
        else:
            return 0

    def add_set1_nearest(self):
        items = list(self.Set1Distances.items())
        nearest = items[0]
        point = nearest[0]
        self.H_1.append(point)
        self.color[point] = "orange"
        self.F.remove(point)
        del self.Set1Distances[point]
        del self.Set2Distances[point]

    def add_set2_nearest(self):
        items = list(self.Set2Distances.items())
        nearest = items[0]
        point = nearest[0]
        self.H_2.append(point)
        self.color[point] = "violet"
        self.F.remove(point)
        del self.Set2Distances[point]
        del self.Set1Distances[point]

    # SeCos-Algorithm from paper applied to convex hulls in R^d
    def greedy_alg(self):
        end = False
        oracle_calls = 0
        random.shuffle(self.F)

        # label vector of the elements (binary classification {-1, 1}
        labels = -1 * np.ones(shape=(self.N, 1))
        outside_1 = self.F + self.H_2
        outside_2 = self.F + self.H_1

        # Set labels of initial labelled data
        for i in self.H_1:
            labels[i] = 1
        for i in self.H_2:
            labels[i] = 0

        # Set initial hulls
        Hull1 = self.ConvexHull1
        inside1, added = get_points_inside_convex_hull(self.Point_List, Hull1, self.H_1, self.F + self.H_2)
        oracle_calls += 1

        if not intersect(inside1, self.H_2):
            for x in added:
                labels[x] = 1
                self.color[x] = "orange"
            self.H_1 = inside1

        # Set initial hulls
        Hull2 = self.ConvexHull2
        inside2, added = get_points_inside_convex_hull(self.Point_List, Hull2, self.H_2, self.F + self.H_1)
        oracle_calls += 1

        if not intersect(inside2, self.H_1):
            for x in added:
                labels[x] = 0
                self.color[x] = "violet"
            self.H_2 = inside2

        if not intersect(inside1, inside2):
            for x in self.H_1:
                if x in self.F:
                    self.F.remove(x)
            for x in self.H_2:
                if x in self.F:
                    self.F.remove(x)

            while len(self.F) > 0 and end == False:
                # print(len(self.F))
                next_point = self.F.pop()
                Hull1 = self.ConvexHull1
                Hull1.add_points([self.get_point(next_point)], 1)
                inside1, added = get_points_inside_convex_hull(self.Point_List, Hull1, self.H_1, self.F + self.H_2,
                                                               next_point)
                oracle_calls += 1

                if not intersect(inside1, self.H_2):
                    self.ConvexHull1.add_points([self.get_point(next_point)], 1)
                    for x in added:
                        labels[x] = 1
                        self.color[x] = "orange"
                        if x in self.F:
                            self.F.remove(x)
                    self.H_1 = inside1

                else:
                    # initialize first half space
                    PointsH_1 = np.ndarray(shape=(len(self.H_1), self.dim))
                    counter = 0
                    for i in self.H_1:
                        PointsH_1[counter] = self.get_point(i)
                        counter += 1

                    self.ConvexHull1 = ConvexHull(PointsH_1, 1)

                    Hull2 = self.ConvexHull2
                    Hull2.add_points([self.get_point(next_point)], 1)
                    inside2, added = get_points_inside_convex_hull(self.Point_List, Hull2, self.H_2, self.F + self.H_1,
                                                                   next_point)
                    oracle_calls += 1

                    if not intersect(self.H_1, inside2):
                        self.ConvexHull2.add_points([self.get_point(next_point)], 1)
                        for x in added:
                            labels[x] = 0
                            self.color[x] = "violet"
                            if x in self.F:
                                self.F.remove(x)
                        self.H_2 = inside2
                    else:
                        # initialize second half space
                        PointsH_2 = np.ndarray(shape=(len(self.H_2), self.dim))
                        counter = 0
                        for i in self.H_2:
                            PointsH_2[counter] = self.get_point(i)
                            counter += 1
                        self.ConvexHull2 = ConvexHull(PointsH_2, 1)
        else:
            return ([], [], False)

        return (oracle_calls, labels, True)

    def greedy_fast_alg(self):
        end = False
        oracle_calls = 0
        random.shuffle(self.F)

        # label vector of the elements (binary classification {1, 0} -1 are unclassified
        labels = -1 * np.ones(shape=(self.N, 1))
        outside_1 = self.F + self.H_2
        outside_2 = self.F + self.H_1

        # Set labels of initial labelled data
        for i in self.H_1:
            labels[i] = 1
        for i in self.H_2:
            labels[i] = 0

        # Set initial hulls
        inside1, added, intersection = get_inside_points(self.Point_List, self.H_1, self.F, CheckSet=self.H_2)
        oracle_calls += 1

        if not intersection:
            for x in added:
                labels[x] = 1
                self.color[x] = "orange"
            self.H_1 = inside1

        # Set initial hulls
        inside2, added, intersection = get_inside_points(self.Point_List, self.H_2, self.F, CheckSet=self.H_1)
        oracle_calls += 1

        if not intersection:
            for x in added:
                labels[x] = 0
                self.color[x] = "violet"
            self.H_2 = inside2

        if not intersect(inside1, inside2):
            for x in self.H_1:
                if x in self.F:
                    self.F.remove(x)
            for x in self.H_2:
                if x in self.F:
                    self.F.remove(x)

            while len(self.F) > 0 and end == False:
                #print(len(self.F))
                next_point = self.F.pop()
                inside1, added, intersection = get_inside_points(self.Point_List, self.H_1, self.F, next_point,
                                                                 self.H_2)
                oracle_calls += 1

                if not intersection:
                    for x in added:
                        labels[x] = 1
                        self.color[x] = "orange"
                        if x in self.F:
                            self.F.remove(x)
                    self.H_1 = inside1

                else:
                    inside2, added, intersection = get_inside_points(self.Point_List, self.H_2, self.F, next_point,
                                                                     self.H_1)
                    oracle_calls += 1

                    if not intersection:
                        for x in added:
                            labels[x] = 0
                            self.color[x] = "violet"
                            if x in self.F:
                                self.F.remove(x)
                        self.H_2 = inside2
        else:
            return ([], [], False)

        return (oracle_calls, labels, True)

    def greedy_alg2(self):
        end = False
        oracle_calls = 0
        random.shuffle(self.F)

        # label vector of the elements (binary classification {-1, 1}
        labels = -1 * np.ones(shape=(self.N, 1))
        outside_1 = self.F + self.H_2
        outside_2 = self.F + self.H_1

        # Set labels of initial labelled data
        for i in self.H_1:
            labels[i] = 1
        for i in self.H_2:
            labels[i] = 0

        # Set initial hulls
        Hull1 = self.ConvexHull1
        inside1, added = get_points_inside_convex_hull(self.Point_List, Hull1, self.H_1, self.F + self.H_2)
        oracle_calls += 1

        if not intersect(inside1, self.H_2):
            for x in added:
                labels[x] = 1
                self.color[x] = "orange"
            self.H_1 = inside1

        # Set initial hulls
        Hull2 = self.ConvexHull2
        inside2, added = get_points_inside_convex_hull(self.Point_List, Hull2, self.H_2, self.F + self.H_1)
        oracle_calls += 1

        if not intersect(inside2, self.H_1):
            for x in added:
                labels[x] = 0
                self.color[x] = "violet"
            self.H_2 = inside2

        if not intersect(inside1, inside2):
            for x in self.H_1:
                if x in self.F:
                    self.F.remove(x)
            for x in self.H_2:
                if x in self.F:
                    self.F.remove(x)

            while len(self.F) > 0 and end == False:
                #print(len(self.F))
                next_point = self.F.pop()

                if (random.randint(0, 1)):
                    Hull1 = self.ConvexHull1
                    Hull1.add_points([self.get_point(next_point)], 1)
                    inside1, added = get_points_inside_convex_hull(self.Point_List, Hull1, self.H_1, self.F + self.H_2,
                                                                   next_point)
                    oracle_calls += 1

                    if not intersect(inside1, self.H_2):
                        self.ConvexHull1.add_points([self.get_point(next_point)], 1)
                        for x in added:
                            labels[x] = 1
                            self.color[x] = "orange"
                            if x in self.F:
                                self.F.remove(x)
                        self.H_1 = inside1

                    else:
                        # initialize first half space
                        PointsH_1 = np.ndarray(shape=(len(self.H_1), self.dim))
                        counter = 0
                        for i in self.H_1:
                            PointsH_1[counter] = self.get_point(i)
                            counter += 1

                        self.ConvexHull1 = ConvexHull(PointsH_1, 1)

                        Hull2 = self.ConvexHull2
                        Hull2.add_points([self.get_point(next_point)], 1)
                        inside2, added = get_points_inside_convex_hull(self.Point_List, Hull2, self.H_2,
                                                                       self.F + self.H_1, next_point)
                        oracle_calls += 1

                        if not intersect(self.H_1, inside2):
                            self.ConvexHull2.add_points([self.get_point(next_point)], 1)
                            for x in added:
                                labels[x] = 0
                                self.color[x] = "violet"
                                if x in self.F:
                                    self.F.remove(x)
                            self.H_2 = inside2
                        else:
                            # initialize second half space
                            PointsH_2 = np.ndarray(shape=(len(self.H_2), self.dim))
                            counter = 0
                            for i in self.H_2:
                                PointsH_2[counter] = self.get_point(i)
                                counter += 1

                            self.ConvexHull2 = ConvexHull(PointsH_2, 1)

                else:
                    Hull2 = self.ConvexHull2
                    Hull2.add_points([self.get_point(next_point)], 1)
                    inside2, added = get_points_inside_convex_hull(self.Point_List, Hull2, self.H_2, self.F + self.H_1,
                                                                   next_point)
                    oracle_calls += 1

                    if not intersect(inside2, self.H_1):
                        self.ConvexHull2.add_points([self.get_point(next_point)], 1)
                        for x in added:
                            labels[x] = 0
                            self.color[x] = "violet"
                            if x in self.F:
                                self.F.remove(x)
                        self.H_2 = inside2

                    else:
                        # initialize first half space
                        PointsH_2 = np.ndarray(shape=(len(self.H_2), self.dim))
                        counter = 0
                        for i in self.H_2:
                            PointsH_2[counter] = self.get_point(i)
                            counter += 1

                        self.ConvexHull2 = ConvexHull(PointsH_2, 1)

                        Hull1 = self.ConvexHull1
                        Hull1.add_points([self.get_point(next_point)], 1)
                        inside1, added = get_points_inside_convex_hull(self.Point_List, Hull1, self.H_1,
                                                                       self.F + self.H_2, next_point)
                        oracle_calls += 1

                        if not intersect(self.H_2, inside1):
                            self.ConvexHull1.add_points([self.get_point(next_point)], 1)
                            for x in added:
                                labels[x] = 1
                                self.color[x] = "orange"
                                if x in self.F:
                                    self.F.remove(x)
                            self.H_1 = inside1
                        else:
                            # initialize second half space
                            PointsH_1 = np.ndarray(shape=(len(self.H_1), self.dim))
                            counter = 0
                            for i in self.H_1:
                                PointsH_1[counter] = self.get_point(i)
                                counter += 1

                            self.ConvexHull1 = ConvexHull(PointsH_1, 1)
        else:
            return ([], [], False)

        return (oracle_calls, labels, True)

    def optimal_alg(self):
        time_point = time.time()
        oracle_calls = 0
        counter = 0
        labels = -1 * np.ones(shape=(self.N, 1))
        outside_points_1 = [x for x in self.Set1Distances.keys()] + self.H_2
        outside_points_2 = [x for x in self.Set2Distances.keys()] + self.H_1

        # add labels
        for i in self.H_1:
            labels[i] = 1
        for i in self.H_2:
            labels[i] = 0

        # check if hulls are intersecting
        Hull1 = self.ConvexHull1
        inside1, added = get_points_inside_convex_hull(self.Point_List, Hull1, self.H_1, outside_points_1)
        self.H_1 = inside1
        Hull2 = self.ConvexHull2
        inside2, added = get_points_inside_convex_hull(self.Point_List, Hull2, self.H_2, outside_points_2)
        self.H_2 = inside2

        if not intersect(inside1, inside2):
            while (len(self.Set1Distances) > 0 or len(self.Set2Distances) > 0):
                #print(len(self.Set1Distances), len(self.Set2Distances))
                added = []

                # First set is nearer to nearest not classified point
                if self.decide_nearest():

                    time_point = time_step("Find Neighbour:", time_point)

                    if len(self.Set1Distances) > 0:
                        next_point = self.Set1Distances.popitem()[0]

                        Hull1 = self.ConvexHull1
                        Hull1.add_points([self.get_point(next_point)], 1)
                        time_point = time_step("Adding Convex Hull points 1:", time_point)

                        inside1, added = get_points_inside_convex_hull(self.Point_List, Hull1, self.H_1,
                                                                       outside_points_1)
                        oracle_calls += 1
                        time_point = time_step("Getting inside points:", time_point)

                        # if there is no intersection the point can be added to the first convex set
                        if not intersect(inside1, self.H_2):
                            time_point = time_step("Intersection Test:", time_point)
                            self.ConvexHull1 = Hull1
                            time_point = time_step("Adding Convex Hull points:", time_point)

                            for x in added:
                                # add to labels
                                labels[x] = 1
                                self.color[x] = "orange"
                                if x in self.Set1Distances.keys():
                                    del self.Set1Distances[x]
                                if x in self.Set2Distances.keys():
                                    del self.Set2Distances[x]

                                outside_points_1.remove(x)
                            self.H_1 = inside1

                            time_point = time_step("Update arrays:", time_point)


                        # if there is an intersection we have to check if it can be added to the second set
                        else:
                            time_point = time_step("Intersection Test:", time_point)
                            # Renew first half space
                            PointsH_1 = np.ndarray(shape=(len(self.H_1), self.dim))
                            counter = 0
                            for i in self.H_1:
                                PointsH_1[counter] = self.get_point(i)
                                counter += 1
                            self.ConvexHull1 = ConvexHull(PointsH_1, 1)

                            # Test second half space
                            Hull2 = self.ConvexHull2
                            Hull2.add_points([self.get_point(next_point)], 1)
                            inside2, added = get_points_inside_convex_hull(self.Point_List, Hull2, self.H_2,
                                                                           outside_points_2)
                            oracle_calls += 1

                            # the point can be added to the second set,
                            # if we reach this point the first time all the other points which are classified did not change the optimal margin
                            if not intersect(self.H_1, inside2):
                                self.ConvexHull2 = Hull2
                                for x in added:
                                    # add to labels
                                    labels[x] = 0
                                    self.color[x] = "violet"
                                    if x in self.Set1Distances.keys():
                                        del self.Set1Distances[x]
                                    if x in self.Set2Distances.keys():
                                        del self.Set2Distances[x]
                                    outside_points_2.remove(x)
                                self.H_2 = inside2


                            # the point cannot be added to any set
                            else:
                                # Renew second half space
                                PointsH_2 = np.ndarray(shape=(len(self.H_2), self.dim))
                                counter = 0
                                for i in self.H_2:
                                    PointsH_2[counter] = self.get_point(i)
                                    counter += 1
                                self.ConvexHull2 = ConvexHull(PointsH_2, 1)
                                if next_point in outside_points_1:
                                    outside_points_1.remove(next_point)
                                if next_point in outside_points_2:
                                    outside_points_2.remove(next_point)

                    time_point = time_step("Point add Hull:", time_point)


                # Second set is nearer to nearest not classified point
                else:

                    time_point = time_step("Find Neighbour:", time_point)

                    if len(self.Set2Distances) > 0:
                        next_point = self.Set2Distances.popitem()[0]
                        Hull2 = self.ConvexHull2
                        Hull2.add_points([self.get_point(next_point)], 1)
                        inside2, added = get_points_inside_convex_hull(self.Point_List, Hull2, self.H_2,
                                                                       outside_points_2)
                        oracle_calls += 1

                        # we can add the new point to the second, the nearer set
                        if not intersect(inside2, self.H_1):
                            self.ConvexHull2 = Hull2
                            for x in added:
                                # add to labels
                                labels[x] = 0
                                self.color[x] = "violet"
                                if x in self.Set1Distances.keys():
                                    del self.Set1Distances[x]
                                if x in self.Set2Distances.keys():
                                    del self.Set2Distances[x]
                                outside_points_2.remove(x)
                            self.H_2 = inside2



                        # we check if we can add the point to the first set
                        # if we reach this point the first time all the other points which are classified did not change the optimal margin
                        else:
                            # Renew second half space
                            PointsH_2 = np.ndarray(shape=(len(self.H_2), self.dim))
                            counter = 0
                            for i in self.H_2:
                                PointsH_2[counter] = self.get_point(i)
                                counter += 1
                            self.ConvexHull2 = ConvexHull(PointsH_2, 1)

                            # Test first half space
                            Hull1 = self.ConvexHull1
                            Hull1.add_points([self.get_point(next_point)], 1)
                            inside1, added = get_points_inside_convex_hull(self.Point_List, Hull1, self.H_1,
                                                                           outside_points_1)
                            oracle_calls += 1

                            # the point can be classified to the second set
                            if not intersect(self.H_2, inside1):
                                self.ConvexHull1 = Hull1
                                for x in added:
                                    # add to labels
                                    labels[x] = 1
                                    self.color[x] = "orange"
                                    if x in self.Set1Distances.keys():
                                        del self.Set1Distances[x]
                                    if x in self.Set2Distances.keys():
                                        del self.Set2Distances[x]
                                    outside_points_1.remove(x)
                                self.H_1 = inside1




                            # we cannot classify the point
                            else:
                                # Renew first half space
                                PointsH_1 = np.ndarray(shape=(len(self.H_1), self.dim))
                                counter = 0
                                for i in self.H_1:
                                    PointsH_1[counter] = self.get_point(i)
                                    counter += 1
                                self.ConvexHull1 = ConvexHull(PointsH_1, 1)
                                if next_point in outside_points_1:
                                    outside_points_1.remove(next_point)
                                if next_point in outside_points_2:
                                    outside_points_2.remove(next_point)

                time_point = time_step("Point add Hull:", time_point)
        else:
            return ([], [], False)

        return (oracle_calls, labels, True)

    def optimal_hull_alg(self):
        time_point = time.time()
        oracle_calls = 0
        counter = 0
        labels = -1 * np.ones(shape=(self.N, 1))
        outside_points_1 = [x for x in self.Set1HullDistances.keys()] + self.H_2
        outside_points_2 = [x for x in self.Set2HullDistances.keys()] + self.H_1

        # add labels
        for i in self.H_1:
            labels[i] = 1
        for i in self.H_2:
            labels[i] = 0

        # check if hulls are intersecting
        Hull1 = self.ConvexHull1
        inside1, added = get_points_inside_convex_hull(self.Point_List, Hull1, self.H_1, outside_points_1)
        self.H_1 = inside1
        Hull2 = self.ConvexHull2
        inside2, added = get_points_inside_convex_hull(self.Point_List, Hull2, self.H_2, outside_points_2)
        self.H_2 = inside2

        if not intersect(inside1, inside2):
            while (len(self.Set1HullDistances) > 0 or len(self.Set2HullDistances) > 0):
                #print(len(self.Set1HullDistances), len(self.Set2HullDistances))
                added = []

                # First set is nearer to nearest not classified point
                if self.decide_nearest_hull():

                    time_point = time_step("Find Neighbour:", time_point)

                    if len(self.Set1HullDistances) > 0:
                        next_point = self.Set1HullDistances.popitem()[0]

                        Hull1 = self.ConvexHull1
                        Hull1.add_points([self.get_point(next_point)], 1)
                        time_point = time_step("Adding Convex Hull points 1:", time_point)

                        inside1, added = get_points_inside_convex_hull(self.Point_List, Hull1, self.H_1,
                                                                       outside_points_1)
                        oracle_calls += 1
                        time_point = time_step("Getting inside points:", time_point)

                        # if there is no intersection the point can be added to the first convex set
                        if not intersect(inside1, self.H_2):
                            time_point = time_step("Intersection Test:", time_point)
                            self.ConvexHull1 = Hull1
                            time_point = time_step("Adding Convex Hull points:", time_point)

                            for x in added:
                                # add to labels
                                labels[x] = 1
                                self.color[x] = "orange"
                                if x in self.Set1HullDistances.keys():
                                    del self.Set1HullDistances[x]
                                if x in self.Set2HullDistances.keys():
                                    del self.Set2HullDistances[x]

                                outside_points_1.remove(x)
                            self.H_1 = inside1

                            time_point = time_step("Update arrays:", time_point)


                        # if there is an intersection we have to check if it can be added to the second set
                        else:
                            time_point = time_step("Intersection Test:", time_point)
                            # Renew first half space
                            PointsH_1 = np.ndarray(shape=(len(self.H_1), self.dim))
                            counter = 0
                            for i in self.H_1:
                                PointsH_1[counter] = self.get_point(i)
                                counter += 1
                            self.ConvexHull1 = ConvexHull(PointsH_1, 1)

                            # Test second half space
                            Hull2 = self.ConvexHull2
                            Hull2.add_points([self.get_point(next_point)], 1)
                            inside2, added = get_points_inside_convex_hull(self.Point_List, Hull2, self.H_2,
                                                                           outside_points_2)
                            oracle_calls += 1

                            # the point can be added to the second set,
                            # if we reach this point the first time all the other points which are classified did not change the optimal margin
                            if not intersect(self.H_1, inside2):
                                self.ConvexHull2 = Hull2
                                for x in added:
                                    # add to labels
                                    labels[x] = 0
                                    self.color[x] = "violet"
                                    if x in self.Set1HullDistances.keys():
                                        del self.Set1HullDistances[x]
                                    if x in self.Set2HullDistances.keys():
                                        del self.Set2HullDistances[x]
                                    outside_points_2.remove(x)
                                self.H_2 = inside2


                            # the point cannot be added to any set
                            else:
                                # Renew second half space
                                PointsH_2 = np.ndarray(shape=(len(self.H_2), self.dim))
                                counter = 0
                                for i in self.H_2:
                                    PointsH_2[counter] = self.get_point(i)
                                    counter += 1
                                self.ConvexHull2 = ConvexHull(PointsH_2, 1)
                                if next_point in outside_points_1:
                                    outside_points_1.remove(next_point)
                                if next_point in outside_points_2:
                                    outside_points_2.remove(next_point)

                    time_point = time_step("Point add Hull:", time_point)


                # Second set is nearer to nearest not classified point
                else:

                    time_point = time_step("Find Neighbour:", time_point)

                    if len(self.Set2HullDistances) > 0:
                        next_point = self.Set2HullDistances.popitem()[0]
                        Hull2 = self.ConvexHull2
                        Hull2.add_points([self.get_point(next_point)], 1)
                        inside2, added = get_points_inside_convex_hull(self.Point_List, Hull2, self.H_2,
                                                                       outside_points_2)
                        oracle_calls += 1

                        # we can add the new point to the second, the nearer set
                        if not intersect(inside2, self.H_1):
                            self.ConvexHull2 = Hull2
                            for x in added:
                                # add to labels
                                labels[x] = 0
                                self.color[x] = "violet"
                                if x in self.Set1HullDistances.keys():
                                    del self.Set1HullDistances[x]
                                if x in self.Set2HullDistances.keys():
                                    del self.Set2HullDistances[x]
                                outside_points_2.remove(x)
                            self.H_2 = inside2



                        # we check if we can add the point to the first set
                        # if we reach this point the first time all the other points which are classified did not change the optimal margin
                        else:
                            # Renew second half space
                            PointsH_2 = np.ndarray(shape=(len(self.H_2), self.dim))
                            counter = 0
                            for i in self.H_2:
                                PointsH_2[counter] = self.get_point(i)
                                counter += 1
                            self.ConvexHull2 = ConvexHull(PointsH_2, 1)

                            # Test first half space
                            Hull1 = self.ConvexHull1
                            Hull1.add_points([self.get_point(next_point)], 1)
                            inside1, added = get_points_inside_convex_hull(self.Point_List, Hull1, self.H_1,
                                                                           outside_points_1)
                            oracle_calls += 1

                            # the point can be classified to the second set
                            if not intersect(self.H_2, inside1):
                                self.ConvexHull1 = Hull1
                                for x in added:
                                    # add to labels
                                    labels[x] = 1
                                    self.color[x] = "orange"
                                    if x in self.Set1HullDistances.keys():
                                        del self.Set1HullDistances[x]
                                    if x in self.Set2HullDistances.keys():
                                        del self.Set2HullDistances[x]
                                    outside_points_1.remove(x)
                                self.H_1 = inside1




                            # we cannot classify the point
                            else:
                                # Renew first half space
                                PointsH_1 = np.ndarray(shape=(len(self.H_1), self.dim))
                                counter = 0
                                for i in self.H_1:
                                    PointsH_1[counter] = self.get_point(i)
                                    counter += 1
                                self.ConvexHull1 = ConvexHull(PointsH_1, 1)
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
                items = list(self.Set1Distances.items())
                if len(items) > 0:
                    next_point = items[len(items) - 1][0]

                    Hull1 = self.ConvexHull1
                    Hull1.add_points([self.get_point(next_point)], 1)
                    inside1 = get_points_inside_convex_hull(self.Point_List, Hull1)
                    oracle_calls += 1
                    if not intersect(inside1, self.H_2):
                        self.ConvexHull1.add_points([self.get_point(next_point)], 1)
                        for x in inside1:
                            if x not in self.H_1:
                                self.color[x] = "orange"
                                self.F.remove(x)
                                del self.Set1Distances[x]
                                del self.Set2Distances[x]
                        self.H_1 = inside1

                    else:
                        # Renew first half space
                        PointsH_1 = np.ndarray(shape=(len(self.H_1), 2))
                        counter = 0
                        for i in self.H_1:
                            PointsH_1[counter] = self.get_point(i)
                            counter += 1
                        self.ConvexHull1 = ConvexHull(PointsH_1, 1)

                        # Test second half space
                        Hull2 = self.ConvexHull2
                        Hull2.add_points([self.get_point(next_point)], 1)
                        inside2 = get_points_inside_convex_hull(self.Point_List, Hull2)
                        oracle_calls += 1

                        if not intersect(self.H_1, inside2):
                            self.ConvexHull2.add_points([self.get_point(next_point)], 1)
                            for x in inside2:
                                if x not in self.H_2:
                                    self.color[x] = "violet"
                                    self.F.remove(x)
                                    del self.Set1Distances[x]
                                    del self.Set2Distances[x]
                            self.H_2 = inside2
                        else:
                            # Renew second half space
                            PointsH_2 = np.ndarray(shape=(len(self.H_2), 2))
                            counter = 0
                            for i in self.H_2:
                                PointsH_2[counter] = self.get_point(i)
                                counter += 1
                            self.ConvexHull2 = ConvexHull(PointsH_2, 1)
                            self.F.remove(next_point)
                            del self.Set1Distances[next_point]
                            del self.Set2Distances[next_point]

            else:
                items = list(self.Set2Distances.items())
                if len(items) > 0:
                    next_point = items[len(items) - 1][0]
                    Hull2 = self.ConvexHull2
                    Hull2.add_points([self.get_point(next_point)], 1)
                    inside2 = get_points_inside_convex_hull(self.Point_List, Hull2, self.H_2)
                    oracle_calls += 1

                    if not intersect(inside2, self.H_1):
                        self.ConvexHull2.add_points([self.get_point(next_point)], 1)
                        for x in inside2:
                            if x not in self.H_2:
                                self.color[x] = "violet"
                                self.F.remove(x)
                                del self.Set1Distances[x]
                                del self.Set2Distances[x]
                        self.H_2 = inside2

                    else:
                        # Renew second half space
                        PointsH_2 = np.ndarray(shape=(len(self.H_2), 2))
                        counter = 0
                        for i in self.H_2:
                            PointsH_2[counter] = self.get_point(i)
                            counter += 1
                        self.ConvexHull2 = ConvexHull(PointsH_2, 1)

                        # Test first half space
                        Hull1 = self.ConvexHull1
                        Hull1.add_points([self.get_point(next_point)], 1)
                        inside1 = get_points_inside_convex_hull(self.Point_List, Hull1)
                        oracle_calls += 1

                        if not intersect(self.H_2, inside1):
                            self.ConvexHull1.add_points([self.get_point(next_point)], 1)
                            for x in inside1:
                                if x not in self.H_1:
                                    self.color[x] = "orange"
                                    self.F.remove(x)
                                    del self.Set1Distances[x]
                                    del self.Set2Distances[x]
                            self.H_1 = inside1
                        else:
                            # Renew first half space
                            PointsH_1 = np.ndarray(shape=(len(self.H_1), 2))
                            counter = 0
                            for i in self.H_1:
                                PointsH_1[counter] = self.get_point(i)
                                counter += 1
                            self.ConvexHull1 = ConvexHull(PointsH_1, 1)
                            self.F.remove(next_point)
                            del self.Set1Distances[next_point]
                            del self.Set2Distances[next_point]

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


def get_points_inside_convex_hull(Set, Hull, inside, outside_points, added_point=""):
    inside_points = inside.copy()
    added_points = []
    for i in outside_points:
        point = Set[i]
        inside = True
        for face in Hull.equations:
            c = np.dot(point, face[:-1]) + face[-1]  # point[0]*face[0] + point[1]*face[1] + face[2]
            # print(i, point, c)
            if c < 1e-14 and c > 0:  # 0.0000000000000005
                c = 0
            if (c > 0):
                inside = False
                break
        if (inside == True):
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
        # set the start points 0 and number//2 and find neighbours
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


def set_training_testing(X, y, num_red, num_green, seed=0):
    # Generate samples
    pos_points = []
    neg_points = []

    training_points = np.zeros(shape=(num_red + num_green, X.shape[1]))
    test_points = np.zeros(shape=(len(X) - num_red + num_green, X.shape[1]))
    training_labels = np.zeros(shape=(num_red + num_green, 1))
    test_labels = np.zeros(shape=(len(X) - num_red + num_green, 1))

    counter = 0
    random_training = np.zeros(shape=(len(X), 1))
    while counter != len(X) or (len(pos_points) < num_red and len(neg_points) < num_green):
        #random.seed(seed)
        elem = random.randint(0, len(X) - 1)
        if random_training[elem] == 0:
            if y[elem] == 1 and len(pos_points) < num_red:
                pos_points.append(elem)
            elif y[elem] == 0 and len(neg_points) < num_green:
                neg_points.append(elem)
            random_training[elem] = 1
            counter += 1

    # Set training and test sets
    counter = 0
    counter1 = 0
    counter2 = 0
    for x in X:
        if counter in pos_points:
            training_points[counter1] = X[counter]
            training_labels[counter1] = 1
            counter1 += 1
        elif counter in neg_points:
            training_points[counter1] = X[counter]
            training_labels[counter1] = 0
            counter1 += 1
        else:
            test_points[counter2] = X[counter]
            test_labels[counter2] = y[counter]
            counter2 += 1
        counter += 1

    return pos_points, neg_points, training_points, training_labels, test_points, test_labels


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
    # X, y = _convert_arff_data(data, 3, 1)
    # X, y = load_breast_cancer(True)
