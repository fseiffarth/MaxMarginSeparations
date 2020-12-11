'''
Created on 10.08.2018

@author: florian
'''
import sys
import time
import random
import math
from collections import OrderedDict
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


from Algos import get_points_inside_convex_hull, intersect, get_inside_points, time_step, dist


####Class for the point set in which separation is done
class BinarySeparation(object):
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
        self.F = list(set(range(self.size_E)) - set([x for x in self.C_A or x in self.C_B]))

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

        # Draw convex initial_hulls of disjoint convex sets
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

    # SeCos-Algorithm from paper applied to convex initial_hulls in R^d
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

        # E initial initial_hulls
        inside1, added, intersection = get_inside_points(self.E, self.C_A, self.F, CheckSet=self.C_B)
        oracle_calls += 1

        if not intersection:
            for x in added:
                labels[x] = 1
                self.colors_E[x] = "orange"
            self.C_A = inside1

        # E initial initial_hulls
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

        # E initial initial_hulls
        Hull1 = self.C_A
        inside1, added = get_points_inside_convex_hull(self.E, Hull1, self.C_A, self.F + self.C_B)
        oracle_calls += 1

        if not intersect(inside1, self.C_B):
            for x in added:
                labels[x] = 1
                self.colors_E[x] = "orange"
            self.C_A = inside1

        # E initial initial_hulls
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

        # check if initial_hulls are intersecting
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
