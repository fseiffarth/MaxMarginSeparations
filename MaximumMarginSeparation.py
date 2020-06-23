import time

import numpy as np
import networkx as nx
from DataToSQL import *


class MaximumMarginSeparation:
    def __init__(self, ground_set, closure_operator, linkage_function):
        self.ground_set = ground_set

        self.closure_operator = closure_operator
        self.linkage_function = linkage_function

        self.closed_set_a = []
        self.closed_set_b = []

    def sorted_linkages(self, closure_a, closure_b):
        """

        :rtype: tuple of linkage value and minimal set a or b
        """

        min_linkage_values = np.zeros(len(closure_a), dtype=np.float64)
        min_linkage_values.fill(-1)
        min_linkage_set = np.zeros(len(closure_a), dtype=np.float64)
        elements = np.arange(len(closure_a))

        F = np.logical_not(np.logical_or(closure_a, closure_b))

        for x in np.nonzero(F)[0]:
            linkage_a = self.linkage_function(closure_a, x)
            linkage_b = self.linkage_function(closure_b, x)
            if linkage_a < linkage_b:
                min_linkage_values[x] = linkage_a
                min_linkage_set[x] = 1
            elif linkage_a > linkage_b:
                min_linkage_values[x] = linkage_b
                min_linkage_set[x] = 2
            else:
                min_linkage_values[x] = linkage_a
                min_linkage_set[x] = np.random.randint(1, 3)

        p = min_linkage_values.argsort()
        elements = elements[p]
        min_linkage_set = min_linkage_set[p]
        min_linkage_values = min_linkage_values[p]

        return elements, min_linkage_values, min_linkage_set

    def add_element(self, ground_set, element):
        ground_set[element] = 1
        return ground_set

    def remove_element(self, ground_set, element):
        ground_set[element] = 0
        return ground_set

    def sets_intersecting(self, set_a, set_b):
        return len(np.nonzero(np.logical_and(set_a, set_b))[0])

    def maximum_margin_separation(self, set_a, set_b):
        start = time.time()
        closure_a = self.closure_operator(set_a)
        closure_b = self.closure_operator(set_b)
        print("Initial Closure: ", time.time() - start)

        H_1 = closure_a.copy()
        H_2 = closure_b.copy()

        F = np.logical_not(np.logical_or(H_1, H_2))

        start = time.time()
        elements, min_linkage_values, min_linkage_set = self.sorted_linkages(closure_a, closure_b)
        print("Linkage", time.time() - start)

        if not self.sets_intersecting(closure_a, closure_b):
            while len(np.nonzero(F)[0]):
                for x in np.nonzero(min_linkage_values + 1)[0]:
                    if F[elements[x]]:
                        F[elements[x]] = 0
                        if min_linkage_set[x] == 1:
                            self.add_element(H_1, elements[x])
                            new_closure_1 = self.closure_operator(np.nonzero(H_1)[0])
                            if not self.sets_intersecting(new_closure_1, H_2):
                                H_1 = new_closure_1
                                F = np.logical_and(F, np.logical_not(H_1))
                            else:
                                self.remove_element(H_1, elements[x])
                                self.add_element(H_2, elements[x])
                                new_closure_2 = self.closure_operator(np.nonzero(H_2)[0])
                                if not self.sets_intersecting(new_closure_2, H_1):
                                    H_2 = new_closure_2
                                    F = np.logical_and(F, np.logical_not(H_2))
                                else:
                                    self.remove_element(H_2, elements[x])
                        else:
                            self.add_element(H_2, elements[x])
                            new_closure_2 = self.closure_operator(np.nonzero(H_2)[0])
                            if not self.sets_intersecting(new_closure_2, H_1):
                                H_2 = new_closure_2
                                F = np.logical_and(F, np.logical_not(H_2))
                            else:
                                self.add_element(H_1, elements[x])
                                self.remove_element(H_2, elements[x])
                                new_closure_1 = self.closure_operator(np.nonzero(H_1)[0])
                                if not self.sets_intersecting(new_closure_1, H_2):
                                    H_1 = new_closure_1
                                    F = np.logical_and(F, np.logical_not(H_1))
                                else:
                                    self.remove_element(H_1, elements[x])

            return H_1, H_2
        else:
            print("A, B not separable by disjoint sets")
            return -1

    def max_margin_classification(self, set_a, set_b, target_set_a, target_set_b):
        if self.maximum_margin_separation(set_a, set_b) == -1:
            return -1
        else:
            self.closed_set_a, self.closed_set_b = self.maximum_margin_separation(set_a, set_b)

            target_a = np.zeros((len(self.closed_set_a)), dtype=np.bool)
            target_b = np.zeros((len(self.closed_set_a)), dtype=np.bool)

            np.put(target_a, target_set_a, np.ones(len(target_set_a), dtype=np.bool))
            np.put(target_b, target_set_b, np.ones(len(target_set_b), dtype=np.bool))

            target_a_correct = len(np.nonzero(np.logical_and(self.closed_set_a, target_a))[0])
            target_b_correct = len(np.nonzero(np.logical_and(self.closed_set_b, target_b))[0])

            return target_a_correct, target_b_correct


    def result_to_database(self):
        database = DataToSQL(file_path="")
