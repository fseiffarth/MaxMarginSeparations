import random
import sys

import networkx as nx
import numpy as np
from ordered_set import OrderedSet

from MCSSAlgos.Domains.DomainBase import Domain
from MCSSAlgos.Domains.Graphs import Graph
from MCSSAlgos.InitUpdate.InitUpdateBase import InitUpdate, PartialOrder


def get_query_strategy(name: str, threshold: int = sys.maxsize):
    switch = {"greedy": GraphInitUpdateGreedy(),
              "greedy_random": GraphInitUpdateGreedy(random_init=True),
              "nn": GraphInitUpdateNN(threshold=threshold),
              "farthest": GraphInitUpdateFarthestPoint(),
              }
    try:
        return switch[name]
    except:
        raise IndexError(
            "{} is not a valid query strategy. Use one from these [greedy, greedy_random, nn, farthest]".format(name))


class GraphPartialOrder(PartialOrder):
    def __init__(self):
        super(GraphPartialOrder, self).__init__()

    def pop_minimum_element(self):
        if len(self.order) > self.argmin_val:
            e = random.sample(self.order[self.argmin_val], 1)[0]
            self.order[self.argmin_val].remove(e)
        else:
            raise IndexError("The index of argmin_val and order length do not fit together.")
        return e


class GraphInitUpdateGreedy(InitUpdate):
    def __init__(self, random_init: bool = False):
        super(GraphInitUpdateGreedy, self).__init__(GraphPartialOrder())
        self.random_init = random_init

    def init(self, domain: Graph, initial_sets: list, unlabeled_elements: set):
        for x in unlabeled_elements:
            initial_sets_nums = list(range(len(initial_sets)))
            if self.random_init:
                self.v[x] = OrderedSet(random.sample(initial_sets_nums, len(initial_sets_nums)))
            else:
                self.v[x] = OrderedSet(initial_sets_nums)
            try:
                self.partial_order.order[self.partial_order.argmin_val].add(x)
            except:
                self.partial_order.order[self.partial_order.argmin_val] = {x}

    def update(self, domain: Graph, extended_object: int, added_objects: set, invalid_extensions: set) -> set:
        self.partial_order.order[self.partial_order.argmin_val].difference_update(added_objects)


class GraphInitUpdateFarthestPoint(InitUpdate):
    def __init__(self, random_extend: bool = True):
        super(GraphInitUpdateFarthestPoint, self).__init__(GraphPartialOrder())
        self.random_extend = random_extend
        self.distances = np.array([])
        self.nearest_set = np.array([])
        self.distances_sort_dict = {}

    def init(self, domain: Graph, initial_sets: list, unlabeled_elements: set):
        self.distances = np.zeros(domain.size, dtype=np.int)
        self.distances.fill(-1)
        self.nearest_set = np.zeros(domain.size, dtype=np.int)
        self.nearest_set.fill(-1)
        for i, _set in enumerate(initial_sets):
            for x in _set:
                self.distances[x] = 0
                sp = nx.single_source_shortest_path(domain.data_object, x)
                for node, shortest_path in sp.items():
                    if node in unlabeled_elements:
                        if self.distances[node] == -1:
                            self.distances[node] = len(shortest_path) - 1
                            self.nearest_set[node] = i
                        else:
                            if self.distances[node] >= len(shortest_path) - 1 and self.random_extend:
                                if self.distances[node] > len(shortest_path) - 1:
                                    self.distances[node] = len(shortest_path) - 1
                                    self.nearest_set[node] = i
                                else:
                                    self.nearest_set[node] = random.sample([self.nearest_set[node], i], i)[0]

        self.distances_sort_dict = dict(enumerate(np.sort(np.unique(self.distances))[::-1]))

        # define partial order
        self.partial_order.argmin_val = 0
        for elem in unlabeled_elements:
            try:
                self.partial_order.order[self.distances_sort_dict[self.distances[elem]]].add(elem)
            except:
                self.partial_order.order[self.distances_sort_dict[self.distances[elem]]] = {elem}
            self.v[elem] = OrderedSet()
            self.v[elem].add(self.nearest_set[elem])
            sample_set = set(range(len(initial_sets)))
            sample_set.remove(self.nearest_set[elem])
            samples = random.sample(sample_set, len(initial_sets) - 1)
            for i in samples:
                self.v[elem].add(i)

    def update(self, domain: Graph, extended_object: int, added_objects: set, invalid_extensions: set) -> set:
        for elem in added_objects:
            try:
                self.partial_order.order[self.distances_sort_dict[self.distances[elem]]].remove(elem)
            except:
                pass
        while len(self.partial_order.order[self.partial_order.argmin_val]) == 0 and self.partial_order.argmin_val+1 in self.partial_order.order:
            self.partial_order.argmin_val += 1


class GraphInitUpdateNN(InitUpdate):
    def __init__(self, random_extend: bool = True, threshold: int = sys.maxsize):
        super(GraphInitUpdateNN, self).__init__(GraphPartialOrder())
        self.random_extend = random_extend
        self.threshold = threshold

    def init(self, domain: Graph, initial_sets: list, unlabeled_elements: set):
        self.partial_order.order[0] = set.union(*initial_sets)
        self.partial_order.argmin_val = 1
        for i, _set in enumerate(initial_sets):
            for elem in _set:
                self.w[elem] = 0
                neighbors = domain.data_object[elem]
                for n in neighbors:
                    if n in unlabeled_elements:
                        self.w[n] = 1
                        try:
                            self.partial_order.order[1].add(n)
                        except:
                            self.partial_order.order[1] = {n}
                        try:
                            self.v[n]
                            if random.randint(0, 1) == 0:
                                first_index = self.v[n][0]
                                index_i = self.v[n].index(self.v[n][0])
                                self.v[n][index_i] = first_index
                        except:
                            self.v[n] = OrderedSet()
                            self.v[n].add(i)
                            sample_set = set(range(len(initial_sets)))
                            sample_set.remove(i)
                            samples = random.sample(sample_set, len(initial_sets) - 1)
                            for j in samples:
                                self.v[n].add(j)

    def update(self, domain: Domain, extended_object: int, added_objects: set, invalid_extensions: set) -> set:
        for added_object in added_objects.union({extended_object}):
            try:
                self.w[added_object]
                try:
                    self.partial_order.order[self.w[added_object]].remove(added_object)
                except:
                    pass
            except:
                self.w[added_object] = 0

            for neighbor in domain.data_object[added_object]:
                # update the partial order
                try:
                    self.w[neighbor]
                except:
                    self.w[neighbor] = self.w[added_object] + 1
                    try:
                        self.partial_order.order[self.w[neighbor]].add(neighbor)
                    except:
                        self.partial_order.order[self.w[neighbor]] = {neighbor}

                # update the tags
                if self.w[neighbor] == self.w[added_object] + 1:
                    for tag in self.v[added_object]:
                        if self.threshold != sys.maxsize or tag not in invalid_extensions:
                            try:
                                self.v[neighbor].add(tag)
                            except:
                                self.v[neighbor] = OrderedSet()
                                self.v[neighbor].add(tag)
                        else:
                            try:
                                self.v[neighbor].remove(tag)
                            except:
                                pass

        while len(self.partial_order.order[self.partial_order.argmin_val]) == 0 and self.partial_order.argmin_val+1 in self.partial_order.order:
            self.partial_order.argmin_val += 1
