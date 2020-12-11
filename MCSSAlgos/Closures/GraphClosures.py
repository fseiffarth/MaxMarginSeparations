import collections
import sys
from typing import Optional

from MCSSAlgos.Closures.ClosureBase import Closure
import networkx as nx
import numpy as np

from MCSSAlgos.Domains.DomainBase import Domain


def get_graph_closure(name: str):
    if name == "sp":
        return GraphClosureSPFast()
    elif name[:2] == "sp" and int(name[2:]) > 0:
        return GraphClosureSPFast(int(name[2:]))
    else:
        raise IndexError("{} is not an valid parameter for a graph closure".format(name))


class GraphClosureNoPaths(Closure):
    def cl(self, elements: set, data_object: Domain, forbidden_elements: set = {}, added_element: int = -1):
        return elements, set()


class GraphClosureSPFast(Closure):
    def __init__(self, threshold: Optional[int] = sys.maxsize):
        super(GraphClosureSPFast, self).__init__()
        self.threshold = threshold

    def cl(self, elements: set, domain: Domain, forbidden_elements: set = {}, added_element: int = -1):
        def add_element_to_closed(closed_set: set, element: int, threshold: Optional[int] = sys.maxsize):
            new_closed = [element]
            while new_closed:
                next_extend_element = new_closed.pop()
                try:
                    new_closed = list(set(new_closed).union(bfs_search(next_extend_element, closed_set, threshold)))
                except:
                    return False
            return True

        def bfs_search(element: int, closed_interval_set: set, threshold: Optional[int] = sys.maxsize):
            closed_interval_set.add(element)
            new_elements = set()
            distance_list = np.zeros(domain.data_object.number_of_nodes(), dtype=np.int)
            distance_list.fill(-2)
            in_element = np.zeros(domain.data_object.number_of_nodes(), dtype=bool)
            in_element[list(closed_interval_set)] = 1

            search_dict = {element: []}

            # forward search through tree O(m)
            distance_list[element] = 0
            queue = collections.deque()
            queue.append(element)
            visited_set = {element}
            visited_elements_in_queue = 1
            while queue and (len(visited_set) != len(closed_interval_set) or visited_elements_in_queue != 0):
                s = queue.pop()
                if 0 < threshold < distance_list[s]:
                    break
                if in_element[s]:
                    visited_elements_in_queue -= 1
                if domain.is_tree and (s != element and in_element[s]):
                    continue
                else:
                    for neighbour in domain.data_object[s]:
                        if distance_list[neighbour] < 0:
                            if in_element[neighbour]:
                                visited_set.add(neighbour)
                                visited_elements_in_queue += 1
                            next_distance = distance_list[s] + 1
                            if next_distance > threshold:
                                break
                            distance_list[neighbour] = next_distance
                            queue.appendleft(neighbour)
                        if distance_list[neighbour] == distance_list[s] + 1:
                            try:
                                search_dict[neighbour].append(s)
                            except:
                                search_dict[neighbour] = [s]

            # backward search through tree O(m)
            distance_list.fill(0)
            distance_list[list(closed_interval_set)] = 1
            queue = collections.deque()
            for e in closed_interval_set:
                queue.append(e)
            while queue:
                s = queue.pop()
                try:
                    search_dict[s]
                    for neighbour in search_dict[s]:
                        if distance_list[neighbour] == 0:
                            if neighbour in forbidden_elements:
                                return False
                            distance_list[neighbour] = 1
                            closed_interval_set.add(neighbour)
                            new_elements.add(neighbour)
                            queue.appendleft(neighbour)

                except:
                    pass

            return new_elements

        closed_set = set()
        i = 0

        # Add only one element to elements (the result is only closed if elements was closed before)
        if added_element != -1:
            closed_set = elements.copy()
            if not add_element_to_closed(closed_set, added_element, self.threshold):
                return set(), set(), False
        else:
            for x in elements:
                if i == 0:
                    closed_set.add(x)
                else:
                    if not add_element_to_closed(closed_set, x, self.threshold):
                        return domain.get_elements(), domain.get_elements()
                i += 1
        return closed_set, closed_set.difference(elements), True


class GraphClosureSP(Closure):
    def cl(self, elements: set, domain: Domain, forbidden_elements: set = {}, added_element: int = -1):
        def bfs_closure(graph, node, closure, new_closure_elements):
            lengths = nx.single_source_shortest_path_length(graph, node)
            current_elements = closure.copy()
            while len(np.nonzero(current_elements)[0]):
                for x in np.nonzero(current_elements)[0]:
                    current_elements[x] = 0
                    for n in graph[x]:
                        if closure[n] == 0 and lengths[n] < lengths[x]:
                            current_elements[n] = 1
                            new_closure_elements[n] = 1
                            closure[n] = 1
            return closure, new_closure_elements

        # calculates the closure of the nodes from node_list in the graph
        def graph_closure(graph, nodes):
            closure = np.zeros(graph.number_of_nodes(), dtype=np.bool)
            np.put(closure, nodes, np.ones(len(nodes), dtype=np.bool))
            new_closure_elements = closure.copy()
            while len(np.nonzero(new_closure_elements)[0]):
                left_nodes = np.nonzero(new_closure_elements)[0]
                for node in left_nodes:
                    new_closure_elements[node] = 0
                    closure, new_closure_elements = bfs_closure(graph, node, closure, new_closure_elements)
            return set(np.where(closure == True)[0])

        new_closure = graph_closure(domain.data_object, list(elements))
        return new_closure, new_closure.difference(elements)


class GraphClosureSP2(Closure):
    def cl(self, elements: set, data_object: object, forbidden_elements: set = {}, added_element: int = -1):
        pass
