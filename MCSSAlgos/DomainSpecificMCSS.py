from abc import ABC
from itertools import combinations

from MCSSAlgos.Closures.ClosureBase import Closure
from MCSSAlgos.Domains.DomainBase import Domain
from MCSSAlgos.InitUpdate.InitUpdateBase import InitUpdate


class DomainSpecificMCSS(ABC):
    def __init__(self, domain: Domain, closure: Closure, init_update: InitUpdate):
        self.domain = domain
        self.closure = closure
        self.init_update = init_update
        self.closure_sets = []
        self.query_num = 0
        self.considered_element_order = []
        self.extended_element_order = []

    def run(self, input_sets):
        self.query_num = 0
        for input_set in input_sets:
            self.closure_sets.append(self.closure.cl(input_set, self.domain)[0])
            self.query_num += 1
        pairs = combinations(range(len(input_sets)), 2)
        for (i, j) in pairs:
            if len(self.closure_sets[i].intersection(self.closure_sets[j])) != 0:
                return "NO"
        else:
            F = self.domain.get_elements().difference(set.union(*self.closure_sets))
            self.init_update.init(self.domain, self.closure_sets, F)
            while self.init_update.can_pop_element():
                e = self.init_update.partial_order.pop_minimum_element()
                self.considered_element_order.append(e)
                try:
                    tag = self.init_update.v[e]
                    invalid_extensions = set()
                    added = set()
                    for x in tag.items:
                        new_closure, added, extension_is_valid = self.closure.cl(self.closure_sets[x], self.domain, set.union(
                            *[l for i, l in enumerate(self.closure_sets) if i != x]), e)
                        self.query_num += 1
                        if extension_is_valid:
                            for y in range(len(input_sets)):
                                if x != y:
                                    if len(new_closure.intersection(self.closure_sets[y])) != 0:
                                        extension_is_valid = False
                                        invalid_extensions.add(x)
                                        new_closure = []
                                        added = []
                        else:
                            invalid_extensions.add(x)
                        if extension_is_valid:
                            self.closure_sets[x] = new_closure.copy()
                            self.extended_element_order.append(e)
                            break
                    self.init_update.update(domain=self.domain, extended_object=e, added_objects=added, invalid_extensions=invalid_extensions)
                except:
                    pass

        return self.closure_sets, self.query_num, self.extended_element_order, self.considered_element_order
