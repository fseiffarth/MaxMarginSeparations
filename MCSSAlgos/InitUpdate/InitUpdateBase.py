from abc import ABC, abstractmethod

from ordered_set import OrderedSet

from MCSSAlgos.Domains.DomainBase import Domain


class PartialOrder(ABC):
    def __init__(self):
        self.order = {int: set}
        self.argmin_val = 0

    @abstractmethod
    def pop_minimum_element(self):
        pass


class InitUpdate(ABC):
    def __init__(self, partial_order: PartialOrder):
        self.v = {int: OrderedSet}
        self.w = {}
        self.partial_order = partial_order

    @abstractmethod
    def init(self, domain: Domain, initial_sets: list, unlabeled_elements: set):
        pass

    @abstractmethod
    def update(self, domain: Domain, extended_object: int, added_objects: set, invalid_extensions: set) -> set:
        pass

    def can_pop_element(self):
        return self.partial_order.argmin_val in self.partial_order.order and len(self.partial_order.order[self.partial_order.argmin_val]) > 0
