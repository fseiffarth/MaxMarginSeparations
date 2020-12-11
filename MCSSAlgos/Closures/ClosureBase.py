from abc import ABC, abstractmethod


class Closure(ABC):
    @abstractmethod
    def cl(self, elements: set, data_object: object, forbidden_elements: set = {}, added_element:int = -1):
        pass
