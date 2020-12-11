from abc import ABC, abstractmethod


class Domain(ABC):
    def __init__(self, data_object: object, size: int):
        self.data_object = data_object
        self.size = size

    @abstractmethod
    def get_elements(self) -> set:
        pass
