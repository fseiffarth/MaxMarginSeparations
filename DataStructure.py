"""Main class for all the datastructures"""
from abc import abstractmethod


class DataStructure:
    def __init__(self, data):
        self.data = data
        self.size = None
        self.closure_operator = None
        self.linkage_function = None

        self.labels = None
        self.class_num = None
        self.training_samples = None
        self.closed_training_samples = None

    @abstractmethod
    def set_training_data(self):
        pass

    @abstractmethod
    def label_data(self):
        pass

    @abstractmethod
    def linkage_pre_computation(self):
        pass

    @abstractmethod
    def plot_data(self):
        pass






