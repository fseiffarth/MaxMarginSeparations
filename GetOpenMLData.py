import random

from sklearn import datasets
import numpy as np


class OpenMLData:
    def __init__(self, db_id, positive_class=1, multilabel=False):
        self.db_id = db_id
        self.data_X, self.data_y = datasets.fetch_openml(data_id=db_id, return_X_y=True)
        self.class_number = 0
        self.label_distribution = {}
        self.data_size = len(self.data_X)

        if multilabel == True:
            label_list = []
            for i in range(len(self.data_y)):
                if self.data_y[i] in label_list:
                    new_class_label = label_list.index(self.data_y[i])
                    self.data_y[i] = new_class_label
                else:
                    label_list.append(self.data_y[i])
                    self.data_y[i] = len(label_list) - 1
            self.class_number = len(label_list)
            for i in range(self.class_number):
                self.label_distribution[i] = 0
            for i in self.data_y:
                self.label_distribution[i] += 1
            self.data_y = self.data_y.astype(np.int)
        else:
            self.data_y = np.where(self.data_y == positive_class, 1, 0)
            self.data_y = self.data_y.astype(np.int)
            self.class_number = 2

    def get_random_training_data(self, training_size):
        training_sets = []
        for _ in range(self.class_number):
            training_sets.append([])
        random_choice = random.sample(range(len(self.data_X)), training_size)
        for x in random_choice:
            training_sets[self.data_y[x]].append(x)

        return training_sets

    def get_random_training_data_by_class(self, training_size):
        classes = []
        training_sets = []
        for _ in range(self.class_number):
            classes.append([])
            training_sets.append([])
        for i, l in enumerate(self.data_y, 0):
            classes[l].append(i)

        for i in range(self.class_number):
            if len(classes[i]) >= training_size:
                training_sets[i] = random.sample(classes[i], training_size)
            else:
                raise ValueError("Training size is greater than some class size")
        return training_sets

    def make_svm_data(self, training_sets):
        return [self.data_X[i] for y in training_sets for i in y], [self.data_y[i] for y in training_sets for i in y]


def main():
    data = OpenMLData(1460)
    print(data.data)


if __name__ == '__main__':
    main()
