import random

from sklearn import datasets
import numpy as np

from DataGenerator import DataGenerator


class OpenMLData:
    def __init__(self, db_id, positive_class=1, multilabel=False, synthetic=False):
        self.db_id = db_id
        #check for synthetic data ids=0, 1, 2
        if synthetic:
            centers = [[(0, 0), (4, 0)], [(0, 0, 0), (4, 0, 0)], [(0, 0, 0, 0), (4, 0, 0, 0)]]
            self.data_X, self.data_y = DataGenerator.generate_blobs(n_samples=1000, centers=[centers[self.db_id-2][0], centers[self.db_id-2][1]],
                                                cluster_std=1, shuffle=False,
                                                random_state=int(1/random.random()), separable=True)

        else:
            self.data_X, self.data_y = datasets.fetch_openml(data_id=db_id, return_X_y=True)

        self.data_X.astype(dtype=np.longdouble)
        self.class_number = 0
        self.label_distribution = {}
        self.data_size = len(self.data_X)
        self.training_size = 0

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
        self.training_size = training_size
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
        self.training_size = self.class_number * training_size
        return training_sets

    def make_svm_data(self, training_sets):
        return [self.data_X[i] for y in training_sets for i in y], [self.data_y[i] for y in training_sets for i in y]

    def get_train_sizes(self):
        train_sizes = []
        maximum = np.min([y for x, y in self.label_distribution.items()])//10
        for i in range(5):
            train_sizes.append(1 + (maximum-1) // 4 * i)
        return train_sizes

    def get_majority_size(self):
        pass




def main():
    data = OpenMLData(1460)
    print(data.data)


if __name__ == '__main__':
    main()
