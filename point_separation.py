from sklearn import svm
from Classifier import *
from Algos import *
from joblib import Parallel, delayed
from DataGenerator import DataGenerator
from DataToSQL import DataToSQL
from GetOpenMLData import OpenMLData


class PointSeparation(object):
    def __init__(self, database, dataset="Synthetic", positive_class=1, classifiers=["greedy", "opt"],
                 point_cloud_sizes=[1000],
                 training_sizes=[3, 4, 5, 10, 25, 50], run_number=100,
                 centers=[(0, 0), (7, 0)], plotting=False, job_num=16):
        self.classifiers = classifiers
        self.training_sizes_job_num = min(job_num, len(training_sizes))
        self.point_cloud_sizes_job_num = min(job_num // self.training_sizes_job_num, len(point_cloud_sizes))
        self.plotting = plotting
        self.database = database
        self.point_cloud_sizes = point_cloud_sizes
        self.centers = centers
        self.run_number = run_number
        self.training_sizes = training_sizes
        self.positive_class = positive_class

        self.dataset = dataset
        if self.dataset != "Synthetic":
            openMlData = OpenMLData(self.dataset, positive_class)
            self.data_X = openMlData.data_X
            self.data_y = openMlData.data_y

        if self.training_sizes == "rel":
            self.training_sizes = [len(self.data_X) // 20]

    def center_to_str(self):
        string = ""
        for x in self.centers[0]:
            string += str(x)
        for x in self.centers[1]:
            string += str(x)
        return string

    def run_experiment(self):
        if self.dataset == "Synthetic":
            for data_size in self.point_cloud_sizes:
                Parallel(n_jobs=self.training_sizes_job_num)(
                    delayed(self.run_single_experiment)(data_size, train_size) for train_size in self.training_sizes)
        else:
            Parallel(n_jobs=self.training_sizes_job_num)(
                delayed(self.run_single_experiment)(0, train_size) for train_size in self.training_sizes)

    def run_single_experiment(self, data_size, training_size):
        if training_size > len(self.centers[0]):
            counter = 0
            seed_counter = 0

            while counter < self.run_number * len(self.classifiers):
                # Generate synthetic blob data
                if self.dataset == "Synthetic":
                    X, y = DataGenerator.generate_blobs(n_samples=data_size, centers=[self.centers[0], self.centers[1]],
                                                        cluster_std=1, shuffle=False,
                                                        random_state=seed_counter * counter, separable=True)
                    dimension = len(self.centers[0])
                else:
                    X, y = self.data_X, self.data_y
                    dimension = len(X[0])
                seed_counter += 1

                num_red = training_size
                num_green = training_size

                red_points, green_points, training_points, training_labels, test_points, test_labels = set_training_testing(
                    X, y, num_red, num_green)

                separable = False
                for classifier in self.classifiers:
                    if classifier != "svm":
                        classification, separable = ConvexHullClassifier(classifier=classifier, plot=self.plotting,
                                                                         dimension=dimension,
                                                                         random=False, points=X,
                                                                         positive_points=red_points,
                                                                         negative_points=green_points, y=y)

                        if separable:
                            if counter % 1 == 0:
                                print("Size: {} Train Size: {} Example number: {} Dimension: {}".format(data_size,
                                                                                                        training_size,
                                                                                                        counter,
                                                                                                        dimension))
                            counter += 1


                    elif classifier == "svm" and (separable or len(self.classifiers) == 1):
                        counter += 1
                        clf = svm.SVC(kernel='linear', C=1000)
                        clf.fit(training_points, training_labels)
                        classification = clf.predict(X)
                        if self.plotting:
                            plot(X, color_list_testing(y, red_points, green_points), dim=dimension,
                                 name=classifier + str(len(X)) + str(len(red_points) + len(green_points)), model=clf)

                    columns = ['Timestamp', 'Accuracy', 'DefaultVal', 'Coverage', 'Correct', 'Wrong', 'Unclassified',
                               'ACorrect', 'AFalse', 'AUnclassified', 'BCorrect', 'BFalse',
                               'BUnclassified', 'Num', 'NumberA', 'NumberB', 'NumberBlue', 'NumTrainingA',
                               'NumTrainingB']
                    column_types = ["FLOAT" for x in columns]
                    print(classifier, len(classification))

                    if self.dataset == "Synthetic":
                        table_name = "ECML2020" + classifier + str(dimension) + "D" + "synthetic"
                    else:
                        table_name = "ECML2020" + classifier + str(self.dataset)

                    if separable or (classifier == "svm" and len(self.classifiers) == 1):
                        self.database.experiment_to_database(
                            table_name, columns,
                            column_types,
                            get_data_values(X, y, classification, red_points, green_points))

