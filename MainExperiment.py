import itertools
import time
from sklearn import datasets, svm

from GetOpenMLData import OpenMLData
from MultiClassSeparation import MultiClassSeparation
from Algos import generate_heatmap, plot, prediction_colormap, plot_prediction, intersect
import numpy as np
from joblib import Parallel, delayed


class Evaluation:
    def __init__(self):
        self.runtime = []
        self.accuracy = []
        self.w_accuracy = []
        self.coverage = []
        self.w_coverage = []
        self.training_size = 0
        self.algo = []

    def set_values(self, openMLData, algo, training, prediction, runtime):
        prediction_dist = {}
        for x in prediction:
            if x in prediction_dist.keys():
                prediction_dist[x] += 1
            else:
                prediction_dist[x] = 1

        acc = 0
        covered = 0
        acc_w = 0
        cover_w = 0
        for key, value in prediction_dist.items():
            key = int(key)
            if key != -1:
                c_i = np.where(np.asarray(prediction) == key)
                c_i_hat = np.setdiff1d(c_i, training[key])
                covered += len(c_i_hat)
                label_indices = np.where(openMLData.data_y == key)
                correct = np.intersect1d(c_i_hat, label_indices)
                acc += len(correct)
                if len(c_i_hat > 0):
                    acc_w += len(correct) / len(c_i_hat)
                else:
                    acc_w += 0
                cover_w += len(correct) / (openMLData.label_distribution[key] - len(training[key]))
        self.training_size = sum(len(row) for row in training)
        if covered == 0:
            self.accuracy.append(0)
        else:
            self.accuracy.append(acc / covered)
        self.w_accuracy.append(acc_w / openMLData.class_number)
        self.coverage.append(covered / (openMLData.data_size - self.training_size))
        self.w_coverage.append(cover_w / openMLData.class_number)
        self.runtime.append(runtime)
        self.algo.append(algo)

    def evaluation_to_database(self, database, data_id):
        columns = ['Timestamp', 'Algo', 'Accuracy', 'AccuracyW', 'Coverage', 'CoverageW', 'Runtime',
                   'TrainingSize']
        column_types = ["FLOAT" for x in columns]
        column_types[1] = "TEXT"
        table_name = "Data_" + str(data_id)
        for i in range(len(self.accuracy)):
            database.experiment_to_database(table_name, columns, [
                [time.time(), self.algo[i], self.accuracy[i], self.w_accuracy[i], self.coverage[i], self.w_coverage[i],
                 self.runtime[i], self.training_size]], column_types)


def training_sets_disjoint(openmlData, mcs):
    for pair in itertools.combinations(range(openmlData.class_number), r=2):
        if intersect(mcs.S[pair[0]], mcs.S[pair[1]]):
            return False
    return True


def run_single_experiment(database, openmlData, data_id, algos, train_size, number, print_time=False, plot=False, linprog_calc=None,
                          synthetic_data=False):
    start_time = time.time()
    eval = Evaluation()
    num = 0
    intersect_count = 0
    while num < number:
        num += 1
        training_sets = openmlData.get_random_training_data_by_class(train_size)
        # check if training sets are disjoint
        mcs = MultiClassSeparation(training_sets=training_sets, openMLData=openmlData, print_runtimes=print_time, linprog_calc=linprog_calc)
        if not training_sets_disjoint(openmlData, mcs):
            num -= 1
            intersect_count += 1
            print("Training Sets Intersect:", train_size, "Count:", intersect_count, "Number:", num)
            continue
        else:
            for algo in algos:
                if algo == "svm":
                    svm_data = openmlData.make_svm_data(training_sets)
                    start = time.time()
                    clf = svm.SVC(kernel='linear', C=1000, break_ties=True)
                    clf.fit(svm_data[0], svm_data[1])
                    prediction = clf.predict(openmlData.data_X)
                    new_time = time.time()
                    if plot:
                        generate_heatmap(openmlData, prediction, "SVM")
                        plot_prediction(openmlData, prediction, dim=2, algo="SVM")
                    if print_time:
                        print("SVM {}".format(new_time - start))
                    eval.set_values(openmlData, algo, training_sets, prediction, new_time - start)
                else:
                    if algo == "greedy":
                        mcs.is_greedy = True
                        start = time.time()
                        _, prediction, _ = mcs.generalized_algorithm()
                        new_time = time.time()
                        if plot:
                            generate_heatmap(openmlData, prediction, "Greedy")
                            plot_prediction(openmlData, prediction, dim=2, algo="Greedy")
                        if print_time:
                            print("Greedy {}".format(new_time - start))
                        eval.set_values(openmlData, algo, training_sets, prediction, new_time - start)
                    else:
                        mcs.is_greedy = False
                        if algo == "maxmargin":
                            start = time.time()
                            _, prediction, _ = mcs.generalized_algorithm()
                            new_time = time.time()
                            if plot:
                                generate_heatmap(openmlData, prediction, "Generalized Algo")
                                plot_prediction(openmlData, prediction, dim=2, algo="Generalized Algo")
                            if print_time:
                                print("Generalized {}".format(new_time - start))
                            eval.set_values(openmlData, algo, training_sets, prediction, new_time - start)
                        if algo == "OvA":
                            start = time.time()
                            _, prediction, _ = mcs.one_vs_all()
                            new_time = time.time()
                            if plot:
                                generate_heatmap(openmlData, prediction, "One vs all")
                                plot_prediction(openmlData, prediction, dim=2, algo="One vs all")
                            if print_time:
                                print("One vs. all {}".format(new_time - start))
                            eval.set_values(openmlData, algo, training_sets, prediction, new_time - start)
                        if algo == "OvOlinkage":
                            start = time.time()
                            _, prediction, _ = mcs.one_vs_one(confidence_measure="linkage")
                            new_time = time.time()
                            if plot:
                                generate_heatmap(openmlData, prediction, "One vs one")
                                plot_prediction(openmlData, prediction, dim=2, algo="One vs one")
                            if print_time:
                                print("One vs. one {}".format(new_time - start))
                            eval.set_values(openmlData, algo, training_sets, prediction, new_time - start)
                        if algo == "OvOstd":
                            start = time.time()
                            _, prediction, _ = mcs.one_vs_one()
                            new_time = time.time()
                            if plot:
                                generate_heatmap(openmlData, prediction, "One vs one")
                                plot_prediction(openmlData, prediction, dim=2, algo="One vs one")
                            if print_time:
                                print("One vs. one {}".format(new_time - start))
                            eval.set_values(openmlData, algo, training_sets, prediction, new_time - start)
        if not synthetic_data:
            print("Num: {}/{}".format(num, number) + " Estimated Duration: " + str(
                (number - num) * (time.time() - start_time) / num) + "s")
    eval.evaluation_to_database(database, data_id)


def run_synthetic(database, algos, dim=2, number=1000, train_sizes=None, print_time=False, plot=False):
    run_start = time.time()
    print(dim, number)
    for i in range(number):
        openmlData = OpenMLData(db_id=dim, multilabel=True, synthetic=True)
        if openmlData.class_number == 2:
            algos = algos[0:3]
        if train_sizes is None:
            train_sizes = openmlData.get_train_sizes()
        Parallel(n_jobs=-1)(
            delayed(run_single_experiment)(database, openmlData, dim, algos, x, 1, print_time, plot, True) for x in
            train_sizes)
        print("Num: {}/{}".format(i+1, number) + " Estimated Duration: " + str(
            (number - (i+1)) * (time.time() - run_start) / (i+1)) + "s")
    print("Run Duration: ", time.time() - run_start, " s")


def run(database, algos, data_id, number, train_sizes=None, print_time=False, plot=False, linprog_calc=False):
    run_start = time.time()
    print(data_id, number)
    openmlData = OpenMLData(db_id=data_id, multilabel=True)
    if openmlData.class_number == 2:
        algos = algos[0:3]
    if train_sizes is None:
        train_sizes = openmlData.get_train_sizes()

    Parallel(n_jobs=-1)(
        delayed(run_single_experiment)(database, openmlData, data_id, algos, x, number, print_time, plot, linprog_calc) for x in
        train_sizes)
    print("Run Duration: ", time.time() - run_start, " s")
