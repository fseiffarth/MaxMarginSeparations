import os
import sqlite3
import time
from random import random

import matplotlib
import numpy as np
from math import sqrt

import tikzplotlib
from scipy.io import arff
from scipy.optimize import linprog
from scipy.spatial.qhull import ConvexHull
import matplotlib.pyplot as plt
import seaborn as sb


def dist(x, y):
    return sqrt(np.dot(x - y, x - y))


def intersect(hull_points_A, hull_points_B):
    for i in hull_points_A:
        if i in hull_points_B:
            return True
    return False


def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b, method="simplex")
    return lp.success


def get_inside_points(Points, X, Outside, add_point="", CheckSet=""):
    inside_points = X.copy()
    added_points = []
    if add_point != "":
        inside_points.append(add_point)
        added_points.append(add_point)

    inside_points_array = np.zeros(shape=(len(inside_points), len(Points[0])))
    counter = 0
    for i in inside_points:
        for j in range(0, len(Points[i])):
            inside_points_array[counter][j] = Points[i][j]
        counter += 1

    if CheckSet != "":
        for i in CheckSet:
            if in_hull(inside_points_array, Points[i]):
                return [], [], True

    for i in Outside:
        if in_hull(inside_points_array, Points[i]):
            inside_points.append(i)
            added_points.append(i)
    return inside_points, added_points, False


def get_points_inside_convex_hull(E, convex_hull, inside_p, outside_points, added_point=""):
    inside_points = inside_p.copy()
    added_points = []

    start = time.time()
    """
    for i in outside_points:
        point = E[i]
        inside = True
        for face in convex_hull.equations:
            c = np.dot(point, face[:-1]) + face[-1]  # point[0]*face[0] + point[1]*face[1] + face[2]
            # print(i, point, c)
            if 1e-14 > c > 0:  # 0.0000000000000005
                c = 0
            if c > 0:
                inside = False
                break
        if inside == True:
            inside_points.append(i)
            added_points.append(i)

    if added_point != "":
        inside_points.append(added_point)
        added_points.append(added_point)

    #print("Slow", time.time() - start)
    
    start = time.time()
    """
    outside = E[outside_points]
    outside_array = (
                np.matmul(outside, convex_hull.equations.transpose()[:-1]) + convex_hull.equations.transpose()[-1]).max(
        axis=1)
    t = np.argwhere(outside_array < 1e-14)
    t = t.reshape((len(t))).astype(int)
    added_points = list(np.array(outside_points)[t])
    #print("Fast", time.time() - start)

    return inside_points, added_points


def random_point_set(number, dimension, start_points1_number, start_points2_number, field_size=1):
    correct = False
    Point_List = []

    while correct == False:
        correct = True
        X_coord = field_size * np.random.rand(number // 2)
        Y_coord = field_size * np.random.rand(number // 2)
        if dimension == 3:
            Z_coord = field_size * np.random.rand(number // 2)

        X_coord = np.append(X_coord, 0 + field_size * np.random.rand(number // 2))
        Y_coord = np.append(Y_coord, 0 + field_size * np.random.rand(number // 2))
        if dimension == 3:
            Z_coord = np.append(Z_coord, 0 + field_size * np.random.rand(number // 2))

        Point_List = np.ndarray(shape=(number, dimension))

        if dimension == 3:
            for i in range(0, number):
                Point_List[i] = [X_coord[i], Y_coord[i], Z_coord[i]]
        else:
            for i in range(0, number):
                Point_List[i] = [X_coord[i], Y_coord[i]]

        # start point dense
        # set the start E 0 and number//2 and find neighbours
        start_pointsA = [0]
        start_pointsB = [number // 2]
        dist_points = 0.2
        while (len(start_pointsA) != start_points1_number) or (len(start_pointsB) != start_points2_number):
            dist_points += 0.1
            for i in range(number):
                if i not in start_pointsA and i not in start_pointsB:
                    if len(start_pointsA) < start_points1_number:
                        if dist(Point_List[i], Point_List[0]) < dist_points:
                            start_pointsA.append(i)
                    if len(start_pointsB) < start_points2_number:
                        if dist(Point_List[i], Point_List[number // 2]) < dist_points:
                            start_pointsB.append(i)

        # print(len(start_pointsA), len(start_pointsB))

        # initialize first half space
        HullPoints1 = start_pointsA
        PointsH_1 = np.ndarray(shape=(len(HullPoints1), dimension))
        counter = 0
        outside_points = [x for x in range(number) if x not in HullPoints1]
        for i in HullPoints1:
            PointsH_1[counter] = Point_List[i]
            counter += 1

        ConvexHull1 = ConvexHull(PointsH_1, 1)
        insideH_1, added = get_points_inside_convex_hull(Point_List, ConvexHull1, HullPoints1, outside_points)
        HullPoints1 = insideH_1

        # Initialize second half space
        HullPoints2 = start_pointsB
        PointsH_2 = np.ndarray(shape=(len(HullPoints2), dimension))
        counter = 0
        outside_points = [x for x in range(number) if x not in HullPoints2]
        for i in HullPoints2:
            PointsH_2[counter] = Point_List[i]
            counter += 1

        ConvexHull2 = ConvexHull(PointsH_2, 1)

        insideH_2, added = get_points_inside_convex_hull(Point_List, ConvexHull2, HullPoints2, outside_points)
        HullPoints2 = insideH_2

        # Only allow those cases where there is no intersection of the convex hull at the beginning
        if intersect(HullPoints1, HullPoints2) == True:
            correct = False
            continue

    return (Point_List, HullPoints1, HullPoints2)


def generate_start_points(class_label, labels, number):
    pos_labels = [x for x in range(len(labels)) if labels[x] == class_label]
    return random.sample(pos_labels, number)


def time_step(string_name, time_point, level=1):
    time_dur = time.time() - time_point
    time_point = time.time()
    tabs = ""
    for i in range(level):
        tabs += "\t"
    # print(tabs, string_name, time_dur, "s")
    return time_point


def plot_svm(ax, model):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
    # plot support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')


def plot(data, color_list, dim=3, name="Test", model="", save=False):
    if dim == 3:
        X_coord = []
        Y_coord = []
        Z_coord = []

        ax = plt.axes(projection='3d')

        for i in range(len(data)):
            X_coord.append(data[i][0])
            Y_coord.append(data[i][1])
            Z_coord.append(data[i][2])

        ax.scatter3D(X_coord, Y_coord, Z_coord, c=color_list)
    else:
        ax = plt.axes()

        x_val_dict = {}
        y_val_dict = {}

        for i, x in enumerate(color_list, 0):
            if x not in x_val_dict:
                x_val_dict[x] = [data[i][0]]
                y_val_dict[x] = [data[i][1]]
            else:
                x_val_dict[x].append(data[i][0])
                y_val_dict[x].append(data[i][1])

        for key, value in x_val_dict.items():
            for i in range(len(value)):
                circle = plt.Circle((value[i], y_val_dict[key][i]), 0.05, color="green", fill=False)
                ax.add_artist(circle)
        for key, value in x_val_dict.items():
            plt.scatter(value, y_val_dict[key], c=key)

        if model:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # create grid to evaluate model
            xx = np.linspace(xlim[0], xlim[1], 30)
            yy = np.linspace(ylim[0], ylim[1], 30)
            YY, XX = np.meshgrid(yy, xx)
            xy = np.vstack([XX.ravel(), YY.ravel()]).T
            Z = model.decision_function(xy).reshape(XX.shape)

            # plot decision boundary and margins
            plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=1,
                        linestyles=['--', '-', '--'])
            # plot support vectors
            plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                        linewidth=1, facecolors='none', edgecolors='k')

        if save:
            tikzplotlib.save(name + ".tex")
        plt.show()

    # tikz_save("/home/florian/Dokumente/Forschung/EigeneForschung/SpringerLatex/" + "PlotOneClass" + ".tex", wrap = False)


def plot_prediction(openMLData, prediction, color_list, dim=3, name="Test", model="", save=False, algo=""):
    if dim == 3:
        X_coord = []
        Y_coord = []
        Z_coord = []

        ax = plt.axes(projection='3d')

        for i in range(openMLData.class_number):
            X_coord.append(openMLData.data_X[i][0])
            Y_coord.append(openMLData.data_X[i][1])
            Z_coord.append(openMLData.data_X[i][2])

        ax.scatter3D(X_coord, Y_coord, Z_coord, c=color_list)
    else:
        ax = plt.axes()

        x_val_dict = {}
        y_val_dict = {}

        color = ""
        for i, x in enumerate(openMLData.data_X):
            if prediction[i] == -1:
                color = "blue"
            elif openMLData.data_y[i] == prediction[i]:
                color = "green"
            else:
                color = "red"

            circle = plt.Circle((x[0], x[1]), 0.05, color=color, fill=False)
            ax.add_artist(circle)

        for i, x in enumerate(color_list, 0):
            if x not in x_val_dict:
                x_val_dict[x] = [openMLData.data_X[i][0]]
                y_val_dict[x] = [openMLData.data_X[i][1]]
            else:
                x_val_dict[x].append(openMLData.data_X[i][0])
                y_val_dict[x].append(openMLData.data_X[i][1])
        for key, value in x_val_dict.items():
            plt.scatter(value, y_val_dict[key], c=key)

        if model:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # create grid to evaluate model
            xx = np.linspace(xlim[0], xlim[1], 30)
            yy = np.linspace(ylim[0], ylim[1], 30)
            YY, XX = np.meshgrid(yy, xx)
            xy = np.vstack([XX.ravel(), YY.ravel()]).T
            Z = model.decision_function(xy).reshape(XX.shape)

            # plot decision boundary and margins
            plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=1,
                        linestyles=['--', '-', '--'])
            # plot support vectors
            plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                        linewidth=1, facecolors='none', edgecolors='k')

        if save:
            tikzplotlib.save(name + ".tex")
        plt.title(algo)
        plt.show()

    # tikz_save("/home/florian/Dokumente/Forschung/EigeneForschung/SpringerLatex/" + "PlotOneClass" + ".tex", wrap = False)


def color_list(Label_List):
    color_list = []
    for x in Label_List:
        if x == 0:
            color_list.append("green")
        elif x == 1:
            color_list.append("red")

    return color_list


def color_list_testing(labels, train_a, train_b):
    colorlist = np.where(labels == 0, "orange", "violet")
    colorlist[train_a] = ["blue"]
    colorlist[train_b] = ["red"]
    return colorlist


def color_list_result(Node_List, Label_List, pos_points, neg_points):
    color_list = []
    for x in range(len(Node_List)):
        if x in pos_points:
            color_list.append("red")
        elif x in neg_points:
            color_list.append("green")
        else:
            if Label_List[x] == 0:
                color_list.append("violet")
            elif Label_List[x] == 1:
                color_list.append("orange")
            elif Label_List[x] == -1:
                color_list.append("blue")

    return color_list


def color_list_training(Node_List, pos_points, neg_points):
    color_list = []
    for x in range(len(Node_List)):
        if x in pos_points:
            color_list.append("red")
        elif x in neg_points:
            color_list.append("green")
        else:
            color_list.append("blue")
    return color_list


def print_error_evaluation(name, X, y, classification, pos_points, neg_points):
    num_training = (len(pos_points) + len(neg_points))
    num = len(X)
    num_test = (num - num_training)
    red = 0
    green = 0

    # Fehlerauswertung
    error = 0
    error_red = 0
    error_green = 0
    unclassified_red = 0
    unclassified_green = 0
    correct_red = 0
    correct_green = 0

    counter = 0
    for x in classification.flatten():
        label = y.flatten()[counter]

        if label == 0:
            green += 1
        else:
            red += 1

        # errors
        if x == -1:
            if label == 1:
                unclassified_red += 1
            else:
                unclassified_green += 1
        elif x == 1:
            if label == 1:
                correct_red += 1
            else:
                error_red += 1
        elif x == 0:
            if label == 0:
                correct_green += 1
            else:
                error_green += 1
        counter += 1

    correct_green -= len(neg_points)
    correct_red -= len(pos_points)

    error = error_green + error_red
    correct = correct_green + correct_red
    accuracy = correct / (correct + error)
    unclassified = unclassified_green + unclassified_red
    recall = (correct + error) / num_test
    print(name)
    print("Accuracy", correct / (error + correct), "Correct:", correct, "Error: ", error, "Unclassified: ",
          unclassified, "Training: ", len(pos_points) + len(neg_points))


"""If saving a classification result to a sqlite check if db exists otherwise create"""


def create_table(db_name, table_name, column_list):
    if not os.path.isfile(db_name):
        con = sqlite3.connect(db_name)
        cur = con.cursor()

        string = ""
        for entry in column_list:
            string += entry
            string += ","
        string = string[:-1]
        cur.execute("CREATE TABLE " + table_name + " (" + string + ");")


"""Saving classification result to database"""


def get_data_values(X, y, classification, pos_points, neg_points):
    num_training = (len(pos_points) + len(neg_points))
    num = len(X)
    num_test = (num - num_training)
    red = 0
    green = 0

    # Fehlerauswertung
    error = 0
    error_red = 0
    error_green = 0
    unclassified_red = 0
    unclassified_green = 0
    correct_red = 0
    correct_green = 0

    counter = 0
    for x in classification.flatten():
        label = y.flatten()[counter]

        if label == 0:
            green += 1
        else:
            red += 1

        # errors
        if x == -1:
            if label == 1:
                unclassified_red += 1
            else:
                unclassified_green += 1
        elif x == 1:
            if label == 1:
                correct_red += 1
            else:
                error_red += 1
        elif x == 0:
            if label == 0:
                correct_green += 1
            else:
                error_green += 1
        counter += 1

    correct_green -= len(neg_points)
    correct_red -= len(pos_points)

    error = error_green + error_red
    correct = correct_green + correct_red
    accuracy = correct / (correct + error)
    unclassified = unclassified_green + unclassified_red
    recall = (correct + error) / num_test

    # ['Timestamp','Accuracy','DefaultVal','Recall','Correct','Wrong','Unclassified','RedCorrect','RedFalse','RedUnclassified','GreenCorrect','GreenFalse','GreenUnclassified','Num','NumberRed','NumberGreen','NumberBlue','NumTrainingRed','NumTrainingGreen']

    values = [[time.time(), round(float(accuracy), 2), max(red, green) / num, round(float(recall), 2), int(correct),
               int(error), int(unclassified), int(correct_red), int(error_red), int(unclassified_red),
               int(correct_green), int(error_green), int(unclassified_green), int(num), int(red), int(green),
               int(unclassified), int(len(pos_points)), int(len(neg_points))]]
    return values


"""Saving classification result to database"""


def add_row_to_database(databasename, table_name, column_list, X, y, classification, pos_points, neg_points):
    con = sqlite3.connect(databasename)
    cur = con.cursor()

    string = ""
    for entry in column_list:
        string += entry
        string += ","

    num_training = (len(pos_points) + len(neg_points))
    num = len(X)
    num_test = (num - num_training)
    red = 0
    green = 0

    # Fehlerauswertung
    error = 0
    error_red = 0
    error_green = 0
    unclassified_red = 0
    unclassified_green = 0
    correct_red = 0
    correct_green = 0

    counter = 0
    for x in classification.flatten():
        label = y.flatten()[counter]

        if label == 0:
            green += 1
        else:
            red += 1

        # errors
        if x == -1:
            if label == 1:
                unclassified_red += 1
            else:
                unclassified_green += 1
        elif x == 1:
            if label == 1:
                correct_red += 1
            else:
                error_red += 1
        elif x == 0:
            if label == 0:
                correct_green += 1
            else:
                error_green += 1
        counter += 1

    correct_green -= len(neg_points)
    correct_red -= len(pos_points)

    error = error_green + error_red
    correct = correct_green + correct_red
    accuracy = correct / (correct + error)
    unclassified = unclassified_green + unclassified_red
    recall = (correct + error) / num_test

    # ['Timestamp','Accuracy','DefaultVal','Recall','Correct','Wrong','Unclassified','RedCorrect','RedFalse','RedUnclassified','GreenCorrect','GreenFalse','GreenUnclassified','Num','NumberRed','NumberGreen','NumberBlue','NumTrainingRed','NumTrainingGreen']

    to_db = [[time.time(), round(float(accuracy), 2), max(red, green) / num, round(float(recall), 2), int(correct),
              int(error), int(unclassified), int(correct_red), int(error_red), int(unclassified_red),
              int(correct_green), int(error_green), int(unclassified_green), int(num), int(red), int(green),
              int(unclassified), int(len(pos_points)), int(len(neg_points))]]

    cur.executemany(
        "INSERT INTO " + table_name + " (" + string[:-1] + ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);", to_db)
    con.commit()
    con.close()


def set_training_testing(E, E_labels, A_size, B_size, seed=0):
    # Generate samples
    A_elements = []
    B_elements = []

    A_B_vectors = np.zeros(shape=(A_size + B_size, E.shape[1]))
    test_points = np.zeros(shape=(len(E) - A_size + B_size, E.shape[1]))
    A_B_labels = np.zeros(shape=(A_size + B_size, 1))
    test_labels = np.zeros(shape=(len(E) - A_size + B_size, 1))

    counter = 0
    random_training = np.zeros(shape=(len(E), 1))
    while counter != len(E) or (len(A_elements) < A_size and len(B_elements) < B_size):
        # random.seed(seed)
        elem = random.randint(0, len(E) - 1)
        if random_training[elem] == 0:
            if E_labels[elem] == 1 and len(A_elements) < A_size:
                A_elements.append(elem)
            elif E_labels[elem] == 0 and len(B_elements) < B_size:
                B_elements.append(elem)
            random_training[elem] = 1
            counter += 1

    # E training and test sets
    counter = 0
    counter1 = 0
    counter2 = 0
    for x in E:
        if counter in A_elements:
            A_B_vectors[counter1] = E[counter]
            A_B_labels[counter1] = 1
            counter1 += 1
        elif counter in B_elements:
            A_B_vectors[counter1] = E[counter]
            A_B_labels[counter1] = 0
            counter1 += 1
        else:
            test_points[counter2] = E[counter]
            test_labels[counter2] = E_labels[counter]
            counter2 += 1
        counter += 1

    return A_elements, B_elements, A_B_vectors, A_B_labels, test_points, test_labels


def load_data(file_path, max_number, n_features=3, max_labels=1):
    number_of_points = 5000
    n_features = 5
    n_labels = 1

    data, meta = arff.loadarff("/home/florian/scikit_learn_data/mozilla4.arff")
    print(data)
    X = np.zeros(shape=(min(len(range(number_of_points)), len(data)), n_features))
    y = np.zeros(shape=(min(len(range(number_of_points)), len(data)), n_labels))
    counter = 0
    label_name = ""
    for row in data:
        if counter < number_of_points:
            for i in range(n_features):
                X[counter][i] = row[i]
            if counter == 0:
                label_name = row[n_features]
            if row[n_features] == label_name:
                y[counter][0] = 0
            else:
                y[counter][0] = 1
        counter += 1
    # E, E_labels = _convert_arff_data(data, 3, 1)
    # E, E_labels = load_breast_cancer(True)


def generate_heatmap(openMLData, prediction, algo=''):
    data = np.zeros(shape=(openMLData.class_number, openMLData.class_number))
    prediction_dist = {}
    for x in prediction:
        if x in prediction_dist.keys():
            prediction_dist[x] += 1
        else:
            prediction_dist[x] = 1
    for i in range(openMLData.data_size):
        data[int(prediction[i])][openMLData.data_y[i]] += 1. / prediction_dist[openMLData.data_y[i]]
    sb.heatmap(data, cmap="Blues")
    plt.title(algo)
    plt.show()


def prediction_colormap(openMLData, prediction):
    color_list = []
    cmap = matplotlib.cm.get_cmap('viridis')
    for x in prediction:
        color_list.append(cmap(float(x) / (openMLData.class_number - 1)))
    return color_list
