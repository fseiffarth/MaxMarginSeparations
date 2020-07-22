from DataToSQL.DataToSQL import DataToSQL
import numpy as np
import matplotlib.pyplot as plt


def evaluation_points(src_path, db_name):
    dataset = DataToSQL.DataToSQL(file_path=src_path,
                                  db_name=db_name)

    sizes = [1000]
    train_sizes = [6, 8, 10, 20, 50, 100]

    # data to plot
    n_groups = 6
    n_bars = 3

    # create plot
    fig, ax = plt.subplots()
    bar_width = 0.1
    opacity = 0.8
    labels = []

    for dimension in [2, 3, 4]:
        for v in ["Accuracy", "Coverage"]:
            color = ["red", "blue", "green"]
            for i, alg in enumerate(["greedy", "opt", "svm"]):
                height = []
                out_string = "\\addplot[fill=" + color[i] + ", draw = " + color[i] + "] coordinates {"
                for j, x in enumerate(train_sizes, 0):
                    table_name = 'ECML2020' + alg + str(dimension) + "D" + "synthetic"
                    val = dataset.get_from_query(
                        'SELECT Avg(' + v + ') FROM ' + table_name + ' WHERE Num=1000' + ' AND NumTrainingA+NumTrainingB =' + str(
                            x))[0][0]
                    if (val != None):
                        labels.append(str(x))
                        height.append(val)
                        out_string += "(" + str(x) + "," + str(val) + ") " + "[0] "
                        index = np.arange(len(height))

                out_string += "};\n"
                print(out_string)
                rects1 = plt.bar(index + bar_width * i, height, bar_width,
                                 alpha=opacity,
                                 color=color[i])

            plt.xlabel('Training Size')
            plt.ylabel(v)
            plt.title("Dimension " + str(dimension))
            plt.xticks(index + (bar_width / 2) * (n_bars - 1), labels)
            plt.legend()

            plt.tight_layout()
            plt.show()


def evaluation_graphs(src_path, db_name):
    dataset = DataToSQL.DataToSQL(file_path=src_path,
                                  db_name=db_name)

    graph_size = 100
    train_sizes = [2, 4]
    edge_densities = [1, 1.1, 1.2, 1.3, 1.4, 1.5]
    marks = ["bo-", "rs-", "y-"]
    for y_label in ["Accuracy", "Coverage"]:
        for i, size in enumerate(train_sizes, 0):
            out_string = ""
            x_values = []
            y_values = []
            for j, x in enumerate(edge_densities, 0):
                query = 'SELECT Avg(' + y_label + ') FROM ECML2020 WHERE GraphSize=' + str(
                    graph_size) + ' AND RedSize+GreenSize =' + str(size) + ' AND EdgeDensity=' + str(x)
                val = dataset.get_from_query(query)[0][0]
                if (val is not None):
                    out_string += str(x) + " " + str(val) + "\\\\"
                x_values.append(x)
                y_values.append(val)
            plt.plot(x_values, y_values, marks[i])
                # print("Size: {} TrainSize: {} Density {}".format(str(graph_size), str(size), str(x)))
            print(out_string)

        x_values = []
        y_values = []
        if y_label == "Accuracy":
            for j, x in enumerate(edge_densities, 0):
                query = 'SELECT Avg(MAX(TargetRedSize, TargetGreenSize)/100) FROM ECML2020 WHERE GraphSize=' + str(
                        graph_size) + ' AND EdgeDensity=' + str(x)
                val = dataset.get_from_query(query)[0][0]
                x_values.append(x)
                y_values.append(val)
            plt.plot(x_values, y_values, marks[2])
            plt.legend(["train size 2", "train size 4", "baseline"])
        else:
            plt.legend(["train size 2", "train size 4"])

        plt.xlabel('Edge Density')
        plt.ylabel(y_label)
        plt.title("graph Size " + str(graph_size))

        plt.show()


def evaluation_trees(src_path, db_name):
    dataset = DataToSQL.DataToSQL(file_path=src_path,
                                  db_name=db_name)
    train_sizes = [2, 4, 6, 8, 10]
    marks = ["bo-", "rs-", "y-"]
    for tree_size in [1000, 20000]:
        for a, alg in enumerate(["greedy", "opt"], 0):
            x_values = []
            y_values = []
            for i, x in enumerate(train_sizes, 0):
                out_string = ""
                query = 'SELECT Avg(Accuracy) FROM ECML2020' + alg +' WHERE NumNodes=' + str(tree_size) + ' AND NumTrainingRed+NumTrainingGreen =' + str(x)
                val = dataset.get_from_query(query)[0][0]
                if (val is not None):
                    out_string += str(x) + " " + str(val) + "\\\\"
                x_values.append(x)
                y_values.append(val)
            plt.plot(x_values, y_values, marks[a])
            print(out_string)

            x_values = []
            y_values = []


        for j, x in enumerate(train_sizes, 0):
            query = 'SELECT Avg(MAX(NumberRed, NumberGreen)/NumNodes) FROM ECML2020greedy WHERE NumNodes=' + str(
                    tree_size) + ' AND NumTrainingRed+NumTrainingGreen =' + str(x)
            val = dataset.get_from_query(query)[0][0]
            x_values.append(x)
            y_values.append(val)
        plt.plot(x_values, y_values, marks[2])
        plt.legend(["greedy", "max margin", "baseline"])


        plt.xlabel('Training Size')
        plt.ylabel("Accuracy")
        plt.title("graph Size " + str(tree_size))
        plt.show()
