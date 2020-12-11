import argparse

import matplotlib

from DataToSQL.DataToSQL import DataToSQL
import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np

from GetOpenMLData import OpenMLData


def evaluation(dataset, data_id, train_sizes=[], quantities=["CoverageW"], path="", std=True):
    marks = ["o", "x", "o", "o", "o", "o"]

    all_algos = ["svm", "greedy", "maxmargin", "OvA", "OvOstd", "OvOlinkage", "majority_baseline"]
    colormap = "viridis"
    cmap = matplotlib.cm.get_cmap(colormap)
    algo_color_map = {}
    for i, algo in enumerate(all_algos):
        algo_color_map[algo] = cmap(float(i) / (len(all_algos) - 1))


    fig, axs = plt.subplots(3, 2)
    for q, quantity in enumerate(quantities):
        l, k = q // 2, q % 2
        print(l, k)
        axs[l, k].set_xlabel("Training Set Size")
        axs[l, k].set_ylabel(quantity)

        query = 'SELECT Algo FROM Data_' + str(
            data_id) + ' GROUP BY Algo'
        algos = dataset.get_from_query(query)
        algos.append("majority_baseline")
        for a, algo in enumerate(algos, 0):
            algo = algo[0]
            x_values = []
            y_values = []
            y_values_std = []

            if not train_sizes:
                query = 'SELECT TrainingSize FROM Data_' + str(
                    data_id) + ' GROUP BY TrainingSize'
                val = dataset.get_from_query(query)
                train_sizes = [int(x[0]) for x in val]

            for i, x in enumerate(train_sizes, 0):
                if algo == "majority_baseline":
                    x_values.append(x)
                    if data_id in [2, 3, 4]:
                        synthetic = True
                    else:
                        synthetic = False
                    y_values.append(OpenMLData(data_id, synthetic=synthetic))
                else:
                    if std == True:
                        std_query = 'SELECT Sum((Quantity-Average)*(Quantity-Average))/Count(*) FROM (SELECT Average, Algo, ' + quantity + ' as Quantity FROM (SELECT avg(' + str(
                            quantity) + ') as Average, Algo as A From Data_' + str(
                            data_id) + ' WHERE TrainingSize =' + str(x) + ' AND Algo="' + algo + '") as t JOIN Data_' + str(
                            data_id) + ' WHERE t.A = Data_' + str(
                            data_id) + '.Algo AND TrainingSize =' + str(x) + ')'
                        query = 'SELECT Avg(' + quantity + ') FROM Data_' + str(
                            data_id) + ' WHERE Algo="' + algo + '" AND TrainingSize =' + str(x)
                    else:
                        query = 'SELECT Avg(' + quantity + ') FROM Data_' + str(
                            data_id) + ' WHERE Algo="' + algo + '" AND TrainingSize =' + str(x)
                    val = dataset.get_from_query(query)[0][0]
                    val_std = dataset.get_from_query(std_query)[0][0]
                if val is not None:
                    x_values.append(x)
                    y_values.append(val)
                if val_std is not None:
                    val_std = np.sqrt(val_std)
                    y_values_std.append(val_std)

            axs[l, k].plot(x_values, y_values, c=algo_color_map[algo], marker=marks[a])
            axs[l, k].fill_between(x_values, y_values, [y_values[i] - y_values_std[i] for i in range(len(y_values))],
                                   facecolor=algo_color_map[algo], alpha=0.2)
            axs[l, k].fill_between(x_values, y_values, [y_values[i] + y_values_std[i] for i in range(len(y_values))],
                                   facecolor=algo_color_map[algo], alpha=0.2)
    tikzplotlib.save(path + "Tikzpictures/" + "Data_" + str(
        data_id) + ".tex")
    plt.show()


def evaluation_by_quantity(database, ids, quantity, path, train_sizes=[], std=True):
    marks = ["v", "x", "s", "^", "*", "."]

    all_algos = ["svm", "greedy", "maxmargin", "OvA", "OvOstd", "OvOlinkage"]
    database_map = {2:"SYNTHETIC2D", 3:"SYNTHETIC3D", 4:"SYNTHETIC4D", 1462:"BANKNOTE", 1460:"BANANA", 61:"IRIS", 42110:"S1", 1499:"SEEDS", 679:"SLEEP", 1523:"VERTEBRA", 1541:"VOLCANOES4D", 803: "DELTAAILERONS"}
    colormap = "viridis"
    cmap = matplotlib.cm.get_cmap(colormap)
    algo_color_map = {}
    algo_mark_map = {}
    for i, algo in enumerate(all_algos):
        algo_color_map[algo] = cmap(float(i) / (len(all_algos) - 1))
        algo_mark_map[algo] = marks[i]

    fig, axs = plt.subplots(2, 3)
    for q, id in enumerate(ids):
        k=q//3
        l=q%3
        train_sizes = []

        query = 'SELECT Algo FROM Data_' + str(
            id) + ' GROUP BY Algo'
        algos = database.get_from_query(query)
        lines = []
        for a, algo in enumerate(algos, 0):
            algo = algo[0]
            x_values = []
            y_values = []
            y_values_std = []

            if not train_sizes:
                query = 'SELECT TrainingSize FROM Data_' + str(
                    id) + ' GROUP BY TrainingSize'
                val = database.get_from_query(query)
                train_sizes = [int(x[0]) for x in val]

            for i, x in enumerate(train_sizes, 0):
                out_string = ""
                if std == True:
                    std_query = 'SELECT Sum((Quantity-Average)*(Quantity-Average))/Count(*) FROM (SELECT Average, Algo, ' + quantity + ' as Quantity FROM (SELECT avg(' + str(
                        quantity) + ') as Average, Algo as A From Data_' + str(
                        id) + ' WHERE TrainingSize =' + str(x) + ' AND Algo="' + algo + '") as t JOIN Data_' + str(
                        id) + ' WHERE t.A = Data_' + str(
                        id) + '.Algo AND TrainingSize =' + str(x) + ')'
                    query = 'SELECT Avg(' + quantity + ') FROM Data_' + str(
                        id) + ' WHERE Algo="' + algo + '" AND TrainingSize =' + str(x)
                else:
                    query = 'SELECT Avg(' + quantity + ') FROM Data_' + str(
                        id) + ' WHERE Algo="' + algo + '" AND TrainingSize =' + str(x)
                val = database.get_from_query(query)[0][0]
                val_std = database.get_from_query(std_query)[0][0]
                if val is not None:
                    x_values.append(x)
                    y_values.append(val)
                if val_std is not None:
                    val_std = np.sqrt(val_std)
                    y_values_std.append(val_std)

            axs[k, l].set_title(database_map[id])
            if quantity!="Runtime":
                axs[k, l].set_ylim((0, 1.05))
            line, = axs[k, l].plot(x_values, y_values, c=algo_color_map[algo], marker=algo_mark_map[algo], label=str(algo))
            lines.append(line)
            axs[k, l].fill_between(x_values, y_values, [y_values[i] - y_values_std[i] for i in range(len(y_values))],
                                   facecolor=algo_color_map[algo], alpha=0.2)
            axs[k, l].fill_between(x_values, y_values, [y_values[i] + y_values_std[i] for i in range(len(y_values))],
                                   facecolor=algo_color_map[algo], alpha=0.2)
    axs[1, 1].legend(loc=8, bbox_to_anchor=(0.5, -0.4), ncol=6)
    if quantity!="Runtime":
        tikzplotlib.save(path + "Tikzpictures/" + str(quantity) + ".tex", extra_tikzpicture_parameters={f'scale=0.55'}, extra_groupstyle_parameters={f'group name={"name"}', f'horizontal sep={0}', f'vertical sep={45}', f'yticklabels at={"edge left"}'})
    else:
        tikzplotlib.save(path + "Tikzpictures/" + str(quantity) + ".tex", extra_tikzpicture_parameters={f'scale=0.55'}, extra_groupstyle_parameters={f'group name={"name"}', f'vertical sep={45}'})
    plt.show()


def main():
    scr_path = "LongVersion/"

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-id', '--id', type=int,
                        help='Id of the openMLDataset')
    parser.add_argument('-synthetic', '--synthetic', type=bool, help='<Optional> Use synthetic datasets in 2, 3, 4 dimensions (use id)',
                        required=False)
    args = parser.parse_args()

    database = DataToSQL.DataToSQL(file_path=scr_path,
                                   db_name="LongVersion")
    quantities = ["Accuracy", "AccuracyW", "Coverage", "CoverageW", "Runtime"]

    evaluation(database, data_id=args.id, quantities=quantities, path=scr_path)

    #ids = [2, 3, 4, 1462, 1460, 803]
    ids = [61, 42110, 1499, 679, 1523, 1541]
    for quantity in quantities:
        evaluation_by_quantity(database, ids=ids, quantity=quantity, path=scr_path)


if __name__ == '__main__':
    main()
