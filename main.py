from joblib import Parallel, delayed

from DataToSQL.DataToSQL import DataToSQL
from evaluation import evaluation_points, evaluation_graphs, evaluation_trees
from graph_separation import run_single_experiment
from point_separation import PointSeparation
from tree_separation import TreeSeparation


def max_margin_separation_points(scr_path, db_name):
    database = DataToSQL.DataToSQL(file_path=scr_path,
                                   db_name=db_name)
    dimensions = [[(0, 0), (4, 0)], [(0, 0, 0), (4, 0, 0)], [(0, 0, 0, 0), (4, 0, 0, 0)]]

    for d in dimensions:
        separation = PointSeparation(database=database, dataset="Synthetic", classifiers=["greedy", "opt", "svm"],
                                     point_cloud_sizes=[1000], training_sizes=[3, 4, 5, 10, 25, 50], run_number=1,
                                     centers=d, plotting=False, job_num=16)

        separation.run_experiment()


def max_margin_separation_trees(scr_path, db_name):
    dataset = DataToSQL.DataToSQL(file_path=scr_path,
                                  db_name=db_name)
    tree_separation = TreeSeparation(dataset=dataset, classifiers=["greedy", "opt"],
                                     tree_sizes=[1000, 20000],
                                     training_sizes=[1, 2, 3, 4, 5], number_of_examples=10,
                                     step_number=1, class_balance=1 / 4, plotting=False, job_num=16)

    tree_separation.run_tree_experiment()


def max_margin_separation_graphs(scr_path, db_name):
    dataset = DataToSQL.DataToSQL(file_path=scr_path,
                                  db_name=db_name)
    columns = ['Accuracy', 'Coverage', 'Correct', 'Unclassified',
               'RedCorrect', 'GreenCorrect', 'RedSize', 'GreenSize', 'TargetRedSize', 'TargetGreenSize', 'GraphSize',
               'EdgeDensity']
    column_types = ["FLOAT" for x in columns]

    for num_nodes in [100]:
        for train_set_size in range(1, 3):
            for num in range(0, 1):
                Parallel(n_jobs=4)(
                    delayed(run_single_experiment)(dataset, columns, column_types, num_nodes, edge_density / 10.0,
                                                   num * num_nodes * edge_density, train_set_size)
                    for edge_density in range(10, 16, 1))


def main():
    scr_path = "~/MaxMarginSeparations/ResultsPaper/"
    scr_path = "~/MaxMarginSeparations/ResultsNew/"
    db_points = "PointSeparation"
    db_graphs = "GraphSeparation"
    db_trees = "TreeSeparation"
    #max_margin_separation_graphs(scr_path, db_graphs)
    #max_margin_separation_points(scr_path, db_points)
    #max_margin_separation_trees(scr_path, db_trees)
    print("Experiments Finished")
    print("Start Evaluation")
    evaluation_points(scr_path, db_points)
    evaluation_graphs(scr_path, db_graphs)
    evaluation_trees(scr_path, db_trees)


if __name__ == '__main__':
    main()
