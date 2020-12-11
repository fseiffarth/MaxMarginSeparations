import time

from joblib import Parallel, delayed
from sklearn import datasets, svm

from DataToSQL.DataToSQL import DataToSQL
from MainExperiment import run, run_synthetic
import argparse


def main():
    scr_path = "D:\EigeneDokumente\Forschung\Code/MaxMarginSeparations/ResultsPaper/"
    scr_path = "LongVersion/"

    db_id = 61
    algo = "svm"
    parser = argparse.ArgumentParser(description='Run different separation algorithms on finite point sets.')
    parser.add_argument('-id', '--id', metavar='id', type=int,
                        help='Id of the openMLDataset')
    parser.add_argument('-runs', '--runs', metavar='runs', type=int,
                        help='Number of runs')
    parser.add_argument('-algos', '--algos', nargs='+',
                        help='<Optional> Choose different algorithms from "svm", "maxmargin", "greedy", "OvA", "OvOlinkage", "OvOstd"',
                        required=False)
    parser.add_argument('-train_sizes', '--train_sizes', nargs='+', type=int,
                        help='<Optional> Choose the training sizes per class manually',
                        required=False)
    parser.add_argument('-print_time', '--print_time', type=bool, help='<Optional> Print runtime of single algorithms',
                        required=False)
    parser.add_argument('-linprog', '--linprog', type=bool, help='<Optional> Use linear programming calculation to calculate if element is in convex hull, better for small element number in higher dimensions',
                        required=False)
    parser.add_argument('-synthetic', '--synthetic', type=bool, help='<Optional> Use synthetic datasets in 2, 3, 4 dimensions (use id)',
                        required=False)
    parser.add_argument('-plot', '--plot', type=bool, help='<Optional> Plot results of algorithms', required=False)
    args = parser.parse_args()
    database = DataToSQL.DataToSQL(file_path=scr_path,
                                   db_name="LongVersion")

    algos = (["svm", "maxmargin", "greedy", "OvA", "OvOlinkage", "OvOstd"] if args.algos is None else args.algos)
    print_time = (False if args.print_time is None else args.print_time)
    plot = (False if args.plot is None else args.plot)

    if 2 <= args.id <= 4 and args.synthetic:
        run_synthetic(database=database, algos=algos, dim=args.id, number=args.runs,train_sizes=args.train_sizes, print_time=print_time,
                      plot=plot)
    else:
        run(database, algos=algos, data_id=args.id, number=args.runs,train_sizes=args.train_sizes, print_time=print_time, plot=plot, linprog_calc=args.linprog)


if __name__ == '__main__':
    main()
