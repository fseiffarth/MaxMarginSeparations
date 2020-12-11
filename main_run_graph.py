from DataToSQL.DataToSQL import DataToSQL
import argparse

from graph_multiclass import run_graph


def main():
    scr_path = "LongVersion/"

    parser = argparse.ArgumentParser(description='Run different separation algorithms on trees and graphs.')
    parser.add_argument('-is_tree', '--is_tree', type=int, help='Run algorithms with trees or graphs')
    parser.add_argument('-runs', '--runs', metavar='runs', type=int,
                        help='Number of runs for each size')
    parser.add_argument('-graph_sizes', '--graph_sizes', nargs='+', type=int,
                        help='Choose the data set size')
    parser.add_argument('-class_numbers', '--class_numbers', nargs='+', type=int,
                        help='Choose the number of classes manually')
    parser.add_argument('-database_name', '--database_name', type=str,
                        help='<Optional> Choose the database name manually, standard is LongVersionGraphs',
                        required=False)
    parser.add_argument('-train_sizes', '--train_sizes', nargs='+', type=int,
                        help='<Optional> Choose the training sizes per class manually',
                        required=False)
    parser.add_argument('-densities', '--densities', nargs='+', type=float,
                        help='<Optional> Choose the density of the graphs',
                        required=False)
    parser.add_argument('-algos', '--algos', nargs='+',
                        help='<Optional> Choose different algorithms from "maxmargin", "greedy"',
                        required=False)

    parser.add_argument('-plot', '--plot', type=bool, help='<Optional> Plot results of algorithms', required=False)
    args = parser.parse_args()
    db_name = ("LongVersionGraphs" if args.database_name is None else args.database_name)
    database = DataToSQL.DataToSQL(file_path=scr_path,
                                   db_name=db_name)

    algos = (["greedy", "max_margin"] if args.algos is None else args.algos)
    plot = (False if args.plot is None else args.plot)
    densities = ([1] if args.is_tree else args.densities)

    run_graph(database, algos=algos, graph_sizes=args.graph_sizes, graph_densities=densities,
              class_numbers=args.class_numbers,
              train_sizes=args.train_sizes, number=args.runs, is_tree=args.is_tree, print_time=False, plot=plot)


if __name__ == '__main__':
    main()
