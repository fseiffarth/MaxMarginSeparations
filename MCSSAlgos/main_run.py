import logging
import sys
import time

from joblib import Parallel, delayed
from tqdm import tqdm, trange

from DataToSQL.DataToSQL import DataToSQL
from MCSSAlgos.Closures import GraphClosures
from MCSSAlgos.DomainSpecificMCSS import DomainSpecificMCSS

from MCSSAlgos.Domains.Graphs import Graph
from MCSSAlgos.InitUpdate import GraphInitUpdate

import click

from MCSSAlgos.utils.graph_utils import random_graph
from MCSSAlgos.utils.helper import load_params
from MCSSAlgos.utils.train_set_utils import create_initial_sets

_logger = logging.getLogger(__name__)


def main_run(database, table_name, p_bar, graph_sizes, graph_densities, class_start_sizes, closure_operators,
             query_strategies):
    values = []
    for i in trange(len(graph_sizes), desc="Graph Size Loop", leave=False):
        graph_size = graph_sizes[i]
        for graph_density in graph_densities:
            graph = random_graph(graph_size, graph_density)
            domain = Graph(graph)
            for class_start_size in class_start_sizes:
                initial_sets = create_initial_sets(domain, class_start_size)
                for cl_name in closure_operators:
                    cl_operator = GraphClosures.get_graph_closure(cl_name)
                    for strategy_name in query_strategies:
                        query_strategy = GraphInitUpdate.get_query_strategy(strategy_name, threshold=cl_operator.threshold)
                        start = time.time()
                        dsmcss = DomainSpecificMCSS(domain, cl_operator, query_strategy)
                        output = dsmcss.run(input_sets=initial_sets)

                        initial_size = len(set.union(*initial_sets))
                        output_size = len(set.union(*output[0]))

                        if output != "NO":
                            table_values = [cl_name, strategy_name, graph_size, graph_density,
                                            initial_size, output_size, output_size/domain.size, output[1],
                                            time.time() - start, len(output[2]), len(output[3])]
                            values.append(table_values)

    table_attributes = ["Closure", "Strategy", "GraphSize", "GraphDensity", "InitialSetSize",
                        "OutputSize", "OutputPercentage", "QueryNumber", "Runtime", "ExtendedElements",
                        "ConsideredElements"]
    table_attributes_types = ["TEXT", "TEXT", "INTEGER", "REAL", "INTEGER", "INTEGER", "REAL"
                              "INTEGER", "REAL", "INTEGER", "INTEGER"]

    database.experiment_to_database(table_name, table_attributes, values,
                                                            experiment_attributes_type=table_attributes_types)
    p_bar.update()

@click.command()
@click.option('-c', '--config', 'cfg_path', required=True,
              type=click.Path(exists=True), help='Path to experiment config file')
def main(cfg_path):
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    params = load_params(cfg_path, _logger)

    db_name = params["db_name"]
    table_name = params["table_name"]
    output_path = params["output_path"]
    closure_operators = params["closure_operators"]
    query_strategies = params["query_strategies"]
    graph_sizes = params["graph_sizes"]
    graph_densities = params["graph_densities"]
    class_start_sizes = params["class_start_sizes"]
    number_of_repetitions = params["number_of_repetitions"]

    database = DataToSQL.DataToSQL(output_path, db_name)
    p_bar = tqdm(total=number_of_repetitions, unit="repetition", desc="Overall experiment")

    for _ in range(number_of_repetitions):
        main_run(database, table_name, p_bar, graph_sizes, graph_densities, class_start_sizes,
                                             closure_operators,
                                             query_strategies)


if __name__ == '__main__':
    main()
