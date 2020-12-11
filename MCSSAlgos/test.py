import time

from MCSSAlgos.Closures.GraphClosures import GraphClosureSP, GraphClosureSPFast
from MCSSAlgos.DomainSpecificMCSS import DomainSpecificMCSS
from networkx import nx

from MCSSAlgos.Domains.Graphs import Graph
from MCSSAlgos.InitUpdate.GraphInitUpdate import GraphInitUpdateGreedy, GraphInitUpdateNN, GraphInitUpdateFarthestPoint
from MCSSAlgos.utils.graph_utils import random_graph


def print_result(name, output, start_time):
    print(name)
    print(
        "OutputSize: {}, Partitions: {}, Query Number: {}, Runtime: {}, Extended Elements: {}, Considered Elements: {}".format(
            len(set.union(*output[0])), output[0], output[1], time.time() - start_time, output[2], output[3]))


for x in range(10):
    # graph = nx.dense_gnm_random_graph(1000, 6000, seed=10)
    graph = nx.random_tree(6, seed=55)
    graph = random_graph(node_number=6, edge_density=1.2, seed=36)
    #nx.write_graphml(graph, 'Graphs/graph.gml')
    #graph = nx.Graph()

    #graph.add_nodes_from(list(range(6)))
    #graph.add_edges_from([(0, 1), (0, 3), (1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])

    graph_dom = Graph(graph)



    if nx.is_connected(graph_dom.data_object):
        # Test Graph closure speed
        # print(graph_dom.all_shortest_paths[0][507])
        print("\nRun: {}".format(x))
        start = time.time()
        gclfast = GraphClosureSPFast()
        x = gclfast.cl(set([0, 4]), graph_dom)
        print(x, time.time() - start)
        start = time.time()
        gclslow = GraphClosureSP()
        x = gclslow.cl(set([0, 4]), graph_dom)
        print(x, time.time() - start)

        start = time.time()
        dsmcss = DomainSpecificMCSS(graph_dom, GraphClosureSPFast(threshold=2), GraphInitUpdateNN(threshold=2))
        output = dsmcss.run([{0}, {5}])
        print_result("NNStrategy", output=output, start_time=start)

        start = time.time()
        dsmcss = DomainSpecificMCSS(graph_dom, GraphClosureSPFast(threshold=2), GraphInitUpdateFarthestPoint())
        output = dsmcss.run([{0}, {5}])
        print_result("FarthestPoint", output=output, start_time=start)

        start = time.time()
        dsmcss = DomainSpecificMCSS(graph_dom, GraphClosureSPFast(threshold=2), GraphInitUpdateGreedy())
        output = dsmcss.run([{0}, {5}])
        print_result("Basic Greedy", output=output, start_time=start)


        start = time.time()
        dsmcss = DomainSpecificMCSS(graph_dom, GraphClosureSPFast(threshold=2), GraphInitUpdateGreedy(random_init=True))
        output = dsmcss.run([{0}, {5}])
        print_result("RandomGreedy", output=output, start_time=start)

        # start = time.time()
        # dsmcss = DomainSpecificMCSS(graph_dom, GraphClosureSP(), GraphInitUpdateGreedy())
        # output = dsmcss.run([{0}, {3}])
        # print(output, time.time()-start)

    else:
        print("Graph is unconnected")
