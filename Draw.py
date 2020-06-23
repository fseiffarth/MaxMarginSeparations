'''
Created on 07.11.2018

@author: florian
'''

from networkx import nx, planar_layout
import matplotlib.pyplot as plt

#import pygraphviz

from networkx.drawing.nx_agraph import graphviz_layout


def draw_graph_with_labels(GraphGenerationObject):
    # Draw the graph
    pos = graphviz_layout(GraphGenerationObject.Graph, prog='neato')
    nx.draw_networkx_nodes(GraphGenerationObject.Graph, pos, nodelist=GraphGenerationObject.green_nodes,
                           node_color="green")
    nx.draw_networkx_nodes(GraphGenerationObject.Graph, pos, nodelist=GraphGenerationObject.red_nodes, node_color="red")
    nx.draw_networkx_nodes(GraphGenerationObject.Graph, pos, nodelist=GraphGenerationObject.blue_nodes,
                           node_color="blue")
    nx.draw_networkx_labels(GraphGenerationObject.Graph, pos, nodelist=GraphGenerationObject.Graph.nodes())
    nx.draw_networkx_edges(GraphGenerationObject.Graph, pos, GraphGenerationObject.Graph.edges(), edge_color="black")
    plt.axis('off')
    plt.show()



def draw_graph_with_prediction(GraphGenerationObject):
    # Draw the graph
    pos = graphviz_layout(GraphGenerationObject.Graph, prog='neato')
    nx.draw_networkx_nodes(GraphGenerationObject.Graph, pos, nodelist=GraphGenerationObject.learning_green_nodes,
                           node_color="green")
    nx.draw_networkx_nodes(GraphGenerationObject.Graph, pos, nodelist=GraphGenerationObject.learning_red_nodes,
                           node_color="red")
    nx.draw_networkx_nodes(GraphGenerationObject.Graph, pos, nodelist=GraphGenerationObject.blue_nodes,
                           node_color="blue")
    nx.draw_networkx_labels(GraphGenerationObject.Graph, pos, nodelist=GraphGenerationObject.Graph.nodes())
    nx.draw_networkx_edges(GraphGenerationObject.Graph, pos, GraphGenerationObject.Graph.edges(),
                           edge_color="black")
    plt.axis('off')
    plt.show()



def draw_graph_with_labels_training(GraphGenerationObject):
    # Draw the graph
    pos = graphviz_layout(GraphGenerationObject.Graph, prog='neato')
    nx.draw_networkx_nodes(GraphGenerationObject.Graph, pos, nodelist=GraphGenerationObject.learning_green_nodes,
                           node_color="green")
    nx.draw_networkx_nodes(GraphGenerationObject.Graph, pos, nodelist=GraphGenerationObject.learning_red_nodes,
                           node_color="red")
    nx.draw_networkx_nodes(GraphGenerationObject.Graph, pos, nodelist=GraphGenerationObject.blue_nodes,
                           node_color="blue")
    nx.draw_networkx_labels(GraphGenerationObject.Graph, pos, nodelist=GraphGenerationObject.Graph.nodes())
    nx.draw_networkx_edges(GraphGenerationObject.Graph, pos, GraphGenerationObject.Graph.edges(),
                           edge_color="black")
    plt.axis('off')
    plt.show()


def draw_graph_labels(graph, nodes = [], colors = []):
    # Draw the graph
    pos = graphviz_layout(graph, prog='neato')
    nx.draw_networkx_nodes(graph, pos, nodelist=graph.nodes(),
                           node_color="blue")
    for i, x in enumerate(nodes):
        nx.draw_networkx_nodes(graph, pos, nodelist=x,
                           node_color=colors[i])
    nx.draw_networkx_edges(graph, pos, graph.edges(),
                           edge_color="black")
    plt.axis('off')
    plt.show()