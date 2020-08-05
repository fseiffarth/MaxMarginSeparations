'''
Created on 06.09.2018

@author: florian
'''
from Algos import *


def convex_hull_classifier(classifier="opt", plotting=False, dimension=3, number_of_points=100, number_runs=1, training_size_a=3,
                           training_size_b=3, random_training_set=True, E=[], A_elements=[], B_elements=[], E_labels=[]):
    classified_complete = 0
    unclassified_complete = 0
    for i in range(0, number_runs):
        if random_training_set:
            (E, set_a, set_b) = random_point_set(number_of_points, dimension, training_size_a, training_size_b)
        else:
            (E, set_a, set_b) = (E, A_elements, B_elements)
        classification_setup = ClassificationPointSet(E=E, A=set_a, B=set_b)
        # classification_setup.plot_3d()

        separable = False

        # set the chosen classifier
        if classifier == "opt":
            (calls, classification, separable) = classification_setup.optimal_alg()
        elif classifier == "opt_hull":
            (calls, classification, separable) = classification_setup.optimal_hull_alg()
        elif classifier == "greedy":
            (calls, classification, separable) = classification_setup.greedy_alg()
        elif classifier == "greedy2":
            (calls, classification, separable) = classification_setup.greedy_alg2()
        elif classifier == "greedy_fast":
            (calls, classification, separable) = classification_setup.greedy_fast_alg()

        color_list = color_list_testing(E_labels, A_elements, B_elements)

        # Plot the classified E
        if plotting and separable:
            if dimension == 2:
                classification_setup.plot_2d_classification(
                    classifier + str(len(E)) + str(len(A_elements) + len(B_elements)), color_list)
            else:
                classification_setup.plot_3d_classification()

    return classification, separable
