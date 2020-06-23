'''
Created on 06.09.2018

@author: florian
'''
from Algos import *

def ConvexHullClassifier(classifier = "opt", plot = False, dimension = 3, number_of_points = 100, number_runs = 1, num_start1 = 3, num_start2 = 3, random = True, points = [], positive_points = [], negative_points = [], y=[]):
        
    classified_complete = 0
    unclassified_complete = 0
    for i in range(0, number_runs):
        if random:
            (Set, Hull1, Hull2) = random_point_set(number_of_points, dimension, num_start1, num_start2)
        else:
            (Set, Hull1, Hull2) = (points, positive_points, negative_points)
        test = PointSet(Set, Hull1, Hull2)
        #test.plot_3d()

        separable = False

        #set the chosen classifier
        if classifier == "opt":
            (calls, classification, separable) = test.optimal_alg()
        elif classifier == "opt_hull":
            (calls, classification, separable) = test.optimal_hull_alg()
        elif classifier == "greedy":
            (calls, classification, separable) = test.greedy_alg()
        elif classifier == "greedy2":
            (calls, classification, separable) = test.greedy_alg2()
        elif classifier == "greedy_fast":
            (calls, classification, separable) = test.greedy_fast_alg()


        colorlist = color_list_testing(y, positive_points, negative_points)

        
        #Plot the classified points
        if plot and separable:
            if dimension == 2:
                test.plot_2d_classification(classifier + str(len(points)) + str(len(positive_points) + len(negative_points)), colorlist)
            else:
                test.plot_3d_classification()

    return classification, separable
