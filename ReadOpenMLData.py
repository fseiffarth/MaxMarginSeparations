import time
from sklearn import datasets, svm

from GetOpenMLData import OpenMLData
from MultiClassSeparation import MultiClassSeparation
from Algos import generate_heatmap, plot, prediction_colormap, plot_prediction
from MainExperiment import run

# working 42110, 61, 1523, 1499, 1541, 1552, 679
# binary working 37, 1462, 1460, 803
# 728, 310

# not working 40922

run("svm", 61, 5, 5)

"""
plot = False

openmlData = OpenMLData(db_id=61, multilabel=True)
training_sets = openmlData.get_random_training_data_by_class(15)
svm_data = openmlData.make_svm_data(training_sets)
mcs = MultiClassSeparation(training_sets=training_sets, openMLData=openmlData)

start = time.time()
clf = svm.SVC(kernel='linear', C=1000, break_ties=True)
clf.fit(svm_data[0], svm_data[1])
prediction = clf.predict(openmlData.data_X)
new_time = time.time()
generate_heatmap(openmlData, prediction, "SVM")
colorlist = prediction_colormap(openmlData, prediction)
plot_prediction(openmlData, prediction, colorlist, dim=2, algo="SVM")
print("SVM {}".format(new_time - start))
start = time.time()
_, prediction, _ = mcs.generalized_algorithm()
new_time = time.time()
generate_heatmap(openmlData, prediction, "Generalized Algo")
colorlist = prediction_colormap(openmlData, prediction)
plot_prediction(openmlData, prediction, colorlist, dim=2, algo="Generalized Algo")
print("Generalized {}".format(new_time - start))
start = time.time()
_, prediction, _ = mcs.one_vs_all()
new_time = time.time()
generate_heatmap(openmlData, prediction, "One vs all")
colorlist = prediction_colormap(openmlData, prediction)
plot_prediction(openmlData, prediction, colorlist, dim=2, algo="One vs all")
print("One vs. all {}".format(new_time - start))
start = time.time()
_, prediction, _ = mcs.one_vs_one(confidence_measure="linkage")
new_time = time.time()
generate_heatmap(openmlData, prediction, "One vs one")
colorlist = prediction_colormap(openmlData, prediction)
plot_prediction(openmlData, prediction, colorlist, dim=2, algo="One vs one")
print("One vs. one {}".format(new_time - start))
"""
