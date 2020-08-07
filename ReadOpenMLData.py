import time

import matplotlib
from sklearn import datasets, svm

from GetOpenMLData import OpenMLData
from MultiClassSeparation import MultiClassSeparation
from Algos import generate_heatmap, plot, prediction_colormap, plot_prediction

openmlData = OpenMLData(db_id=42110, multilabel=True)
training_sets = openmlData.get_random_training_data_by_class(5)
svm_data = openmlData.make_svm_data(training_sets)
mcs = MultiClassSeparation(training_sets=training_sets, openMLData=openmlData)

start = time.time()
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(svm_data[0], svm_data[1])
prediction = clf.predict(openmlData.data_X)
generate_heatmap(openmlData, prediction, "SVM")
colorlist = prediction_colormap(openmlData, prediction)
plot_prediction(openmlData, prediction, colorlist, dim=2, algo="SVM")
new_time = time.time()
print("SVM {}".format(new_time - start))
start = time.time()
_, prediction, _ = mcs.generalized_algorithm()
generate_heatmap(openmlData, prediction, "Generalized Algo")
colorlist = prediction_colormap(openmlData, prediction)
plot_prediction(openmlData, prediction, colorlist, dim=2, algo="Generalized Algo")
new_time = time.time()
print("Generalized {}".format(new_time - start))
start = time.time()
_, prediction, _ = mcs.one_vs_all()
generate_heatmap(openmlData, prediction, "One vs all")
colorlist = prediction_colormap(openmlData, prediction)
plot_prediction(openmlData, prediction, colorlist, dim=2, algo="One vs all")
new_time = time.time()
print("One vs. all {}".format(new_time - start))
start = time.time()
_, prediction, _ = mcs.one_vs_one(confidence_measure="linkage")
generate_heatmap(openmlData, prediction, "One vs one")
colorlist = prediction_colormap(openmlData, prediction)
plot_prediction(openmlData, prediction, colorlist, dim=2, algo="One vs one")
new_time = time.time()
print("One vs. one {}".format(new_time - start))

