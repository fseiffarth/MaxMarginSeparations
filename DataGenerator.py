from sklearn import svm
from sklearn.datasets import make_blobs

from Algos import PointSet, set_training_testing, plot
import numpy as np
from mathutils.geometry import intersect_line_plane

class DataGenerator:
    @staticmethod
    def generate_blobs(n_samples, centers, random_state=0, cluster_std=1, shuffle=False, separable=False):
        X, y = make_blobs(n_samples, n_features=len(centers[0]),
                          centers=[centers[0], centers[1]],
                          cluster_std=cluster_std,
                          shuffle=shuffle, random_state=random_state)
        if not separable:
            return X, y
        else:
            clf = svm.SVC(kernel='linear', C=1000)
            clf.fit(X, y)
            classification = clf.predict(X)

            w = clf.coef_
            b = clf.intercept_
            wrong_classified = np.nonzero(classification - y)
            for x in wrong_classified[0]:
                X[x] = DataGenerator.reflect_point(X[x], w, b)

            return X, y

    @staticmethod
    def reflect_point(point, w, b):

        i = 0
        while w[0][i] == 0:
            i += 1
        plane_point = np.zeros(len(w[0]))
        plane_point[i] = -b / w[0][i]

        sign = np.sign(np.dot(plane_point-point, w[0]))

        distance = np.abs(np.dot(point, w[0]) + b)/np.dot(w[0], w[0])
        return point + sign*2*distance*w[0]

def main():
    X, y = DataGenerator.generate_blobs(1000, [[0, 0], [4, 0]], separable=True)

    colorlist = np.where(y==0, "orange", "violet")
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)
    plot(X, colorlist, dim=2, model=clf)


if __name__ == '__main__':
    main()
