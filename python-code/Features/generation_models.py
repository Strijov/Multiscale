# coding: utf-8
""" Created on 23 November 2016. Author: Alexey Goncharov, Anastasia Motrenko """
import numpy as np
from sklearn.cluster import AffinityPropagation

import nonparametric
import monotone



all_transformations = ['univariate_transformation',
                       # 'bivariate_transformation', this one has additional argument
                       'simple_statistics',
                       'haar_transformations',
                       'monotone_linear',
                       'monotone_polinomial_rate',
                       'monotone_sublinear_polinomial_rate',
                       'monotone_logarithmic_rate',
                       'monotone_slow_convergence',
                       'monotone_fast_convergence',
                       'monotone_soft_relu',
                       'monotone_sigmoid',
                       'monotone_soft_max',
                       'monotone_hyberbolic_tangent',
                       'monotone_softsign']


class FeatureGeneration():

    def __init__(self, name="Nonparametric", replace=True, transformations=all_transformations):
        self.name = name
        self.transformations = []
        self.replace = replace
        self.feature_dict = {}

        if isinstance(transformations, str):
            if transformations.lower() == "all":
                transformations= all_transformations
            else:
                transformations = [transformations.lower()]

        if isinstance(transformations, list) and len(self.transformations) == 0:
            for transf in transformations:
                if transf.lower() in all_transformations:
                    self.transformations.append(transf.lower())
                else:
                    print("Invalid transformation function name {}. Valid options for nonparametric feature "
                          "transformation are {}".format(transf, all_transformations))
                if len(self.transformations) == 0:
                    raise ValueError("None of the functions names passed in 'transformations' are valid.")
        else:
            raise TypeError("Parameter 'transformations' should be either list or string, got {}".
                            format(type(transformations)))

        for i, transf in enumerate(self.transformations):
            for module in [nonparametric, monotone]:
                try:
                    self.transformations[i] = getattr(module, transf)
                    break
                except:
                    pass



    def transform(self, X):
        """
        Applies transformaions from self.transformations to matrix X and returns them horizontally stacked

        :param X: original feature matrix
        :type X: numpy.ndarray
        :return: transformed of the feature matrix. If self.replace is False, this also includes original matrix
        :rtype: numpy.ndarray
        """
        new_X = []

        n_feats = X.shape[1]
        if self.replace:
            self.feature_dict = {}
            n_feats = 0

        for transf in self.transformations:
            new_transf = transf(X)
            new_X.append(new_transf)
            self.feature_dict[transf.__name__] = range(n_feats, n_feats + new_transf.shape[1])
            n_feats += new_transf.shape[1]

        new_X = np.hstack(new_X)
        new_X = replace_inf(new_X)
        if self.replace:
            return new_X

        return np.hstack((X, new_X))


    def fit(self, X, y):
        return self



class Nonparametric(FeatureGeneration):

    def __init__(self, name="Nonparametric", replace=True):
        np_transformations = ['univariate_transformation',
                              # 'bivariate_transformation', this one has additional argument
                              'simple_statistics',
                              'haar_transformations']
        FeatureGeneration.__init__(self, name, replace, np_transformations)



class Monotone(FeatureGeneration):

    def __init__(self, name="Monotone", replace=True):
        monotone_transformations = ['monotone_linear',
                              'monotone_polinomial_rate',
                              'monotone_sublinear_polinomial_rate',
                              'monotone_logarithmic_rate',
                              'monotone_slow_convergence',
                              'monotone_fast_convergence',
                              'monotone_soft_relu',
                              'monotone_sigmoid',
                              'monotone_soft_max',
                              'monotone_hyberbolic_tangent',
                              'monotone_softsign']

        FeatureGeneration.__init__(self, name, replace, monotone_transformations)


class CentroidDistances():

    def __init__(self, name=None, replace=False):
        self.centroids_distances = []
        if name is None:
            self.name = "Centroids"
        self.replace = replace

    def fit(self, trainX, trainY=None):
        af = AffinityPropagation().fit(trainX)
        cluster_centers_indices = af.cluster_centers_indices_
        self.centroids = trainX[cluster_centers_indices, :]

        return self

    def transform(self, X):
        self.centroids_distances = []
        for centr in self.centroids:
            self.centroids_distances.append(np.sqrt(np.sum(np.power(X - centr, 2), axis=1))[:, None])

        if self.replace:
            return np.hstack(self.centroids_distances)

        return np.hstack( (X, np.hstack(self.centroids_distances)) )


def replace_inf(X):
    MAX_VALUE = 1e30
    X[np.isinf(X)] = MAX_VALUE
    return X