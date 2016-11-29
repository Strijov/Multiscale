# coding: utf-8
""" Created on 23 November 2016. Author: Alexey Goncharov, Anastasia Motrenko """
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from collections import defaultdict

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


class FeatureGeneration(BaseEstimator):
    """ Applies feature transformation from package Features"""

    def __init__(self, name="Nonparametric", replace=True,
                 transformations=all_transformations,
                 norm=True):
        self.name = name
        self.transformations = []
        self.replace = replace
        self.feature_dict = {}
        self.norm = norm
        self.norm_consts = defaultdict(list)

        if not isinstance(transformations, list):
            transformations = [transformations]
        # if isinstance(transformations, str):
        #     if transformations.lower() == "all":
        #         transformations = all_transformations
        #     else:
        #         transformations = [transformations.lower()]

        # if isinstance(transformations, list) and len(self.transformations) == 0:
        for transf in transformations:
            if transf is None:
                self.transformations = []
                return
            elif transf.lower() in all_transformations:
                self.transformations.append(transf.lower())
            elif transf.lower() == 'centroids':
                self = CentroidDistances(name=self.name, replace=False)  # FIXIT
                return
            elif transf.lower() == 'all':
                self.transformations = all_transformations
                break
            else:
                print("Invalid transformation function name {}. Valid options for nonparametric feature "
                      "transformation are {}".format(transf, all_transformations))

        if len(self.transformations) == 0:
            raise ValueError("None of the functions names passed in 'transformations' are valid.")
        # else:
            #     raise TypeError("Parameter 'transformations' should be either list or string, got {}".
            #                     format(type(transformations)))

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
        if len(self.transformations) == 0:
            return X

        new_feats = []

        n_feats = X.shape[1]
        if self.replace:
            self.feature_dict = {}
            n_feats = 0

        for transf in self.transformations:
            new_transf = _replace_inf(transf(X))
            if self.norm:
                new_transf = self.norm_consts[transf.__name__].transform(new_transf)

            new_feats.append(new_transf)
            self.feature_dict[transf.__name__] = range(n_feats, n_feats + new_transf.shape[1])
            n_feats += new_transf.shape[1]

        new_feats = np.hstack(new_feats)
        #new_feats = _replace_inf(new_feats)
        if self.replace:
            return new_feats

        return np.hstack((X, new_feats))


    def fit(self, X, y):
        """ For now, no stick with nonparametric transformation """

        if not self.norm:
            return self

        for transf in self.transformations:
            new_transf = _replace_inf(transf(X))
            self.norm_consts[transf.__name__] = StandardScaler().fit(new_transf)
        return self


class Nonparametric(FeatureGeneration):
    """ Special case of FeatureGeneration model """

    def __init__(self, name="Nonparametric", replace=True, norm=True):
        np_transformations = ['univariate_transformation',
                              # 'bivariate_transformation', this one has additional argument
                              'simple_statistics',
                              'haar_transformations']
        FeatureGeneration.__init__(self, name, replace, np_transformations)


class Monotone(FeatureGeneration):
    """ Special case of FeatureGeneration model """

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


class CentroidDistances(BaseEstimator):
    """ Computes features based on distance to centriods """

    def __init__(self, name=None, replace=False):
        self.centroids_distances = []
        if name is None:
            self.name = "Centroids"
        self.replace = replace
        self.centroids = None

    def fit(self, train_x, _=None):

        af = AffinityPropagation().fit(train_x)
        cluster_centers_indices = af.cluster_centers_indices_
        self.centroids = train_x[cluster_centers_indices, :]

        return self

    def transform(self, X):
        self.centroids_distances = []
        for centr in self.centroids:
            self.centroids_distances.append(np.sqrt(np.sum(np.power(X - centr, 2), axis=1))[:, None])

        if self.replace:
            return np.hstack(self.centroids_distances)

        return np.hstack( (X, np.hstack(self.centroids_distances)) )


def _replace_inf(X):
    if np.sum(np.isinf(X)) == 0:
        return X
    print(X[np.isinf(X)])
    MAX_VALUE = 1e30
    X[np.isinf(X)] = MAX_VALUE * np.sign(X[np.isinf(X)])
    return X