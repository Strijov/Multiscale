# coding: utf-8
""" Created on 23 November 2016. Author: Alexey Goncharov, Anastasia Motrenko """
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from collections import defaultdict

import nonparametric
import monotone
import parametric

MODULES = [parametric, nonparametric, monotone]

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


class Feature():
    constraints = []
    variables = []
    selected = None


class FeatureTransformation():

    def __init__(self, transformation=None, name=None, replace=True):

        self.name = name
        self.replace = replace

        self.constraints = []
        self.variables = None

        self.transformation = None
        if transformation is None:
            return

        for module in MODULES:
            try:
                self.transformation = getattr(module, transformation)
                break
            except AttributeError:
                pass
        if self.transformation is None:
            raise ValueError("Transformation {} was not found in modules {}"
                              .format(transformation, MODULES))

        if name is None:
            self.name = self.transformation.__name__

        try:
            self.variables, self.constraints = getattr(module, "_set_"+transformation)()
        except AttributeError:
            pass

    def transform(self, X):
        if self.transformation is None:
            return X
        if self.variables is not None:
            return self.transformation(X, np.squeeze(np.asarray(self.variables.value)))

        return self.transformation(X)

    def fit(self, X):
        return self


class FeatureGeneration(BaseEstimator):
    """ Applies feature transformation from package Features"""

    def __init__(self, name="Nonparametric", replace=True,
                 transformations=all_transformations,
                 norm=True):
        self.name = name
        self.transformations = []
        self.constraints = []
        self.variables = []
        self.replace = replace
        self.feature_dict = {}
        self.norm = norm
        self.norm_consts = defaultdict(list)
        self.variables = []
        self.constraints = []

        if not isinstance(transformations, list):
            transformations = [transformations]

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
            new_transform = FeatureTransformation(transformation=transf)
            self.transformations[i] = new_transform
            self.variables.append(new_transform.variables)
            self.constraints.extend(new_transform.constraints)
            self.replace = self.replace or new_transform.replace

        Feature.constraints = self.constraints
        Feature.variables = self.variables


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
            new_transf = _replace_inf(transf.transform(X))
            if self.norm:
                new_transf = self.norm_consts[transf.name].transform(new_transf)

            new_feats.append(new_transf)
            n_feats += new_transf.shape[1]

        new_feats = np.hstack(new_feats)
        if self.replace:
            return new_feats

        return np.hstack((X, new_feats))


    def fit(self, X, y):
        """ For now, no stick with nonparametric transformation """

        if not self.norm:
            return self


        for transf in self.transformations:
            new_transf = _replace_inf(transf.transform(X))
            self.norm_consts[transf.name] = StandardScaler().fit(new_transf)
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