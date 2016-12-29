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
                       'monotone_softsign',
                       'centroids']

class Feature:
    pass

class BaseFeatureTransformation(BaseEstimator):
    """
    Describes feature transformation
    """
    def __init__(self):
        pass

    def transform(self, X):
        return X

    def fit(self, X, y=None):
        return self


class FeatureTransformation(BaseFeatureTransformation):
    """Describes feature transformation

    :param name: feature transformation. Accepted options are: 'univariate_transformation',
                'simple_statistics', 'haar_transformations', 'monotone_linear', 'monotone_polinomial_rate',
                'monotone_sublinear_polinomial_rate', 'monotone_logarithmic_rate', 'monotone_slow_convergence',
                'monotone_fast_convergence', 'monotone_soft_relu', 'monotone_sigmoid', 'monotone_soft_max',
                'monotone_hyberbolic_tangent', 'monotone_softsign'
    :type transformations: str
    """
    def __init__(self, name=None):

        self.name = name

        if self.name == 'centroids':
            self = CentroidDistances(name=self.name, replace=False)
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
    """
    Applies feature transformations from package Features
    """

    def __init__(self, name, transformations=[]):
        """
        :param name: optional
        :type name: str
        :param transformations: instances inherited from BaseFeatureTransformation
        :type transformations: list
        :param norm: defines if feature block should be standardized after transformation
        :type norm: bool
        """
        self.name = name

        if not isinstance(transformations, list):
            transformations = [transformations]

        self.transformations = transformations

        if len(self.transformations) == 0:
            raise ValueError("None of the functions names passed in 'transformations' are valid")

        for transf in transformations:
            if not isinstance(transf, BaseFeatureTransformation):
                raise ValueError("Feature transforms must be inherited from FeatureTransformation")

    def transform(self, X):
        """
        Applies transformaions from self.transformations to matrix X and returns them horizontally stacked

        :param X: original feature matrix
        :type X: numpy.ndarray
        :return: transformed of the feature matrix. If self.replace is False, this also includes original matrix
        :rtype: numpy.ndarray
        """
        return np.hstack([transf.transform(X) for transf in self.transformations])

    def fit_transform(self, X, y):
        return np.hstack([
            transf.fit_transform(X, y)
                if hasattr(transf, "fit_transform")
                else transf.fit(X, y).transform(X)
            for transf in self.transformations
        ])

    def fit(self, X, y):
        for transf in self.transformations:
            transf.fit(X, y)
        return self


class CentroidDistances(BaseFeatureTransformation):
    """ Computes features based on distance to centriods """

    def __init__(self, name=None, replace=False):
        super(CentroidDistances, self).__init__()
        self.centroids_distances = []
        if name is None:
            self.name = "Centroids"
        self.replace = replace
        self.centroids = None

    def fit(self, X, y=None):

        af = AffinityPropagation().fit(X)
        cluster_centers_indices = af.cluster_centers_indices_
        self.centroids = X[cluster_centers_indices, :]

        return self

    def transform(self, X):
        self.centroids_distances = []
        for centr in self.centroids:
            self.centroids_distances.append(np.sqrt(np.sum(np.power(X - centr, 2), axis=1))[:, None])

        if self.replace:
            return np.hstack(self.centroids_distances)

        return np.hstack([X, np.hstack(self.centroids_distances)])
