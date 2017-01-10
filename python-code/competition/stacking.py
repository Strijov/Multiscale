# -*- coding: utf-8 -*-


"""Stacking learning method library"""


import numpy as np
from sklearn.cross_validation import KFold
from sklearn.base import ClassifierMixin, BaseEstimator
import scipy


class Stacking(BaseEstimator, ClassifierMixin):
    """Base class for stacking method of learning"""

    def __init__(self, base_estimators, meta_fitter=None, get_folds=None, n_folds=3, extend_meta=False):
        """Initialize Stacking

        Input parameters:
            base_estimators --- list of tuples (fit(X, y), predict(clf, X)) -- base estimators
            meta_fitter --- meta classifier
            split --- split strategy
        """
        self.base_estimators = base_estimators
        self.meta_fitter = meta_fitter
        if get_folds:
            self.get_folds = get_folds
        else:
            self.get_folds = lambda y, n_folds: KFold(len(y), n_folds, True)
        self.n_folds = n_folds
        self.extend_meta = extend_meta

    def fit(self, X, y):
        """Build compositions of classifiers.

        Input parameters:
            X : array-like or sparse matrix of shape = [n_samples, n_features]         
            y : array-like, shape = [n_samples]

        Output parameters:
            self : object
        """
        self.base_predictors = []
        X = scipy.sparse.csr_matrix(X)
        y = np.array(y)

        self.X_meta, self.y_meta = [], []
        for base_subsample, meta_subsample in self.get_folds(y, self.n_folds):
            meta_features = [X[meta_subsample]] if self.extend_meta else []
            for fit, predict in self.base_estimators:
                base_clf = fit(X[base_subsample], y[base_subsample])
                meta_features.append(
                    scipy.sparse.csr_matrix(
                        predict(base_clf, X[meta_subsample]).reshape(meta_subsample.size, -1)
                    )
                )
            self.X_meta.append(scipy.sparse.hstack(meta_features))
            self.y_meta.extend(y[meta_subsample])

        self.X_meta = scipy.sparse.vstack(self.X_meta)

        self.base_classifiers = [(fit(X, y), predict)
                                    for (fit, predict) in self.base_estimators]
        if self.meta_fitter:
            self.fit_meta(self.meta_fitter)

        return self

    def fit_meta(self, meta_fitter):
        if not hasattr(self, 'X_meta'):
            raise Exception("Fit base classifiers first")
        self.meta_classifier = meta_fitter(self.X_meta, self.y_meta)
        return self

    def predict(self, X):      
        """Predict class for X.

        The predicted class of an input sample is computed as the majority
        prediction of the meta-classifiers.

        Input parameters:
            X : array-like or sparse matrix of shape = [n_samples, n_features]

        Output:
            y : array of shape = [n_samples] -- predicted classes
        """
        if not hasattr(self, 'meta_classifier'):
            raise Exception("Fit meta classifier first")

        X = scipy.sparse.csr_matrix(X)
        estimations_meta = [X] if self.extend_meta else []

        for base_clf, predict in self.base_classifiers:
            estimations_meta.append(
                scipy.sparse.csr_matrix(predict(base_clf, X).reshape(X.shape[0], -1))
            )

        return self.meta_classifier.predict(scipy.sparse.hstack(estimations_meta))
