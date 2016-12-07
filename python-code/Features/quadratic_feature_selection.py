# coding: utf-8
""" Created on 30 November 2016. Author: Roman Isachenko, Alexander Katrutsa """
import numpy as np
import cvxpy as cvx
from Features.generation_models import Feature


def quad_problem_pars(X, y, sim, rel):
    """
    Function generates matrix Q and vector b which represent feature similarities and feature relevances
    :param X: design matrix
    :type X: numpy.ndarray
    :param y: target vector
    :type y: numpy.ndarray
    :param sim: indicator of the way to compute feature similarities
    :type sim: str
    :param rel: indicator of the way to compute feature significance
    :type rel: str
    :return: matrix of features similarities Q, vector of feature relevances b
    :rtype:
    """
    if sim == 'correl':
        Q = np.corrcoef(X, rowvar=0)
    else:
        if sim == 'mi':
            Q = np.zeros([X.shape[1], X.shape[1]])
            for i in range(Q.shape[1]):
                for j in range(i, Q.shape[1]):
                    Q[i, j] = 1.0 #information(X[:, i], X[:, j])
            Q = Q + Q.T - np.diag(np.diag(Q))
        lambdas = np.linalg.eig(Q)
        min_lambda = min(lambdas)
        if min_lambda < 0:
            Q = Q - min_lambda * np.eye(Q.shape[0])
    if rel == 'correl':
        b = np.sum(corr2_coeff(X, y), axis=1) # FIXIT
        # b = np.zeros([X.shape[1], 1])
        # for i in range(X.shape[1]):
        #     b[i] = np.abs(pearsonr(X[:, i], y.flatten())[0])
    if rel == 'mi':
        b = np.zeros([X.shape[1], 1])
        for i in range(X.shape[1]):
            b[i] = 1.0 #information(y.T, X[:, i].T)

    return Q, b


def corr2_coeff(A, B):
    # Row-wise mean of input arrays & subtract from input arrays themselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA.T, B_mB) / np.sqrt(np.dot(ssA[None], ssB[:, None]))


class FeatureSelection(Feature):

    def __init__(self, name="Quadratic MIS", similarity='correl', relevance='correl', threshold=1e-6, on=True):
        self.name = str(name)
        self.similarity = similarity
        self.relevance = relevance
        self.threshold = threshold
        self.structure_vars = None
        self.constraints = []
        self.n_vars = 0
        self.status = None
        self.on = on

    def fit(self, X, Y):
        self.n_vars = X.shape[1]
        if not self.on:
            Feature.selected = range(self.n_vars)
            return self

        if Feature.selected is None:
            Feature.selected = range(self.n_vars)
        self.constraints = Feature.constraints
        self.variables = Feature.variables

        Q, b = quad_problem_pars(X, Y, self.similarity, self.relevance)

        x = cvx.Variable(self.n_vars)  # cvx.Int(n_vars) is infeasible

        objective = cvx.Minimize(cvx.quad_form(x, Q) - b.T * x)
        constraints = [x >= 0, x <= 1]
        constraints.extend(self.constraints)

        prob = cvx.Problem(objective, constraints)

        prob.solve(solver=cvx.ECOS_BB)

        self.structure_vars = np.ones(self.n_vars)
        self.status = prob.status
        if prob.status == "optimal":
            self.structure_vars = np.squeeze(np.array(x.value.flatten()))
            Feature.selected = np.nonzero(self.structure_vars < self.threshold)[0]

        print("Structure variable in [{}, {}], treshold {}"
              .format(self.structure_vars.min(), self.structure_vars.max(), self.threshold))

        return self

    def transform(self, X):
        selected_mask = np.zeros(X.shape[1])
        np.put(selected_mask, Feature.selected, 1.0)
        selected_mask = np.tile(selected_mask, (X.shape[0], 1))
        return X * selected_mask
