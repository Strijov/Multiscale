# coding: utf-8
""" Created on 23 November 2016. Author: Alexey Goncharov """
from __future__ import print_function
import numpy as np
import cvxpy as cvx


def monotone_linear(X, w=(.5, 1)):
    return X * w[0] + w[1]


def _set_monotone_linear():
    var = cvx.Variable(2)
    var.primal_value = np.random.randn(2)
    return var, []


def monotone_exponential_rate(X, w=(1, 0)):
    if w[0] > 0:
        return np.exp(X*w[0] + w[1])

    raise ValueError('Error in monotone_exponential_rate: exp(a*X + b). Parameter "a" should be positive, got {}'.format(w[0]))


def _set_monotone_exponential_rate():
    var = cvx.Variable(2)
    var.primal_value = np.array([.5, 0])
    constraints = [var[0] > 0]
    return var, constraints


def monotone_polinomial_rate(X, w=(2, 1)):
    if w[0] > 1:
        return np.exp(np.log(X + 1) * w[0] + w[1])

    raise ValueError('Error in monotone_polinomial_rate: exp(a*logX + b). Parameter "a" should be greater than 1, got {}'
                     .format(w[0]))


def _set_monotone_polinomial_rate():
    var = cvx.Variable(2)
    var.primal_value = np.array([2, 1])
    constraints = [var[0] > 0]
    return var, constraints


def monotone_sublinear_polinomial_rate(X, w=(.5, 1)):
    if 0 < w[0] < 1:
        return np.exp(np.log(X + 1) * w[0] + w[1])

    raise ValueError('Error in monotone_sublinear_polinomial_rate: exp(a*logX + b). Parameter "a" should be in (0, 1) '
                     'interval, got {}'.format(w[0]))



def _set_monotone_sublinear_polinomial_rate():
    var = cvx.Variable(2)
    var.primal_value = np.array([.5, 0])
    constraints = [var[0] > 0, var[0] < 1]
    return var, constraints


def monotone_logarithmic_rate(X, w=(.5, 1)):
    if 0 < w[0]:
        return np.log(X + 1) * w[0] + w[1]

    raise ValueError('Error in monotone_logarithmic_rate: a*logX + b. Parameter "a" should be positive, got {}'
                     .format(w[0]))



def _set_monotone_logarithmic_rate():
    var = cvx.Variable(2)
    var.primal_value = np.array([.5, 0])
    constraints = [var[0] > 0]
    return var, constraints


def monotone_slow_convergence(X, w=(.5, 1)):
    if w[0] > 0:
        return w[1] + float(w[0]) / (X + 1)

    raise ValueError('Error in monotone_slow_convergence: a/(X + 1) + b. Parameter "a" should be different from 0, got {}'
                     .format(w[0]))



def _set_monotone_slow_convergence():
    var = cvx.Variable(2)
    var.primal_value = np.array([.5, 0])
    return var, [var[0] > 0]


def monotone_fast_convergence(X, w=(.5, 1)):
    if w[0] > 0:
        return w[1] + float(w[0]) * np.exp(-X)
    print('Error in monotone_fast_convergence: a/exp(X) + b. Parameter "a" should be greater than 0, got {}'
          .format(w[0]))


def _set_monotone_fast_convergence():
    var = cvx.Variable(2)
    var.primal_value = np.array([.5, 0])
    return var, [var[0] > 0]


def monotone_soft_relu(X):
    return np.log(1 + np.exp(X))


def monotone_sigmoid(X, w=(.5, 1)):
    if w[0] > 0:
        return 1. / (w[1] + np.exp(-w[0] * X))
    raise ValueError('Error in monotone_exponential_rate: 1/(exp(-a*X) + b). Parameter "a" should be positive, got {}'
                     .format(w[0]))


def _set_monotone_sigmoid():
    var = cvx.Variable(2)
    var.primal_value = np.array([.5, 0])
    return var, [var[0] > 0]


def monotone_soft_max(X):
    return 1. / (1+np.exp(-X))


def monotone_hyberbolic_tangent(X):
    return np.tanh(X)


def monotone_softsign(X):
    return abs(X)/(1. + abs(X))
