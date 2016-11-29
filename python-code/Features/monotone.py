# coding: utf-8
""" Created on 23 November 2016. Author: Alexey Goncharov """
from __future__ import print_function
import numpy as np


def monotone_linear(X, w=(.5, 1)):
    return X * w[0] + w[1]


def monotone_exponential_rate(X, w=(1, 0)):
    if w[0] > 0:
        return np.exp(X*w[0] + w[1])
    else:
        print('Error in monotone_exponential_rate: exp(a*X + b). Parameter "a" should be positive, got {}'.format(w[0]))
        raise ValueError


def monotone_polinomial_rate(X, w=(2, 1)):
    if w[0] > 1:
        return np.exp(np.log(X) * w[0] + w[1])
    else:
        print('Error in monotone_polinomial_rate: exp(a*logX + b). Parameter "a" should be greater than 1, got {}'
              .format(w[0]))
        raise ValueError


def monotone_sublinear_polinomial_rate(X, w=(.5, 1)):
    if 0 < w[0] < 1:
        return np.exp(np.log(X + 1) * w[0] + w[1])
    else:
        print('Error in monotone_sublinear_polinomial_rate: exp(a*logX + b). Parameter "a" should be in (0, 1) '
              'interval, got {}'.format(w[0]))
        raise ValueError


def monotone_logarithmic_rate(X, w=(.5, 1)):
    if 0 < w[0]:
        return np.log(X + 1) * w[0] + w[1]
    else:
        print('Error in monotone_logarithmic_rate: a*logX + b. Parameter "a" should be in (0, 1) interval, got {}'
              .format(w[0]))
        raise ValueError


def monotone_slow_convergence(X, w=(.5, 1)):
    if w[0] != 0 :
        return w[1] + float(w[0]) / (X+1)
    else:
        print('Error in monotone_slow_convergence: a/(X + 1) + b. Parameter "a" should be different from 0, got {}'
              .format(w[0]))
        raise ValueError


def monotone_fast_convergence(X, w=(.5, 1)):
    if w[0] != 0 :
        return w[1]+float(w[0])*np.exp(-X)
    else:
        print('Error in monotone_fast_convergence: a/exp(X) + b. Parameter "a" should be different from 0, got {}'
              .format(w[0]))
        raise ValueError


def monotone_soft_relu(X):
    return np.log(1 + np.exp(X))


def monotone_sigmoid(X, w=(.5, 1)):
    if w[0] > 0:
        return 1. / (w[1] + np.exp(-w[0] * X))
    else:
        print('Error in monotone_exponential_rate: 1/(exp(-a*X) + b). Parameter "a" should be positive, got {}'
              .format(w[0]))
        raise ValueError


def monotone_soft_max(X):
    return 1. / (1+np.exp(-X))


def monotone_hyberbolic_tangent(X):
    return(np.tanh(X))


def monotone_softsign(X):
    return(abs(X)/(1. + abs(X)))
