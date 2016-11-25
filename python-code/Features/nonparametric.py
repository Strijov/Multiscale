# coding: utf-8
""" Created on 23 November 2016. Author: Alexey Goncharov """
import numpy as np


def univariate_transformation(X):
    X1 = np.power(X, 0.5)
    X2 = np.power(X, 1.5)
    X3 = np.arctan(X)
    X4 = np.log(abs(X) + 1)
    X5 = np.multiply(np.log(abs(X) + 1), X)
    return np.hstack((X1, X2, X3, X4, X5))

def bivariate_transformation(X, y):
    X1 = X + y
    X2 = X - y
    X3 = X * y
    X4 = X / (np.sign(y)*(abs(y) + 1))
    X5 = X * np.sqrt(abs(y))
    X6 = X * np.log(abs(y) + 1)
    return np.hstack((X1, X2, X3, X4, X5, X6))

def simple_statistics(X, quantile_number = 10):
    X1 = np.mean(X, axis=1)[:, None]
    X2 = np.max(X, axis=1)[:, None]
    X3 = np.min(X, axis=1)[:, None]
    X4 = np.std(X, axis=1)[:, None]
    X5 = np.transpose(np.percentile(X, np.arange(0, 100, int(100.0 / quantile_number)), axis=1))
    X6 = np.mean(abs(X - X1), axis=1)[:, None]
    return np.hstack((X1, X2, X3, X4, X5, X6))


def make_len_even(X):
    n = X.shape[1]
    if n % 2 == 0:
        return np.copy(X)
    else:
        return np.copy(X[:, :-1])


def haar_iteration_avg(X):
    Y = make_len_even(X)
    haar_avg = np.zeros([Y.shape[0], Y.shape[1] / 2])
    for i in range(1, Y.shape[1] / 2 + 1):
        haar_avg[:, i - 1] = Y[:, (2 * i) - 1] + Y[:, (2 * i - 1) - 1]
    return haar_avg


def haar_iteration_dif(X):
    Y = make_len_even(X)
    haar_dif = np.zeros([Y.shape[0], Y.shape[1] / 2])
    for i in range(1, Y.shape[1] / 2 + 1):
        haar_dif[:, i - 1] = Y[:, (2 * i) - 1] - Y[:, (2 * i - 1) - 1]
    return haar_dif


def haar_transformations(X):
    X_iteration_avg = haar_iteration_avg(X)
    X_iteration_dif = haar_iteration_dif(X)
    X_haar_avg = X_iteration_avg
    X_haar_dif = X_iteration_dif
    while 1 > 0:
        if X_iteration_avg.shape[1] >= 2:
            X_iteration_avg = haar_iteration_avg(X_iteration_avg)
            X_iteration_dif = haar_iteration_dif(X_iteration_dif)
            X_haar_avg = np.hstack((X_haar_avg, haar_iteration_avg(X_haar_avg)))
            X_haar_dif = np.hstack((X_haar_dif, haar_iteration_dif(X_haar_dif)))
        else:
            break
    return np.hstack((X_haar_avg, X_haar_dif))