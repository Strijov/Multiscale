# coding: utf-8
""" Created on 23 November 2016. Author: Alexey Goncharov """
import numpy as np
import cvxpy as cvx


def parametric_sum(x, w):
    return x + w 


def _set_parametric_sum():
    var = cvx.Variable()
    var.primal_value = np.random.randn()
    return var, []


def parametrc_quadratic(x, w):
    return np.multiply(np.power(x, 2), w[0]) + np.multiply(x, w[1]) + w[2]


def _set_parametrc_quadratic():
    var = cvx.Variable(3)
    var.primal_value = np.random.randn(3)
    return var, []


def parametric_qubic(x, w):
    return np.power(x, 3) * w[0] + np.power(x, 2) * w[1] + x * w[2] + w[3]


def _set_parametric_qubic():
    var = cvx.Variable(4)
    var.primal_value = np.random.randn(4)
    return var, []


def parametric_logarithmic_sigmoid(x, w):
    return 1. / (w[0] + np.exp(x * w[1]))


def _set_logarithmic_sigmoid():
    var = cvx.Variable(2)
    var.primal_value = np.random.randn(2)
    return var, []

# def parametric_exponent(x):
#     return(np.exp(x))
#


def parametric_normal(x, w):
    return 1. / (w[0]*np.sqrt(2*np.pi))*np.exp(-np.power(x-w[1], 2)/w[0]**2)


def _set_parametric_normal():
    var = cvx.Variable(2)
    var.primal_value = np.random.randn(2)
    return var, []


def parametric_multiply(x, w):
    return np.multiply(x, w)
    
    
def _set_parametric_multiply():
    var = cvx.Variable(1)
    var.primal_value = np.random.randn(1)
    return var, []


def parametric_monomial(x, w):
    return w[0] * np.power(x, w[1])


def _set_parametric_monomial():
    var = cvx.Variable(2)
    var.primal_value = np.random.randn(2)
    return var, []


def parametric_weibul_2(x, w):
    return w[0]*w[1]*np.power(x, w[1]-1) * np.exp(-w[0]*np.power(x, w[1]))


def _set_parametric_weibul_2():
        var = cvx.Variable(2)
        var.primal_value = np.random.randn(2)
        return var, []
    
    
def parametric_weibul_3(x, w):
    return w[0]*w[1]*np.power(x, w[1]-1)*np.exp(-w[0]*np.power(x-w[2], w[1]))


def _set_parametric_weibul_3():
    var = cvx.Variable(3)
    var.primal_value = np.random.randn(3)
    return var, []

# def parametric_transform(x,w_sum,w_quadratic,w_qubic,w_sigmoid,w_normal,w_multiply,w_monomial,w_weibul2,w_weibul3):
#     return(np.hstack((parametric_sum(x,w_sum),parametrc_quadratic(x,w_quadratic),parametrc_qubic(x,w_qubic),
#                     parametric_logarithmic_sigmoid(x,w_sigmoid), parametric_exponent(x),
#                       parametric_normal(x,w_normal),parametric_multiply(x,w_multiply),parametric_monomial(x,w_monomial),
#                      parametric_weibul_2(x,w_weibul2),parametric_weibul_3(x,w_weibul3))))


# def build_henkel_matrix(x, k, p):
#     x = np.array(x)
#     ts_len = len(x)
#     henkel = [x[ts_len-1:ts_len-k-1:-1]]
#     ts_len = ts_len - p
#     while ts_len >= k + 1:
#         henkel.append(x[ts_len-1:ts_len-k-1:-1])
#         ts_len = ts_len - p
#     return np.matrix(henkel)
#
# def henkel_coefs(x, k, p):
#     matrix = build_henkel_matrix(x, k, p)
#     Y = matrix[:, 0]
#     Z = matrix[:, 1:]
#     return np.array(np.transpose(np.linalg.lstsq(Z,Y)[0]))[0]
#
#
# def SSA(x, k, p):
#     return np.apply_along_axis(henkel_coefs,1,x,k,p)
