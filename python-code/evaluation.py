# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:36:02 2016

@author: yagmur

Evaluation measures for the time series prediction

"""

def mean_squared_error(y_pred, y_true):
    """ the mean squared error"""
    mse = np.mean((y_true - y_pred) **2)
    return mse


def mean_absolute_error(y_pred, y_true):
    """ the mean squared error"""
    mae = np.mean(np.abs((y_true - y_pred)))
    return mae
    

def mean_abs_percent_err(y_pred, y_true):
    """ the mean absolute percentage error"""
    idx = y_true != 0.0
    mape = np.mean(np.abs((y_pred[idx]-y_true[idx])/y_true[idx])) * 100
    return mape

