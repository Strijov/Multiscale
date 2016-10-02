"""
Created on 30 September 2016
@author: Anastasia Motrenko
"""
from __future__ import division
from __future__ import print_function

import copy
import pandas as pd
import numpy as np
import sklearn.pipeline as pipeline

from sklearn.utils.validation import check_is_fitted
from itertools import product
from collections import namedtuple

import my_plots
from Forecasting import frc_class

TsMiniStruct_ = namedtuple('TsMiniStruct', 's norm_div norm_subt name index')
class TsMiniStruct(TsMiniStruct_):
    """ This structure stores a particular time series. The fields are:

    :param s: time series
    :type s: 1d-ndarray
    :param norm_div: Standartisation constant
    :type norm_div: float
    :param norm_subt: Standartisation constant
    :type norm_subt: float
    :param name: Dataset name
    :type name: string
    :param index: time ticks
    :type index: 1d-ndarray
    """
    pass

class RegMatrix:
    """The main class for ts-to-matrix, matrix-to-ts conversions and other data operations. """

    def __init__(self, ts_struct):
        """

        :param ts_struct: input time series
        :type ts_struct: named tuple TsStruct with fields, data, request, history, name, readme
        """

        self.request = ts_struct.request
        self.history = ts_struct.history
        if self.history is None:
            self.history = ts_struct.request
            print("History is not defined.  Do not forget to optimize it!") # FIXIT

        self.nts = len(ts_struct.data)
        self.ts = []

        self.forecasts = [0] * self.nts
        self.idxY = [0] * self.nts
        names = []
        for ts in ts_struct.data:
            # print("nans:", ts.name, np.sum(np.isnan(ts)))
            self.ts.append(TsMiniStruct(ts.as_matrix(), 1, 0, ts.name, np.array(ts.index)))
            names.append(ts.name)
        self.feature_dict = dict.fromkeys(names)

    def create_matrix(self, nsteps=1, norm_flag=True):
        """
        Turn the input set of time series into regression matrix.

        :param nsteps: Number of times request is repeated in Y
        :type nsteps: int
        :param norm_flag: if False, time series are processed without normalisation
        :type norm_flag: bool
        :return: None. Updates attributes self.X, self.Y, self.n_requested_points, self.n_historical_points, self.feature_dict
        """
        # define matrix dimensions:
        nsteps = int(nsteps)
        if nsteps < 1:
            print("nsteps should be at least 1. Setting nsteps = 1")


        self.n_hist_points = [0] * self.nts
        self.n_req_points = [0] * self.nts
        n_rows = [0] * self.nts
        hist = [0]
        # infer dimensions of X and Y
        for i, ts in enumerate(self.ts):
            self.n_req_points[i] = sum(ts.index < ts.index[0] + self.request)*nsteps # here we assume time stamps are uniform
            self.n_hist_points[i] = sum(ts.index < ts.index[0] + self.history)
            n_rows[i] = int(np.floor(len(ts.s) - self.n_hist_points[i]) / self.n_req_points[i])
            hist.append(hist[i] + self.n_hist_points[i])
            self.feature_dict[ts.name] = range(hist[i], hist[i+1])



        n_rows = min(n_rows)
        if n_rows < 4:
            print("Number of rows is ", n_rows, "consider setting a lower value of nsteps or requested points")


        # prepare time series
        # standardize data:
        for i, ts in enumerate(self.ts):
            nnts = replace_nans(ts.s, ts.name)
            self.ts[i] = TsMiniStruct(nnts, ts.norm_div, ts.norm_subt, ts.name, ts.index)

        # init matrices
        self.X = np.zeros((n_rows, 0))
        self.Y = np.zeros((n_rows, 0))
        timex, timey = [0]*self.nts, [0]*self.nts
        for i, ts in enumerate(self.ts):
            # truncate time series
            self.ts[i] = truncate(ts, self.n_hist_points[i], self.n_req_points[i], n_rows)
            self.forecasts[i] = np.zeros_like(self.ts[i].s)#pd.Series(np.zeros_like(self.ts[i].data), index=self.ts[i].data.index, name = self.ts[i].data.name)
            timey[i], timex[i] = self.add_ts_to_matrix(i, norm_flag)


        if np.isnan(self.X).any():
            print("Inputs contain NaNs")

        if np.isnan(self.Y).any():
            print("Targets contain NaNs")

        if not check_time(timey, timex):
            print("Time check failed")



    def add_ts_to_matrix(self, i_ts, norm_flag):
        """
        Adds time series to data matrix

        :param i_ts: Number of ts to add
        :param norm_flag: normalisation flag. Default=True, if False, time series are processed without normalisation
        """

        # ts are not overwritten, only normalization constants
        if norm_flag:
            ts, norm_div, norm_subt = normalize_ts(self.ts[i_ts].s, self.ts[i_ts].name)
            self.ts[i_ts] = TsMiniStruct(self.ts[i_ts].s, norm_div, norm_subt, self.ts[i_ts].name, self.ts[i_ts].index)
            ts = TsMiniStruct(ts, norm_div, norm_subt, self.ts[i_ts].name, self.ts[i_ts].index)
        else:
            ts = self.ts[i_ts]



        n_hist = self.n_hist_points[i_ts]
        n_req = self.n_req_points[i_ts]
        n_rows = self.X.shape[0]

        # reverse time series, so that the top row is always to be forecasted first
        time = np.flipud(ts.index)
        ts = np.flipud(ts.s)

        idxX, idxY = matrix_idx(n_hist, n_req, n_rows)
        self.idxY[i_ts] = idxY
        self.Y = np.hstack((self.Y, ts[idxY]))
        self.X = np.hstack((self.X, ts[idxX]))

        return time[idxY[:, -1]], time[idxX[:, 0]]

    # def matrix_indices(self, n_hist, n_req, n_rows) :
    #
    #     flat_idx = []
    #     for i in xrange(n_rows):
    #         flat_idx.extend(range(i * n_req, (i + 1) * n_req + n_hist))
    #         # idx = np.unravel_index(flat_idx, (n_rows, n_hist + n_req))
    #     return flat_idx


    def _matrix_to_flat_by_ts(self, idx_rows, i_ts):
        idx = _ravel_idx(self.idxY[i_ts][idx_rows, :], len(self.forecasts[i_ts]))
        return np.flipud(idx)

    def matrix_to_flat(self, idx_rows):
        """
        Returns indices of TS entries stored in specific rows of regression matrix self.Y

        :param idx_rows: rows of self.Y matrix
        :type idx_rows: list
        :return: self.nts lists of ts indices. Each list corresponds to one of the input time series
        :rtype: list
        """
        idx = []
        for i in range(self.nts):
            idx.append(self._matrix_to_flat_by_ts(idx_rows, i))
        return idx


    def arrange_time_scales(self, method):
        for i, ts in enumerate(self.ts):
            ts_arranged, index = method(ts.s, ts.index)
            self.ts[i] = TsMiniStruct(ts_arranged, ts.norm_div, ts.norm_subt, index)


    def train_test_split(self, train_test_ratio=0.75, splitter=None):
        if not splitter is None:
            idx_test, idx_train = splitter()
        else: # sequesntial split
            n_train= int(self.X.shape[0]*train_test_ratio)
            idx_test = range(self.X.shape[0] - n_train)
            idx_train = range(self.X.shape[0] - n_train, self.X.shape[0])
        self.trainX, self.trainY = self.X[idx_train, :], self.Y[idx_train]
        self.testX, self.testY = self.X[idx_test, :], self.Y[idx_test]
        self.idx_train, self.idx_test = idx_train, idx_test


    def train_model(self, frc_model, selector=None, generator=None, retrain=True):

        if selector is None:
            selector = frc_class.IdentityModel()
            selector.feature_dict = copy.deepcopy(self.feature_dict)
        if generator is None:
            generator = frc_class.IdentityGenerator()
            generator.feature_dict = copy.deepcopy(self.feature_dict)

        model = pipeline.Pipeline([('gen', generator), ('sel', selector), ('frc', frc_model)])
        #model = pipeline.make_pipeline(generator, selector, frc_model)

        if (not frc_model.is_fitted) or retrain:
                model.fit(self.trainX, self.trainY)

        return model, model.named_steps['frc'], model.named_steps['gen'], model.named_steps['sel']

    def forecast(self, model, idx_rows=None, replace=True):
        # idx are indices of rows to be forecasted
        if idx_rows is None:
            idx_rows = range(self.X.shape[0])

        forecastedY =  model.predict(self.X[idx_rows, :])

        # ravel forecasts and, if replace=True, input new values to self.forecasts vector
        idx_frc = self.add_forecasts(forecastedY, idx_rows, replace)

        return forecastedY, idx_frc

    def add_forecasts(self, frc, idx_rows, replace):

        idx_flat = [0] * self.nts
        for i in xrange(self.nts):
            #_, idxY = matrix_idx(self.n_hist_points[i], self.n_req_points[i], self.X.shape[0])
            idx_flat[i] = _ravel_idx(self.idxY[i][idx_rows, :], len(self.forecasts[i]))#self.n_hist_points[i], self.n_req_points[i], self.X.shape[0])
            # if not np.all(np.flipud(idx_flat[i]) == range(self.n_hist_points[i], len(self.forecasts[i]))):
            #     print("idx_flat =( ")

        if not replace:
            return idx_flat

        for i in xrange(self.nts):
            self.forecasts[i][idx_flat[i]] = _ravel_y(frc[:, :self.n_req_points[i]], self.ts[i].norm_div, self.ts[i].norm_subt)
            # if not np.all(self.forecasts[i][self.n_hist_points[i]:] == self.ts[i].s[self.n_hist_points[i]:]):
            #     print("wrong frc", np.nonzero(self.forecasts[0] != self.ts[i].s)[self.n_hist_points[i]:])
            frc = frc[:, self.n_req_points[i]:]

        return idx_flat



    def mae(self, idx_frc=None, idx_rows=None, out=None):
        """
        Mean Absolute Error calculation.

        :param idx_frc: Indices of forecasted/target TS entries used to compute MAE
        :type idx_frc: list
        :param idx_rows: Alternatively, specify raw indices for matrix Y. If idx_frc is specified, idx_rows is ignored
        :type idx_rows: list
        :param out: Specification of out invokes printing MAE values. out string is printed before the output
        :type out: string
        :return: list of MAPE values, one for each input time series
        :rtype: list
        """
        idx = [0] * self.nts
        if idx_frc is None:
            if idx_rows is None:
                for i in range(self.nts):
                    idx[i] = range(self.n_hist_points[i], len(self.forecasts[i]))

            else:
                for i in range(self.nts):
                    idx[i] = _ravel_idx(self.idxY[i][idx_rows, :], len(self.forecasts[i]))
        else:
            idx = idx_frc


        errors = np.zeros((self.nts))
        for i, ind in enumerate(idx):
            ts = self.ts[i].s
            errors[i] = np.mean(np.abs(ts[ind] - self.forecasts[i][ind]))

        if not out is None:
            print(out, "MAE")
            for i, err in enumerate(errors):
                print(self.ts[i].name, err)

        return errors

    def mape(self, idx_frc=None, idx_rows=None, out=None):
        # type: (list, list, string) -> list
        """
        Mean Absolute Percentage Error calculation.

        :param idx_frc: Indices of forecasted/target TS entries used to compute MAE
        :type idx_frc: list
        :param idx_rows: Alternatively, specify raw indices for matrix Y. If idx_frc is specified, idx_rows is ignored
        :type idx_rows: list
        :param out: Specification of out invokes printing MAPE values. out string is printed before the output
        :type out: string
        :return: list of MAPE values, one for each input time series
        :rtype: list
        """
        idx = [0] * self.nts
        if idx_frc is None:
            if idx_rows is None:
                for i in range(self.nts):
                    idx[i] = range(self.n_hist_points[i], len(self.forecasts[i]))

            else:
                for i in range(self.nts):
                    idx[i] = _ravel_idx(self.idxY[i][idx_rows], len(self.forecasts[i]))
        else:
            idx = idx_frc

        errors = np.zeros((self.nts))
        for i, ind in enumerate(idx):
            denom = np.abs(self.ts[i].s[ind])
            denom[denom == 0] = np.mean(np.abs(self.ts[i].s))
            errors[i] = np.mean(np.divide(np.abs(self.ts[i].s[ind] - self.forecasts[i][ind]), denom))

        if not out is None:
            print(out, "MAPE")
            for i, err in enumerate(errors):
                print(self.ts[i].name, err)

        return errors

    def plot_frc(self, idx_frc=None, idx_rows=None, n_frc = 1, n_hist=3):
        idx = [0] * self.nts
        idx_ts = [0] * self.nts
        if idx_frc is None:
            if idx_rows is None:
                for i in range(self.nts):
                    idx[i] = range(len(self.forecasts[i]) - self.n_req_points[i]*n_frc, len(self.forecasts[i]))
                    idx_ts[i] = range(len(self.forecasts[i]) - self.n_req_points[i]*(n_frc + n_hist), len(self.forecasts[i]))

            else:
                for i in range(self.nts):
                    idx[i] = _ravel_idx(self.idxY[i][idx_rows], len(self.forecasts[i]))
        else:
            idx = idx_frc

        for i, ts in enumerate(self.ts):
            my_plots.plot_forecast(ts, self.forecasts[i], idx_frc=idx[i], idx_ts=idx_ts[i])

def normalize_ts(ts, name=None):

    norm_subt = np.min(ts)
    norm_div = np.max(ts) - norm_subt
    if norm_div == 0:
        print("Time series", name, "is constant")
        norm_div = 1
        norm_subt = 0
    else:
        ts = (ts - norm_subt) / norm_div

    return ts, norm_div, norm_subt


def denormalize(ts_list):
    for i, ts in enumerate(ts_list):
        if not ts.norm_div == 0:
            ts_list[i].s = (ts + ts.norm_subt) * ts.norm_div
    return ts_list

def truncate(ts_struct, n_hist, n_req, n_rows):
    ts = ts_struct.s
    n_points = n_hist + n_req*n_rows

    ts = ts[-n_points:]
    ts_struct = TsMiniStruct(ts, ts_struct.norm_div, ts_struct.norm_subt, ts_struct.name, ts_struct.index[-n_points:])
    return ts_struct

def check_time(y, x):
    check = []
    # y stores earliest time for Y in each row, x stores latest time for X in each row. These two must not overlap
    for ty, tx in product(y, x):
        check.append(np.all(ty > tx))


    return np.all(check)


def replace_nans(ts, name=None):
    #
    if not np.isnan(ts).any():
        return ts

    print("Filling NaNs for TS", name)
    if np.isnan(ts).all():
        print("All inputs are NaN", "replacing with zeros")
        ts = np.zeros_like(ts)
        return ts

    ts_prop = pd.Series(ts).fillna(method="pad")
    ts_back = pd.Series(ts_prop).fillna(method="bfill")
    ts =  ts_back.as_matrix() # (ts_back + ts_prop)[pd.isnull(ts)] / 2

    return ts


def matrix_idx(n_hist, n_req, n_rows):
    """
    Returns indices of ts entries in matrices X and Y given forecast parameters, by rows. Ts are enumerated from the latest entry

    :param n_hist: (X) row length
    :type n_hist: int
    :param n_req: (Y) row length
    :type n_req: int
    :param n_rows: number of rows
    :type n_rows: int
    :return: matrices idxX with shape (n_rows, n_hist) and idxY with shape (n_rows, n_req)
    >>> idxX, idxY = matrix_idx(10, 2, 4)
    >>> print idxX
    [[ 2  3  4  5  6  7  8  9 10 11]
    [ 4  5  6  7  8  9 10 11 12 13]
    [ 6  7  8  9 10 11 12 13 14 15]
    [ 8  9 10 11 12 13 14 15 16 17]]
    >>> print idxY
    [[0 1]
     [2 3]
     [4 5]
     [6 7]]
    """

    flat_idx = []
    for i in xrange(n_rows):
        flat_idx.extend(range(i * n_req, (i + 1) * n_req + n_hist))
        # idx = np.unravel_index(flat_idx, (n_rows, n_hist + n_req))

    idx_matrix = np.reshape(flat_idx, (n_rows, n_hist + n_req))
    idxX = idx_matrix[:, n_req:]
    idxY = idx_matrix[:, :n_req]

    return idxX, idxY


def _ravel_idx(idx_matr, m):
    return m - 1 - np.ravel(idx_matr)#np.flipud(np.ravel(idx_matr))


def _ravel_y(Ymat, norm_div, norm_subt):
    y = np.ravel(Ymat) * norm_div + norm_subt
    return y #np.flipud(y)