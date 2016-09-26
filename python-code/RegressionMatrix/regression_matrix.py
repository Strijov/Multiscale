from __future__ import division
import pandas as pd
import numpy as np
from itertools import product
from collections import namedtuple

import my_plots

tsStruct = namedtuple('tsStruct', 'data request history name readme')
tsMiniStruct = namedtuple('tsMiniStruct', 's norm_div norm_subt name index')

class RegMatrix:
    "For all operations with regression matrices"

    def __init__(self, ts_struct):
        """ ts_structs_list contains namedtuples tsStruct with information for all time series, that will be forecasted
         simultaneously """
        #self.ts = ts_struct
        self.history = ts_struct.history
        self.request = ts_struct.request

        self.nts = len(ts_struct.data)
        self.ts = []
        self.forecasts = [0] * self.nts
        self.idxY = [0] * self.nts
        for ts in ts_struct.data:
            self.ts.append(tsMiniStruct(ts.as_matrix(), 1, 0, ts.name, np.array(ts.index)))


    def create_matrix(self, nsteps=1, norm_flag=True):
        # define matrix dimensions:
        
        self.n_hist_points = [0] * self.nts
        self.n_req_points = [0] * self.nts
        n_rows = [0] * self.nts
        # infer dimensions of X and Y
        for i, ts in enumerate(self.ts):
            self.n_req_points[i] = sum(ts.index < ts.index[0] + self.request)*nsteps # here we assume time stamps are uniform
            self.n_hist_points[i] = sum(ts.index < ts.index[0] + self.history)
            n_rows[i] = int(np.floor(len(ts.s) - self.n_hist_points[i]) / self.n_req_points[i])

        n_rows = min(n_rows)


        # prepare time series
        # standardize data:
        for i, ts in enumerate(self.ts):
            nnts = replace_nans(ts.s, ts.name)
            self.ts[i] = tsMiniStruct(nnts, ts.norm_div, ts.norm_subt, ts.name, ts.index)

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
            print "Inputs contain NaNs"

        if np.isnan(self.Y).any():
            print "Targets contain NaNs"

        if not check_time(timey, timex):
            print "Time check failed"



    def add_ts_to_matrix(self, i_ts, norm_flag):

        # ts are not overwritten, only normalization constants
        if norm_flag:
            ts, norm_div, norm_subt = normalize_ts(self.ts[i_ts].s, self.ts[i_ts].name)
            self.ts[i_ts] = tsMiniStruct(self.ts[i_ts].s, norm_div, norm_subt, self.ts[i_ts].name, self.ts[i_ts].index)
            ts = tsMiniStruct(ts, norm_div, norm_subt, self.ts[i_ts].name, self.ts[i_ts].index)
        else:
            ts = self.ts[i_ts]



        n_hist = self.n_hist_points[i_ts]
        n_req = self.n_req_points[i_ts]
        n_rows = self.X.shape[0]

        # reverse time series, so that the top row is always to be forecasted first
        time = np.flipud(ts.index)
        ts = np.flipud(ts.s)

        # flat_idx = self.matrix_indices(n_hist, n_req, n_rows)
        # matrix = np.fliplr(ts[flat_idx].reshape((n_rows, n_hist + n_req)))
        #
        # time = time[flat_idx].reshape((n_rows, n_hist + n_req))
        #
        # self.Y = np.hstack((self.Y, matrix[:, n_hist:]))
        # self.X = np.hstack((self.X, matrix[:, :n_hist]))

        idxX, idxY = self.matrix_idx(n_hist, n_req, n_rows)
        self.idxY[i_ts] = idxY
        self.Y = np.hstack((self.Y, ts[idxY]))
        self.X = np.hstack((self.X, ts[idxX]))

        return time[idxY[:, -1]], time[idxX[:, 0]]

    def matrix_indices(self, n_hist, n_req, n_rows) :

        flat_idx = []
        for i in xrange(n_rows):
            flat_idx.extend(range(i * n_req, (i + 1) * n_req + n_hist))
            # idx = np.unravel_index(flat_idx, (n_rows, n_hist + n_req))

        return flat_idx


    def matrix_idx(self, n_hist, n_req, n_rows) :

        flat_idx = []
        for i in xrange(n_rows):
            flat_idx.extend(range(i * n_req, (i + 1) * n_req + n_hist))
            # idx = np.unravel_index(flat_idx, (n_rows, n_hist + n_req))

        idx_matrix = np.reshape(flat_idx, (n_rows, n_hist + n_req))
        idxX = idx_matrix[:, n_req:]
        idxY = idx_matrix[:, :n_req]

        return idxX, idxY

    def generate_features(self, gnt):
        pass

    def select_features(self, slt):
        pass


    def arrange_time_scales(self, method):
        for i, ts in enumerate(self.ts):
            ts_arranged, index = method(ts.s, ts.index)
            self.ts[i] = tsMiniStruct(ts_arranged, ts.norm_div, ts.norm_subt, index)


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


    def train_model(self, model):
        model.fit(self.trainX, self.trainY)

        return model

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
            #_, idxY = self.matrix_idx(self.n_hist_points[i], self.n_req_points[i], self.X.shape[0])
            idx_flat[i] = ravel_idx(self.idxY[i][idx_rows, :], len(self.forecasts[i]))#self.n_hist_points[i], self.n_req_points[i], self.X.shape[0])
            # if not np.all(np.flipud(idx_flat[i]) == range(self.n_hist_points[i], len(self.forecasts[i]))):
            #     print "idx_flat =( "

        if not replace:
            return idx_flat

        for i in xrange(self.nts):
            self.forecasts[i][idx_flat[i]] = ravel_y(frc[:, :self.n_req_points[i]], self.ts[i].norm_div, self.ts[i].norm_subt)
            # if not np.all(self.forecasts[i][self.n_hist_points[i]:] == self.ts[i].s[self.n_hist_points[i]:]):
            #     print "wrong frc", np.nonzero(self.forecasts[0] != self.ts[i].s)[self.n_hist_points[i]:]
            frc = frc[:, self.n_req_points[i]:]

        return idx_flat



    def mae(self, idx_frc=None, idx_rows=None, out=None):
        idx = [0] * self.nts
        if idx_frc is None:
            if idx_rows is None:
                for i in range(self.nts):
                    idx[i] = range(self.n_hist_points[i], len(self.forecasts[i]))

            else:
                for i in range(self.nts):
                    idx[i] = ravel_idx(self.idxY[i][idx_rows, :], len(self.forecasts[i]))
        else:
            idx = idx_frc


        errors = np.zeros((self.nts))
        for i, ind in enumerate(idx):
            ts = self.ts[i].s
            errors[i] = np.mean(np.abs(ts[ind] - self.forecasts[i][ind]))

        if not out is None:
            print out, "MAE"
            for i, err in enumerate(errors):
                print self.ts[i].name, err

        return errors

    def mape(self, idx_frc=None, idx_rows=None, out=None):
        idx = [0] * self.nts
        if idx_frc is None:
            if idx_rows is None:
                for i in range(self.nts):
                    idx[i] = range(self.n_hist_points[i], len(self.forecasts[i]))

            else:
                for i in range(self.nts):
                    idx[i] = ravel_idx(self.idxY[i][idx_rows], len(self.forecasts[i]))
        else:
            idx = idx_frc

        errors = np.zeros((self.nts))
        for i, ind in enumerate(idx):
            denom = np.abs(self.ts[i].s[ind])
            denom[denom == 0] = np.mean(np.abs(self.ts[i].s))
            errors[i] = np.mean(np.divide(np.abs(self.ts[i].s[ind] - self.forecasts[i][ind]), denom))

        if not out is None:
            print out, "MAPE"
            for i, err in enumerate(errors):
                print self.ts[i].name, err

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
                    idx[i] = ravel_idx(self.idxY[i][idx_rows], len(self.forecasts[i]))
        else:
            idx = idx_frc

        for i, ts in enumerate(self.ts):
            my_plots.plot_forecast(ts, self.forecasts[i], idx_frc=idx[i], idx_ts=idx_ts[i])

def normalize_ts(ts, name=None):

    norm_subt = np.min(ts)
    norm_div = np.max(ts) - norm_subt
    if norm_div == 0:
        print "Time series", name, "is constant"
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
    ts_struct = tsMiniStruct(ts, ts_struct.norm_div, ts_struct.norm_subt, ts_struct.name, ts_struct.index[-n_points:])
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

    print "Filling NaNs for TS", name
    if np.isnan(ts).all():
        print "All inputs are NaN", "replacing with zeros"
        ts = np.zeros_like(ts)
        return ts

    ts_prop = pd.Series(ts).fillna(method="pad")
    ts_back = pd.Series(ts_prop).fillna(method="bfill")
    ts =  ts_back.as_matrix() # (ts_back + ts_prop)[pd.isnull(ts)] / 2

    return ts


def ravel_idx(idx_matr, m):
    return m - 1 - np.ravel(idx_matr)#np.flipud(np.ravel(idx_matr))


def ravel_y(Ymat, norm_div, norm_subt):
    y = np.ravel(Ymat) * norm_div + norm_subt
    return y #np.flipud(y)