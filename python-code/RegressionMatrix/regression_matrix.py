"""
Created on 30 September 2016
@author: Anastasia Motrenko
"""
from __future__ import division
from __future__ import print_function

import copy
import warnings
import pandas as pd
import numpy as np

from itertools import product
from collections import namedtuple

import my_plots


TsMiniStruct_ = namedtuple('TsMiniStruct', 's norm_div norm_subt name index')
class TsMiniStruct(TsMiniStruct_):
    """ This structure stores a particular time series. The fields are:

    :param s: time series
    :type s: 1d-ndarray
    :param norm_div: Standardisation constant
    :type norm_div: float
    :param norm_subt: Standardisation constant
    :type norm_subt: float
    :param name: Dataset name
    :type name: string
    :param index: time ticks
    :type index: 1d-ndarray
    """
    pass

class RegMatrix:
    """The main class for ts-to-matrix, matrix-to-ts conversions and other data operations. """

    def __init__(self, ts_struct, x_idx=None, y_idx=None, n_historical=2):
        """

        :param ts_struct: input time series
        :type ts_struct: named tuple TsStruct with fields, data, request, history, name, readme
        :param x_idx: indices of time series if ts_struct.data to be used in X matrix
        :type x_idx: list, int, None
        :param y_idx: indices of time series if ts_struct.data to be used in Y matrix
        :type y_idx: list, int, None
        :param n_historical: default number of requested intervals used as history if this parameter is not defined in ts_struct
        :type n_historical: int
        """
        # chech that time indices are all in floats
        ts_struct.to_floats()

        # check that data field contains a list:
        if not isinstance(ts_struct.data, list):
            ts_struct_data = [ts_struct.data]
        else:
            ts_struct_data = ts_struct.data

        self.nts = len(ts_struct_data)
        self.ts = []
        names = []  # used in feature_dict

        for ts in ts_struct_data:
            # time series indices are converted to floats
            # This part should probably go in TsStruct code? #FIXIT
            #ts_index, frequency = pd_time_stamps_to_floats(ts.index)
            self.ts.append(TsMiniStruct(ts.as_matrix(), 1, 0, ts.name, ts.index))
            names.append(ts.name)
        self.feature_dict = dict.fromkeys(names)

        # if isinstance(ts_struct.request, pd.tslib.Timedelta):
        #     self.request = ts_struct.request
        # elif isinstance(ts_struct.one_step, pd.tslib.Timedelta):
        #     self.request = multiply_pd_time_delta(ts_struct.one_step.days, ts_struct.request)
        # else:
        #     self.request = ts_struct.request * ts_struct.one_step
        # self.request = general_time_delta_to_float(self.request, frequency)

        self.request = ts_struct.request * ts_struct.one_step

        # if not isinstance(ts_struct.one_step, pd.tslib.Timedelta):
        #     self.history = ts_struct.history * ts_struct.one_step
            
        if ts_struct.history is None:
            n_historical = n_historical * ts_struct.request
            print("History is not defined.  Do not forget to optimize it!")  # FIXIT
        else:
            n_historical = ts_struct.history

        self.history = n_historical * ts_struct.one_step
        # if isinstance(ts_struct.history, pd.tslib.Timedelta):
        #     self.history = ts_struct.request
        # elif isinstance(self.history, pd.tslib.Timedelta):
        #     self.history = multiply_pd_time_delta(ts_struct.one_step.days, n_historical)
        # else:
        #     self.history = n_historical * ts_struct.request * ts_struct.one_step
        # self.history = general_time_delta_to_float(self.history, frequency)

        # check arguments
        self.x_idx = _check_input_ts_idx(x_idx, range(self.nts))
        self.y_idx = _check_input_ts_idx(y_idx, range(self.nts))

        self.forecasts = [0] * self.nts
        self.idxY = [0] * self.nts

    def create_matrix(self, nsteps=1, norm_flag=True, x_idx=None, y_idx=None):
        """
        Turn the input set of time series into regression matrix.

        :param nsteps: Number of times request is repeated in Y
        :type nsteps: int
        :param norm_flag: if False, time series are processed without normalisation
        :type norm_flag: bool
        :param x_idx: indices of time series if ts_struct.data to be used in X matrix
        :type x_idx: list, int, None
        :param y_idx: indices of time series if ts_struct.data to be used in Y matrix
        :type y_idx: list, int, None
        :return: None. Updates attributes self.X, self.Y, self.n_requested_points, self.n_historical_points, self.feature_dict
        """
        # define matrix dimensions:
        nsteps = int(nsteps)

        if nsteps < 1:
            print("Parameter 'nsteps' should be at least 1. Setting nsteps = 1")
            nsteps = 1

        self.x_idx = _check_input_ts_idx(x_idx, self.y_idx)
        self.y_idx = _check_input_ts_idx(y_idx, self.y_idx)

        self.n_hist_points = [0] * self.nts
        self.n_req_points = [0] * self.nts
        n_rows = [0] * self.nts
        hist = [0]
        # infer dimensions of X and Y
        for i, ts in enumerate(self.ts):
            if i in self.x_idx:
                self.n_hist_points[i] = sum(ts.index < (ts.index[0] + self.history))
            n_req_points = sum(ts.index < (ts.index[0] + self.request) )*nsteps

            if i in self.y_idx:
                self.n_req_points[i] = n_req_points # here we assume time stamps are uniform

            if self.n_req_points[i] >= ts.s.shape[0]:
                print("Request: {}".format(self.request))
                raise ValueError("The length of time series {0} is smaller than the number of requested points: {1} <= {2}"
                                 .format(ts.name, ts.s.shape[0], self.n_req_points[i]))
            if self.n_hist_points[i] >= ts.s.shape[0]:
                print("History: {}".format(self.history))
                raise ValueError("The length of time series {0} is smaller than the number of historical points: {1} <= {2}"
                                 .format(ts.name, ts.s.shape[0], self.n_hist_points[i]))

            n_rows[i] = int(np.floor(len(ts.s) - self.n_hist_points[i]) / n_req_points)
            hist.append(hist[i] + self.n_hist_points[i])
            if i in self.x_idx:
                self.feature_dict[ts.name] = range(hist[i], hist[i+1])

        idx_included = set(self.x_idx + self.y_idx)
        n_rows = min([n_rows[i] for i in idx_included])

        if n_rows < 5:
            print("Number of rows is {} consider setting a lower value of nsteps or requested points,"
                  "(currently n_req = {}, n_hist={})".format(n_rows, self.n_req_points, self.n_hist_points))

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
            add_x = i in self.x_idx
            add_y = i in self.y_idx
            if add_x or add_y:
                timey[i], timex[i] = self.add_ts_to_matrix(i, norm_flag, add_x=add_x, add_y=add_y)

        if np.isnan(self.X).any():
            print("Inputs contain NaNs")

        if np.isnan(self.Y).any():
            print("Targets contain NaNs")

        if not check_time([timey[i] for i in self.y_idx], [timex[i] for i in self.x_idx]):
            raise ValueError("Time check failed timey: {}, timex {}".format(timey, timex))

    def add_ts_to_matrix(self, i_ts, norm_flag, add_x=True, add_y=True):
        """
        Adds time series to data matrix

        :param i_ts: Index of time series to add
        :type i_ts: int
        :param norm_flag: normalisation flag. Default=True, if False, time series are processed without normalisation
        :type norm_flag: bool
        :param add_x: Flag that indicates whether the time series should be added to X matrix
        :type add_x: bool
        :param add_y: Flag that indicates whether the time series should be added to Y matrix
        :type add_y: bool
        :return: None
        """

        # ts are not overwritten, only normalization constants
        if norm_flag:
            ts, norm_div, norm_subt = _normalize_ts(self.ts[i_ts].s, self.ts[i_ts].name)
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
        if add_y:
            self.Y = np.hstack((self.Y, ts[idxY]))
        if add_x:
            self.X = np.hstack((self.X, ts[idxX]))

        return time[idxY[:, -1]], time[idxX[:, 0]]

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

    def train_test_split(self, train_test_ratio=0.75, splitter=None, idx_train=None, idx_test=None):
        """
        Splits row indices of data matrix into train and test.

        :param train_test_ratio: Train to all ratio
        :type train_test_ratio: float
        :param splitter: custom function. By default bottom m*train_test_ratio rows are used for training
        :type splitter: callable
        :return: Updates attributes: idx_train, idx_test, testX, testY, trainX, trainY
        :rtype: None
        """
        if splitter is not None:
            idx_test, idx_train = splitter()
        elif idx_train is not None:
            idx_train = np.sort(idx_train)
        elif idx_test is not None:
            idx_test = np.sort(idx_test)
        else:   # sequential split
            n_train= int(self.X.shape[0]*train_test_ratio)
            idx_test = range(self.X.shape[0] - n_train)
            idx_train = range(self.X.shape[0] - n_train, self.X.shape[0])

        self.trainX, self.trainY = self.X[idx_train, :], self.Y[idx_train]
        self.testX, self.testY = self.X[idx_test, :], self.Y[idx_test]
        self.idx_train, self.idx_test = idx_train, idx_test

    def train_model(self, frc_model, selector=None, generator=None, retrain=True, hyperpars=None, n_cvs=5):
        """
        Initializes and train feature generation, selection and forecasting model in a pipeline

        :param frc_model: Instance of CustomModel()
        :param selector: Instance of model selection class
        :param generator: Instance of model generation class
        :param retrain: Flag that specifies if the model needs retraining
        :type retrain: bool
        :param hyperpars: defines ranges of hyperparameters to tune with cross-validation. If None is specified, 
        no hyperparameters will be optimized 
        :type hyperpars: dict
        :param n_cvs: number of folds in k-fold cross-validation
        :type n_cvs: int
        :return: trained model, forecasting model, generator and selector
        :rtype: tuple
        """

        from Forecasting import frc_class
        if selector is None:
            selector = frc_class.IdentityModel()

        if generator is None:
            generator = frc_class.IdentityGenerator()

        selector.feature_dict = copy.deepcopy(self.feature_dict)
        generator.feature_dict = copy.deepcopy(self.feature_dict)

        # create pipeline with named steps
        model = frc_class.PipelineModel([('gen', generator), ('sel', selector), ('frc', frc_model)]) # pipeline.Pipeline([('gen', generator), ('sel', selector), ('frc', frc_model)])
        model.name = "_".join([str(frc_model.name), str(generator.name), str(selector.name)])

        # once fitted, the model is retrained only if retrain = True
        if frc_model.is_fitted and not retrain:
            return model, model.named_steps['frc'],  model.named_steps['gen'], model.named_steps['sel']

        # if a range of hyperparametr values is specify, tune it via k-fold cross-validation
        if hyperpars is not None:
            best_hyperpars = cv_train(model, self, hyperpars, n_cvs)
            for k, v in zip(hyperpars.keys(), best_hyperpars):
                model.named_steps['frc'].__setattr__(k, v)

        model.fit(self.trainX, self.trainY)

        return model, model.named_steps['frc'], model.named_steps['gen'], model.named_steps['sel']

    def forecast(self, model, idx_rows=None, replace=True):
        """
        Apply trained model to forecast the data

        :param model: Trained model
        :type model: Pipeline
        :param idx_rows: row indices of data matrix to be forecasted
        :type idx_rows: list
        :param replace: Flag that specifies is old forecast values should be replaced with new values
        :type replace: bool
        :return: Forecasted matrix, flat indices of forecasted values
        :rtype: ndarray, list
        """
        if idx_rows is None:
            idx_rows = range(self.X.shape[0])

        forecastedY = model.predict(self.X[idx_rows, :])
        if forecastedY.ndim == 1:
            forecastedY = forecastedY[:, None]

        # ravel forecasts and, if replace=True, input new values to self.forecasts vector
        idx_frc = self.add_forecasts(forecastedY, idx_rows, replace)

        return forecastedY, idx_frc

    def add_forecasts(self, frc, idx_rows, replace):
        """
        Computes flat indices of forecats and optionally replaces old forecast values new ones

        :param frc: Forecasted matrix Y
        :type frc: ndarray
        :param idx_rows: row indices of forecasted Y columns in the data matrix
        :type idx_rows: list
        :param replace: If true, the old forecast values are replaced with new ones
        :type replace: bool
        :return: Flat indices of forecasted time series for each time series
        :rtype: list
        """

        # Infer ts flat indices from matrix structure
        idx_flat = [0] * self.nts
        for i in self.y_idx:
            idx_flat[i] = _ravel_idx(self.idxY[i][idx_rows, :], len(self.forecasts[i]))#self.n_hist_points[i], self.n_req_points[i], self.X.shape[0])

        if replace:  # Replace previous forecast values with new forecasts
            for i in self.y_idx:
                self.forecasts[i][idx_flat[i]] = _ravel_y(frc[:, :self.n_req_points[i]], self.ts[i].norm_div, self.ts[i].norm_subt)
                frc = frc[:, self.n_req_points[i]:]

        return idx_flat

    def mae(self, idx_frc=None, idx_rows=None, out=None, y_idx=None):
        """
        Mean Absolute Error calculation.

        :param idx_frc: Indices of forecasted/target TS entries used to compute MAE
        :type idx_frc: list
        :param idx_rows: Alternatively, specify raw indices for matrix Y. If idx_frc is specified, idx_rows is ignored
        :type idx_rows: list
        :param out: Specification of out invokes printing MAE values. out string is printed before the output
        :type out: string
        :param y_idx: Specifies time series to compute errors for
        :type y_idx: list
        :return: list of MAPE values, one for each input time series
        :rtype: list
        """
        y_idx = _check_input_ts_idx(y_idx, self.y_idx)

        idx = [0] * self.nts
        if idx_frc is None:
            if idx_rows is None:
                for i in y_idx:
                    idx[i] = range(self.n_hist_points[i], len(self.forecasts[i]))

            else:
                for i in y_idx:
                    idx[i] = _ravel_idx(self.idxY[i][idx_rows, :], len(self.forecasts[i]))
        else:
            idx = idx_frc

        errors = np.zeros(self.nts)
        for i in y_idx:
            ind = idx[i]
            ts = self.ts[i].s
            errors[i] = np.mean(np.abs(ts[ind] - self.forecasts[i][ind]))

        if out is not None:
            print(out, "MAE")
            for i in y_idx:
                print(self.ts[i].name, errors[i])

        return errors[y_idx]

    def mape(self, idx_frc=None, idx_rows=None, out=None, y_idx=None):
        """
        Mean Absolute Percentage Error calculation.

        :param idx_frc: Indices of forecasted/target TS entries used to compute MAE
        :type idx_frc: list
        :param idx_rows: Alternatively, specify raw indices for matrix Y. If idx_frc is specified, idx_rows is ignored
        :type idx_rows: list
        :param out: Specification of out invokes printing MAPE values. out string is printed before the output
        :type out: string
        :param y_idx: Specifies time series to compute errors for
        :type y_idx: list
        :return: list of MAPE values, one for each input time series
        :rtype: list
        """
        y_idx = _check_input_ts_idx(y_idx, self.y_idx)

        idx = [0] * self.nts
        if idx_frc is None:
            if idx_rows is None:
                for i in y_idx:
                    idx[i] = range(self.n_hist_points[i], len(self.forecasts[i]))

            else:
                for i in y_idx:
                    idx[i] = _ravel_idx(self.idxY[i][idx_rows], len(self.forecasts[i]))
        else:
            idx = idx_frc

        errors = np.zeros((self.nts))
        for i in y_idx:
            ind = idx[i]
            denom = np.abs(self.ts[i].s[ind])
            denom[denom == 0] = np.mean(np.abs(self.ts[i].s))
            errors[i] = np.mean(np.divide(np.abs(self.ts[i].s[ind] - self.forecasts[i][ind]), denom))

        if out is not None:
            print(out, "MAPE")
            for i in y_idx:
                print(self.ts[i].name, errors[i])

        return errors[y_idx]

    def plot_frc(self, idx_frc=None, idx_rows=None, n_frc=1, n_hist=3, folder="fig", save=True, y_idx=None):
        """
        Plots forecasts along with time series

        :param idx_frc: Flat indices of forecasts to plot. Dominates idx_rows and n_frc
        :type idx_frc: list
        :param idx_rows: Row indices to plot. Dominates n_frc
        :type idx_rows: list
        :param n_frc: number of requested intervals to plot
        :type n_frc: int
        :param n_hist: number of historical intervals to plot
        :type n_hist: int
        :param folder: saving directory
        :type folder: str
        :param save: if True, the plots will be saved
        :type save: bool
        :param y_idx: defines indices of time series that will be plotted
        :type y_idx: int, list
        :return: None
        """
        idx = [0] * self.nts
        idx_ts = [0] * self.nts
        y_idx = _check_input_ts_idx(y_idx, self.y_idx)
        if idx_frc is None:
            if idx_rows is None:
                for i in y_idx:
                    idx[i] = range(len(self.forecasts[i]) - self.n_req_points[i]*n_frc, len(self.forecasts[i]))
                    idx_ts[i] = range(len(self.forecasts[i]) - self.n_req_points[i]*(n_frc + n_hist), len(self.forecasts[i]))

            else:
                for i in y_idx:
                    idx[i] = _ravel_idx(self.idxY[i][idx_rows], len(self.forecasts[i]))
        else:
            idx = idx_frc

        filenames = []
        for i in y_idx:
            if save:
                filename = self.ts[i].name + ".png"
                filename = "_".join(filename.split(" "))
                filenames.append(filename)
            else:
                filename = None
            my_plots.plot_forecast(self.ts[i], self.forecasts[i], idx_frc=idx[i], idx_ts=idx_ts[i], folder=folder, filename=filename)

        return filenames

    # def optimize_history(self, frc_model, sel_model=None, gen_model=None,  hist_range=None, n_fold=5):
    #     """
    #     Selects the optimal value of parameter history from the given range
    #
    #     :param frc_model: forecasting model
    #     :type frc_model: callable
    #     :param sel_model: feature selection model
    #     :type sel_model: callable
    #     :param gen_model: feature generation model
    #     :type gen_model: callable
    #     :param hist_range: range of history values to evaluate
    #     :type hist_range: list
    #     :return: history
    #     :rtype: int
    #     """
    #     if hist_range is None:
    #         hist_range = range(self.request, self.request*10, self.request)
    #
    #     mse = []
    #     for hist in hist_range:
    #         self.history = hist
    #         self.create_matrix()
    #         kf = KFold(self.X.shape[0], n_folds=n_fold)
    #         for idx_train, idx_test in kf:
    #             self.train_test_split(idx_train=idx_train, idx_test=idx_test)
    #             self.train_model(frc_model=frc_model, selector=sel_model, generator=gen_model)
    #             frc, _  = self.forecast()
    #             mse.append(((abs(frc - self.Y)+ 0.0000001)/(abs(self.Y) + 0.0000001))^2)


def _normalize_ts(ts, name=None):
    """
    Standardize time series

    :param ts: time series
    :type ts: numpy 1d-array
    :param name: ts name, optional
    :type name: str
    :return: standardized time series (ts - b)/a \in [0, 1] and standardization parameters a, b
    :rtype: tuple
    """

    norm_subt = np.min(ts)
    norm_div = np.max(ts) - norm_subt
    if norm_div == 0:
        print("Time series", name, "is constant")
        norm_div = 1
        norm_subt = 0
    else:
        ts = (ts - norm_subt) / norm_div

    return ts, norm_div, norm_subt


def _denormalize(ts_list):
    for i, ts in enumerate(ts_list):
        if not ts.norm_div == 0:
            ts_list[i].s = (ts + ts.norm_subt) * ts.norm_div
    return ts_list

def truncate(ts_struct, n_hist, n_req, n_rows):
    """
    Truncates time series, leaving  n_hist + n_req * n_rows points from the beginning

    :param ts_struct: time series structure
    :type ts_struct: TsMiniStruct
    :param n_hist: number of historical points of TS, included into X
    :type n_hist: int
    :param n_req: number of points of TS to forecast (to include into Y)
    :type n_req: int
    :param n_rows: number of rows of X and Y, associated with TS
    :type n_rows: int
    :return: time series structure with modified .s field
    :rtype: TsMiniStruct
    """
    ts = ts_struct.s
    n_points = n_hist + n_req*n_rows

    ts = ts[:n_points]
    ts_struct = TsMiniStruct(ts, ts_struct.norm_div, ts_struct.norm_subt, ts_struct.name, ts_struct.index[:n_points])
    return ts_struct

def check_time(y, x):
    """
    Checks that all x time entries preceed corresponding y time entries
    :param y: for each time series, y stores array of earliest time entries in Y (by rows)
    :type y: list
    :param x: stores latest time for X in each row
    :type x: list
    :return: returns False if x and y overlap
    :rtype: bool
    """
    check = []
    # y stores earliest time for Y in each row, x stores latest time for X in each row. These two must not overlap
    for ty, tx in product(y, x):
        check.append(np.all(ty > tx))


    return np.all(check)


def replace_nans(ts, name=None):
    #
    if not np.isnan(ts).any():
        return ts

    #print("Filling NaNs for TS", name)
    if np.isnan(ts).all():
        print("All inputs if {} are NaN, replacing with zeros".format(name))
        ts = np.zeros_like(ts)
        return ts

    ts_prop = pd.Series(ts).fillna(method="pad")
    ts_back = pd.Series(ts_prop).fillna(method="bfill")
    ts = ts_back.as_matrix() # (ts_back + ts_prop)[pd.isnull(ts)] / 2

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
    for i in range(n_rows):
        flat_idx.extend(range(i * n_req, (i + 1) * n_req + n_hist))
        # idx = np.unravel_index(flat_idx, (n_rows, n_hist + n_req))

    idx_matrix = np.reshape(flat_idx, (n_rows, n_hist + n_req))
    idxX = idx_matrix[:, n_req:]
    idxY = idx_matrix[:, :n_req]

    return idxX, idxY


def _ravel_idx(idx_matr, m):
    return m - 1 - np.ravel(idx_matr)#np.flipud(np.ravel(idx_matr))


def _ravel_y(y_mat, norm_div, norm_subt):
    y = np.ravel(y_mat) * norm_div + norm_subt
    return y

def _check_input_ts_idx(idx, default):
    if idx is None:
        idx = default
    elif isinstance(idx, list):
        idx = idx
    elif isinstance(idx, int):
        idx = [idx]
    else:
        warnings.warn("Unvalid argument idx in RegMatrix", UserWarning)
        idx = default
        
    return list(idx)


def cv_train(raw_model, data, hyperpars, n_cvs):
    from sklearn.model_selection import KFold
    import sys
    if n_cvs is None:
        n_cvs = 5

    X = data.trainX
    Y = data.trainY

    kf = KFold(n_splits=n_cvs)
    kf.get_n_splits(X)

    par_names = list(hyperpars.keys())
    par_values_range = list(hyperpars.values())
    scores = []
    for i, hyperpars in enumerate(product(*par_values_range)):
        scores.append(np.zeros(n_cvs))
        pars = {key: val for key, val in zip(par_names, hyperpars)}

        for k, train_val_index in enumerate(kf.split(X)):
            model = raw_model
            for key, val in pars.items():
                model.named_steps["frc"].__setattr__(key, val)
            print("\r{}, kfold = {}".format(pars, k), end="")
            sys.stdout.flush()
            # getting training and validation data
            train_index, val_index = train_val_index[0], train_val_index[1]
            x_train, x_val = X[train_index], X[val_index]
            y_train, y_val = Y[train_index], Y[val_index]
            # train the model and predict the MSE
            try:
                model.fit(x_train, y_train)
                pred_val = model.predict(x_val)
                scores[-1][k] = mean_squared_error_(pred_val, y_val)
            except BaseException as e:
                print(e)
                if k > 0:
                    scores[-1][k] = scores[-1][k-1]
                else:
                    scores[-1][k] = 1

        scores[-1] = np.mean(scores[-1])
    idx = np.argmin(scores)
    best_hyperpars = list(product(*par_values_range))[idx]
    print("Best hyperpars combo: {} with mse {}".format(zip(par_names, best_hyperpars), scores[idx]))

    return best_hyperpars

def mean_squared_error_(f, y):
    return np.mean((f - y) ** 2)


# def history_from_periods(ts_list, n_periods=3):
#     import Forecasting
#     history = []
#     for ts in ts_list:
#         print(ts.name)
#         history.append(n_periods*Forecasting.preprocess_time_series.find_period(ts))



    