"""
Created on Oct 20, 2016.
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import sys
import time
import optparse
import my_plots

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV, Lasso
from LoadAndSaveData import get_iot_data, load_time_series
from RegressionMatrix import regression_matrix
from Forecasting import frc_class, arima_model, LSTM, GatingEnsemble

N_EXPERTS = 4

def main(file_name=None, line_indices="all", header=True):
    """
    Runs forecasting models and reports results in latex file

    :param file_name: file name (.csv) with data in IoT format
    :type file_name: str
    :param line_indices: indices of lines to read from file.  Lines are enumerated from 1. If "all", read the whole file
    :param header: Specifies if the file contains a header row
    :type header: bool
    :return: latex report
    :rtype: str
    """
    # Init string for latex results:
    latex_str = ""
    time_at_start = time.time()
    folder = os.path.join("fig", str(int(time.time())))
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Load data in IoT format
    try:
        data, metric_ids, host_ids, header_names = get_iot_data.get_data(file_name, line_indices, header)
    except BaseException as e:
        print(e)
        print("Line indices: ", line_indices)
        print("Fileame: ", file_name)
        return None


    # Select only data from first dataset in host_ids:
    dataset = host_ids.keys()[0] # select the first dataset
    ts = load_time_series.from_iot_to_struct(data, host_ids[dataset], dataset) # get all time series from dataset in TsStruct format
    ts.align_time_series() # truncate time series to align starting and ending points
    latex_str += ts.summarize_ts(latex=True)

    # split time series into train and validation
    train, test = ts.train_test_split(train_test_ratio=0.75) # split raw time series into train and test parts

    # Plot periodics:
    for i, tsi in enumerate(ts.data):
        save_to = os.path.join(folder, "decompose", "_".join(tsi.name.split(" ")))
        # infer periodicity and try to decompose ts into tend, seasonality and resid:
        period, msg = arima_model.decompose(tsi, nhist=500, folder=save_to, nsplits=50)
        latex_str += my_plots.check_text_for_latex(tsi.name) + ": "
        latex_str += msg
        latex_str += arima_model.make_report(os.path.join(save_to), write=False) # adds figures from "save_to" to latex_str





    # Declare models to compare:
    random_forest = frc_class.CustomModel(RandomForestRegressor, n_jobs=24, name="RandomForest")
    # mixture_experts = frc_class.CustomModel(GatingEnsemble.GatingEnsemble, name="Mixture",
    #                                          estimators=[RidgeCV(), LassoCV()])
    lstm = frc_class.CustomModel(LSTM.LSTM, name="LSTM", n_epochs=50, plot_loss=True)
    lasso = frc_class.CustomModel(Lasso, name="Lasso", fit_intercept=True, alpha=2.0)
    model_list = [lasso] # random_forest, mixture_experts, lstm

    params_range ={}
    params_range["RandomForest"] = {"n_estimators": [3000]}
    params_range["Mixture"] = {"n_hidden_units":[10, 20, 30, 50, 100]}
    params_range["LSTM"] = {"batch_size": [20, 30, 50, 100]}
    params_range["Lasso"] = {"alpha": [float(i) / 10000 for i in  range(1, 11, 1)]  + [0.01, 0.05]}  # [20, 30, 50, 100]} #[1.0, 1.25, 1.5, 1.75, 2.0]


    WINDOWS = [2, 5, 7, 10, 15, 20]
    N_FOLDS = 5

    for model in model_list:
        model_save_path = os.path.join(folder, model.name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        # select number of trees and history parameter:
        # (history parameter is divisible by request)
        n_req, params, best_train_mse, plt = train_model_CV(train, model, n_fold=N_FOLDS, windows=WINDOWS,
                                        params=params_range[model.name], plot=True)#windows=[5, 10, 25, 50, 75, 100, 150])

        plt.savefig(os.path.join(model_save_path, "cv_optimization.png"))
        plt.clf()

        #n_req, nr_tree, best_train_mse = 10, 500, 0.00658112163657  # previously estimated

        opt_string = model.name + ". Best CV error: {0}, estimated parameters: history = {1}, {2} = {3} " \
                     "\\\\ \n".format(best_train_mse, n_req, my_plots.check_text_for_latex(params.keys()[0]), params.values()[0])
        print(opt_string)
        latex_str += opt_string

        # use selected parameters to forecast trainning data:
        if not len(params) == 0:
            model.__setattr__(params.keys()[0], params.values()[0])
        data = regression_matrix.RegMatrix(ts)
        data.history = n_req * data.request

        data.create_matrix()
        data.train_test_split()

        model, frc, _, _ = data.train_model(frc_model=model)

        # if hasattr(frc, "msg"):
        #     latex_str += msg
        if hasattr(frc, "fig"):
            frc.fig.savefig(os.path.join(model_save_path, "fitting.png"))

        train_frc, _ = data.forecast(model, idx_rows=data.idx_train)
        train_mse = mean_squared_error(train_frc, data.trainY)

        test_frc, _ = data.forecast(model, idx_rows=data.idx_test)
        test_mse = mean_squared_error(test_frc, data.testY)

        latex_str += my_plots.check_text_for_latex(model.name) + "\\\\ \n"
        latex_str += "Train error for estimated parameters: {0}, " \
                     "test error with estimated parameters {1} \\\\ \n".format(train_mse, test_mse)

        err_all = forecasting_errors(data)
        column_names = [("MAE", "train"), ("MAPE", "train"), ("MAE", "test"), ("MAPE", "test")]
        res_all = data_frame_res(err_all, column_names, ts)

        print(model.name)
        print(res_all)

        latex_str += res_all.to_latex()
        latex_str += "\\bigskip \n \\\\"

        data.plot_frc(n_frc=10, n_hist=10, folder=model_save_path)
        latex_str += my_plots.include_figures_from_folder(model_save_path)

    total_time = time.time() - time_at_start
    latex_str += "\n Total time: {0}\n \\".format(total_time)
    my_plots.print_to_latex(latex_str, check=False, file_name="IoT_example", folder=folder)

    return latex_str




def data_frame_res(columns, column_names, ts):
    res = []
    for col, name in zip(columns, column_names):
        res.append(pd.DataFrame(col, index=[t.name for t in ts.data], columns=[name]))

    res = pd.concat(res, axis=1)
    return res


def forecasting_errors(data):
    train_mae = data.mae(idx_rows=data.idx_train)
    train_mape = data.mape(idx_rows=data.idx_train)
    test_mae = data.mae(idx_rows=data.idx_test)
    test_mape = data.mape(idx_rows=data.idx_test)

    return train_mae, train_mape, test_mae, test_mape


def train_model_CV(data, model, n_fold=5, windows=[5, 10, 25, 50, 75, 100, 150],
                   params={}, f_horizon=1, plot=False):

    if len(params) == 0:
        par_name, params_range = None, []
    else:
        par_name, params_range = params.items()[0]
    params_range = params[par_name]
    scores = np.zeros((len(windows), len(params_range), n_fold))

    for w_ind in range(0, len(windows)):
        # obtain the matrix from  the time series data with a given window-size
        data.history = windows[w_ind] * data.request
        mat = regression_matrix.RegMatrix(data)
        mat.create_matrix(f_horizon)
        w_train = mat.X
        y_wtrain = mat.Y
        # (w_train, y_wtrain) = windowize(data, windows[w_ind], f_horizon=f_horizon)

        # cross-validation
        r, c = w_train.shape
        kf = KFold(n_splits=n_fold)
        kf.get_n_splits(w_train)
        for par_ind in range(0, len(params_range)):
            model.__setattr__(par_name, params_range[par_ind])
            n = 0
            for train_index, val_index in kf.split(w_train):
                print("\rWindow size: {0}, {1} = {2}, kfold = {3}".format(windows[w_ind], par_name, params_range[par_ind], n), end="")
                sys.stdout.flush()
                # getting training and validation data
                X_train, X_val = w_train[train_index, :], w_train[val_index, :]
                y_train, y_val = y_wtrain[train_index], y_wtrain[val_index]
                # train the model and predict the MSE
                try:
                    model.fit(X_train, y_train)
                    pred_val = model.predict(X_val)
                    scores[w_ind, par_ind, n] = mean_squared_error(pred_val, y_val)
                except BaseException as e:
                    print(e)
                    if n > 0:
                        scores[w_ind, par_ind, n] = scores[w_ind, par_ind, n-1]
                    else:
                        scores[w_ind, par_ind, n] = 0
                n += 1
    m_scores = np.average(scores, axis=2)
    mse = m_scores.min()

    # select best window_size and best n_tree with smallest MSE
    (b_w_ind, b_tree_ind) = np.where(m_scores == mse)
    b_w_ind, b_tree_ind = b_w_ind[0], b_tree_ind[0]
    window_size, best_par = windows[b_w_ind], params_range[b_tree_ind]
    best_par = {par_name:best_par}

    if not plot:
        return (window_size, best_par, mse)

    plt = my_plots.imagesc(m_scores, xlabel=par_name, ylabel="n_req", yticks=windows, xticks=params_range)
    return window_size, best_par, mse, plt


def mean_squared_error(f, y):
    return np.mean((f-y)**2)


def parse_options():
    """Parses the command line options."""
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option('-f', '--filename',
                      type='string',
                      default=os.path.join('..', 'code','data', 'IotTemplate', 'data.csv'),
                      help='.csv file with input data. Default: %default')
    parser.add_option('-l', '--line-indices',
                      type='string', default="15, 16",
                      help='Line indices to be read from file. Default: %default')
    parser.add_option('-d', '--header',
                      type='string', default='True',
                      help='Header flag. True means the first line of the csv file in the columns 1 to 8 are variable names.\
                       Default: %default')

    opts, args = parser.parse_args()
    opts.__dict__['header'] = bool(opts.__dict__['header'])

    if opts.__dict__['line_indices'] == "all":
        ln = opts.__dict__['line_indices']
    else:

        ln = opts.__dict__['line_indices'].split(",")
        for i, idx in enumerate(ln):
            ln[i] = int(idx)

    return opts.__dict__['filename'], ln, opts.__dict__['header']

if __name__ == '__main__':
    main(*parse_options())


