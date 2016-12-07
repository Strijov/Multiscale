# coding: utf-8
from __future__ import division
from __future__ import print_function

import pandas as pd
import utils_
import my_plots

from sklearn.linear_model import Lasso

from RegressionMatrix import regression_matrix
from Forecasting import frc_class
from Forecasting import LSTM, GatingEnsemble

def main(file_name, line_indices, header):
    """
    Forecast data simultaneously and separately and compare errors

    :param file_name: file name (.csv) with data in IoT format
    :type file_name: str
    :param line_indices: indices of lines to read from file.  Lines are enumerated from 1. If "all", read the whole file
    :param header: Specifies if the file contains a header row
    :type header: bool
    :return: forecasting errors
    :rtype: pandas.DataFrame
    """

    TRAIN_TEST_RATIO = 0.75
    N_PREDICTIONS = 10
    N_EXPERTS = 4
    VERBOSE = True

    # frc_model = frc_class.CustomModel(Lasso, name="Lasso", alpha=0.001)
    # frc_model = frc_class.CustomModel(GatingEnsemble.GatingEnsemble,
    #                                  estimators = [LinearRegression() for i in range(N_EXPERTS)])#(LSTM.LSTM, name="LSTM")

    ts = utils_.safe_read_iot_data(file_name=file_name, line_indices=line_indices, header=header, default="poisson", verbose=VERBOSE)
    if VERBOSE:
        print(ts.summarize_ts())

    # my_plots.plot_multiple_ts(ts.data, shared_x=True)
    data = regression_matrix.RegMatrix(ts)
    # Create regression matrix
    data.create_matrix(nsteps=1, norm_flag=True)
    # Split data for training and testing
    data.train_test_split(TRAIN_TEST_RATIO)

    lr_list = [2e-6, 2e-5, 2e-4]
    n_lstm_units = [20, 30, 40, 50]
    hyperpars = {"learning_rate": lr_list, "n_lstm_units": n_lstm_units}
    frc_model = frc_class.CustomModel(LSTM.LSTM, name="LSTM", n_epochs=20, plot_loss=True)
    model = frc_class.PipelineModel(frc_mdl=frc_model)

    model, frc, _, _ = model.train_model(data.trainX, data.trainY, hyperpars=hyperpars, n_cvs=5)  # model parameters are changed inside

    if hasattr(frc, "fig"):
        frc.fig.savefig("fitting_learn_rate_{}.png".format(frc.learning_rate))


    # data.forecasts returns model obj, forecasted rows of Y matrix and a list [nts] of "flat"/ts indices of forecasted points
    data.forecast(model, replace=True)

    train_mae = data.mae(idx_rows=data.idx_train)
    train_mape = data.mape(idx_rows=data.idx_train)
    test_mae = data.mae(idx_rows=data.idx_test)
    test_mape = data.mape(idx_rows=data.idx_test)

    res1 = pd.DataFrame(train_mae, index=[t.name for t in ts.data], columns=[("MAE", "train")])
    res2 = pd.DataFrame(train_mape, index=[t.name for t in ts.data], columns=[("MAPE", "train")])
    res3 = pd.DataFrame(test_mae, index=[t.name for t in ts.data], columns=[("MAE", "test")])
    res4 = pd.DataFrame(test_mape, index=[t.name for t in ts.data], columns=[("MAPE", "test")])
    res = pd.concat([res1, res2, res3, res4], axis=1)
    print("LSTM")
    print(res)

    data.plot_frc(n_frc=N_PREDICTIONS)

    frc_model = frc_class.CustomModel(Lasso, name="Lasso", alpha=0.001)
    model = frc_class.PipelineModel(frc_mdl=frc_model)
    model,_,_,_ = model.train_model(data.trainX, data.trainY)
    data.forecast(model, replace=True)

    train_mae = data.mae(idx_rows=data.idx_train)
    train_mape = data.mape(idx_rows=data.idx_train)
    test_mae = data.mae(idx_rows=data.idx_test)
    test_mape = data.mape(idx_rows=data.idx_test)


    res1 = pd.DataFrame(train_mae, index=[t.name for t in ts.data], columns=[("MAE", "train")])
    res2 = pd.DataFrame(train_mape, index=[t.name for t in ts.data], columns=[("MAPE", "train")])
    res3 = pd.DataFrame(test_mae, index=[t.name for t in ts.data], columns=[("MAE", "test")])
    res4 = pd.DataFrame(test_mape, index=[t.name for t in ts.data], columns=[("MAPE", "test")])
    res = pd.concat([res1, res2, res3, res4], axis=1)
    print("Lasso")
    print(res)

    return res


if __name__ == '__main__':
    file_name, line_indices, header, _ = utils_.parse_options()
    main(file_name, line_indices, header)
