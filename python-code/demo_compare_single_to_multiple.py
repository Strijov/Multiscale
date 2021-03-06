from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import utils_

from sklearn.linear_model import Lasso

from RegressionMatrix import regression_matrix
from Forecasting import frc_class
# from Forecasting import LSTM, GatingEnsemble


TRAIN_TEST_RATIO = 0.75
N_PREDICTIONS = 10
N_EXPERTS = 4


def main(file_name, line_indices, header):
    """
    Compares simultaneous (all-on-all regression) forecasts to individual (one-on-one). The data is in IoT format

    :param file_name: file name (.csv) with data in IoT format
    :type file_name: str
    :param line_indices: indices of lines to read from file.  Lines are enumerated from 1. If "all", read the whole file
    :param header: Specifies if the file contains a header row
    :type header: bool
    :return:
    :rtype:
    """

    ts = utils_.safe_read_iot_data(file_name, line_indices, header)
    err_all = forecating_errors(ts, range(len(ts.data)))
    column_names = [("MAE", "train"), ("MAPE", "train"), ("MAE", "test"), ("MAPE", "test")]

    res_all = data_frame_res(err_all, column_names, ts)

    train_mae, train_mape, test_mae, test_mape = [None]*len(ts.data), [None]*len(ts.data),[None]*len(ts.data),[None]*len(ts.data)
    for i in xrange(len(ts.data)):
        train_mae[i], train_mape[i], test_mae[i], test_mape[i] = forecating_errors(ts, i)
        train_mae, train_mape = np.hstack(train_mae), np.hstack(train_mape)
        test_mae, test_mape = np.hstack(test_mae), np.hstack(test_mape)

    err_by_one = [train_mae, train_mape, test_mae, test_mape]
    res_by_one = data_frame_res(err_by_one, column_names, ts)
    diff = [np.divide(err1 - err2, err1)*100 for err1, err2 in zip(err_by_one, err_all) ]
    diff_res = data_frame_res(diff, column_names, ts)

    print("Simultaneous forecast")
    print(res_all)
    print("\nIndividual forecasts")
    print(res_by_one)

    print("\nPerformance increase (in percents of individual errors)")
    print(diff_res)
    return res_all, res_by_one


def forecating_errors(ts, ts_idx):

    data = regression_matrix.RegMatrix(ts, y_idx=ts_idx, x_idx=ts_idx)
    # Create regression matrix
    data.create_matrix(nsteps=1, norm_flag=True)

    frc_model = frc_class.CustomModel(Lasso, name="Lasso", alpha=0.001)
    # frc_model = frc_class.CustomModel(LSTM.LSTM, name="LSTM")
    # frc_model = frc_class.CustomModel(GatingEnsemble.GatingEnsemble,
    #                                   estimators=[LinearRegression() for i in range(N_EXPERTS)])  # (LSTM.LSTM, name="LSTM")

    # Split data for training and testing
    data.train_test_split(TRAIN_TEST_RATIO)
    model = frc_class.PipelineModel(gen_mdl=None, sel_mdl=None, frc_mdl=frc_model)
    model, _, _, _ = model.train_model(data.trainX, data.trainY)  # model parameters are changed inside

    data.forecast(model, replace=True)

    train_mae = data.mae(idx_rows=data.idx_train, idx_original=data.original_index)
    train_mape = data.mape(idx_rows=data.idx_train, idx_original=data.original_index)
    test_mae = data.mae(idx_rows=data.idx_test, idx_original=data.original_index)
    test_mape = data.mape(idx_rows=data.idx_test, idx_original=data.original_index)

    return train_mae, train_mape, test_mae, test_mape


def data_frame_res(columns, column_names, ts):
    res = []
    for col, name in zip(columns, column_names):
        res.append(pd.DataFrame(col, index=[t.name for t in ts.data], columns=[name]))

    res = pd.concat(res, axis=1)
    return res

if __name__ == '__main__':
    file_name, line_indices, header, _ = utils_.parse_options()
    main(file_name, line_indices, header)
