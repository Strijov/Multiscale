from __future__ import division
from __future__ import print_function

import os
import optparse
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso, LinearRegression

from LoadAndSaveData import get_iot_data, write_data_to_iot_format, load_time_series
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

    ts = safe_read_iot_data(file_name, line_indices, header)


    err_all = forecating_errors(ts, range(len(ts.data)))
    column_names = [("MAE", "train"), ("MAPE", "train"), ("MAE", "test"), ("MAPE", "test")]

    res_all = data_frame_res(err_all, column_names, ts)

    train_mae, train_mape, test_mae, test_mape = [None]*len(ts.data), [None]*len(ts.data),[None]*len(ts.data),[None]*len(ts.data)
    for i in xrange(len(ts.data)):
        train_mae[i], train_mape[i], test_mae[i], test_mape[i] = forecating_errors(ts, i)

        train_mae, train_mape = np.hstack(train_mae), np.hstack(train_mape)
        test_mae, test_mape =  np.hstack(test_mae), np.hstack(test_mape)

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


def safe_read_iot_data(file_name, line_indices, header):
    """
    If the data can't be read from file_name, first write it to iot format, then read from it.
    """

    if not os.path.exists(file_name):
        load_raw = not os.path.exists(os.path.join("ProcessedData", "EnergyWeather_orig_train.pkl"))
        ts_struct = load_time_series.load_all_time_series(datasets=['EnergyWeather'], load_raw=load_raw,
                                                          name_pattern="missing")[0]

        write_data_to_iot_format.write_ts(ts_struct, file_name)
    data, metric_ids, host_ids, header_names = get_iot_data.get_data(file_name, line_indices, header)
    dataset = host_ids.keys()[0]
    ts = load_time_series.from_iot_to_struct(data, host_ids[dataset], dataset)

    return ts




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
    model, _, _, _ = data.train_model(frc_model=frc_model, generator=None,
                                      selector=None)  # model parameters are changed inside

    data.forecast(model, replace=True)

    train_mae = data.mae(idx_rows=data.idx_train)
    train_mape = data.mape(idx_rows=data.idx_train)
    test_mae = data.mae(idx_rows=data.idx_test)
    test_mape = data.mape(idx_rows=data.idx_test)

    return train_mae, train_mape, test_mae, test_mape


def data_frame_res(columns, column_names, ts):
    res = []
    for col, name in zip(columns, column_names):
        res.append(pd.DataFrame(col, index=[t.name for t in ts.data], columns=[name]))

    res = pd.concat(res, axis=1)
    return res



def parse_options():
    """Parses the command line options."""
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option('-f', '--filename',
                      type='string',
                      default=os.path.join('..', 'code','data', 'IotTemplate', 'data.csv'),
                      help='.csv file with input data. Default: %default')
    parser.add_option('-l', '--line-indices',
                      type='string', default='all',
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
