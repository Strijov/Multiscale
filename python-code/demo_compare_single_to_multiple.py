from __future__ import division
from __future__ import print_function

import os
import optparse
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso, LinearRegression

from LoadAndSaveData import get_iot_data, write_data_to_iot_format, load_time_series
from RegressionMatrix import regression_matrix
from Forecasting import frc_class, LSTM, GatingEnsemble


TRAIN_TEST_RATIO = 0.75
N_PREDICTIONS = 10
N_EXPERTS = 4

def main(file_name, line_indices, header):


    data, metric_ids, host_ids, header_names = get_iot_data.get_data(file_name, line_indices, header)
    dataset = host_ids.keys()[0]
    ts = load_time_series.from_iot_to_struct(data, host_ids[dataset], dataset)


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
    diff = [err1 - err2 for err1, err2 in zip(err_by_one, err_all) ]
    diff_res = data_frame_res(diff, column_names, ts)

    print("Simultaneous forecast")
    print(res_all)
    print("\nIndividual forecasts")
    print(res_by_one)

    print("\nPerformance increase")
    print(diff_res)


    return res_all, res_by_one




def forecating_errors(ts, ts_idx):

    data = regression_matrix.RegMatrix(ts, y_idx=ts_idx, x_idx=ts_idx)
    # Create regression matrix
    data.create_matrix(nsteps=1, norm_flag=True)

    frc_model = frc_class.CustomModel(Lasso, alpha=0.01)
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

    # parser.add_option('-m', '--model',
    #                   type='string', default='model-12-02-2016.pickle',
    #                   help='Filename for trained model serialization. Default: %default')
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
