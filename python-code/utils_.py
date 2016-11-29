import os
import optparse

import pandas as pd
from LoadAndSaveData import load_time_series, write_data_to_iot_format, get_iot_data
from RegressionMatrix import random_data

def parse_options():
    """Parses the command line options."""
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option('-f', '--filename',
                      type='string',
                      default=os.path.join('..', 'code','data', 'IotTemplate', 'data.csv'),
                      help='.csv file with input data. Default: %default')
    parser.add_option('-l', '--line-indices',
                      type='string', default="all",#"15, 16",
                      help='Line indices to be read from file. Default: %default')
    parser.add_option('-d', '--header',
                      type='string', default='True',
                      help='Header flag. True means the first line of the csv file in the columns 1 to 8 are variable names.\
                       Default: %default')
    parser.add_option('-t', '--format_',
                      type='string', default='date',
                      help='Define naming style of the folder with results.\
                           Default: %default')

    opts, args = parser.parse_args()
    opts.__dict__['header'] = bool(opts.__dict__['header'])

    if opts.__dict__['line_indices'] == "all":
        ln = opts.__dict__['line_indices']
    else:

        ln = opts.__dict__['line_indices'].split(",")
        for i, idx in enumerate(ln):
            ln[i] = int(idx)

    return opts.__dict__['filename'], ln, opts.__dict__['header'], opts.__dict__['format_']



def safe_read_iot_data(file_name, line_indices, header, default="EnergyWeather", verbose=False):
    """
    If the data can't be read from file_name, first write it to iot format, then read from it.
    """


    if not os.path.exists(os.path.abspath(file_name)):
        if verbose:
            print("File {} not found, using data generation scheme '{}'.".format(os.path.abspath(file_name), default))
        if default.lower() == "poisson":
            ts_struct = random_data.create_iot_data_poisson(n_ts=5, n_req=2, n_hist=7, max_length=10000, min_length=2000,
                                                    slope=0.0001, trend_noise=0.3, non_zero_ratio=0.1)
        elif default.lower() == "random":
            ts_struct = random_data.create_iot_data(n_ts=3, n_req=2, n_hist=7, max_length=5000,
                                                            min_length=2000, slope=0.0001, trend_noise=0.001)
        else:
            load_raw = not os.path.exists(os.path.join("ProcessedData", "EnergyWeather_orig_train.pkl"))
            ts_struct = load_time_series.load_all_time_series(datasets=['EnergyWeather'], load_raw=load_raw,
                                                              name_pattern="missing")[0]


        write_data_to_iot_format.write_ts(ts_struct, file_name)

    data, metric_ids, host_ids, header_names = get_iot_data.get_data(file_name, line_indices, header)

    ts_list = load_time_series.iot_to_struct_by_dataset(data, host_ids, dataset_idx=[0])

    if len(ts_list) == 0:
        print("Data list, read from {} is empty".format(file_name))
        raise ValueError

    return ts_list[0]


def read_web_arguments(args):
    """
    Reads arguments from server code

    :param args: input arguments
    :type args: list
    :return: parsed arguments
    """
    msg = ''
    file_name, frc_model, n_hist, n_req, train_test = None, None, None, None, 0.75
    # try:
    file_name, frc_model, n_hist, n_req, train_test = args[0], args[1], args[2], args[3], args[4]
    n_hist = int(n_hist)
    n_req = int(n_req)
    train_test = float(train_test)
    # except:
    #     msg += 'Arguments should contain at least 4 fields. Using default.\n'
    #     print(msg)

    pars = {}
    i = 5
    while i < len(args):
        print(args[i])
        if args[i] == 'alpha':
            pars['alpha'] = float(args[i + 1])
        elif args[i] == 'lr':
            pars['learning_rate'] = float(args[i + 1])
        elif args[i] == 'n_units':
            pars['num_lstm_units'] = int(args[i + 1])
        elif args[i] == 'n_epochs':
            pars['n_epochs'] = int(args[i + 1])
        elif args[i] == 'n_estimators':
            pars['n_estimators'] = int(args[i + 1])
        else:
            print("Unexpected keyword {} in passed from server.js".format(args[i]))
            i += 1
        i += 2

    return file_name, frc_model, n_hist, n_req, train_test, pars, msg

def train_test_errors_table(data):

    train_mae = data.mae(idx_rows=data.idx_train)
    train_mape = data.mape(idx_rows=data.idx_train)
    test_mae = data.mae(idx_rows=data.idx_test)
    test_mape = data.mape(idx_rows=data.idx_test)

    ts_names = [data.ts[i].name for i in data.y_idx]

    res1 = pd.DataFrame(train_mae, index=ts_names, columns=[("MAE", "train")])
    res2 = pd.DataFrame(train_mape, index=ts_names, columns=[("MAPE", "train")])
    res3 = pd.DataFrame(test_mae, index=ts_names, columns=[("MAE", "test")])
    res4 = pd.DataFrame(test_mape, index=ts_names, columns=[("MAPE", "test")])
    res = pd.concat([res1, res2, res3, res4], axis=1)

    return res
