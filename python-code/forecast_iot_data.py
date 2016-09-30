from __future__ import division
from __future__ import print_function

import os
import optparse
import pandas as pd

from sklearn.linear_model import Lasso

from LoadAndSaveData import get_iot_data, write_data_to_iot_format, load_time_series
from RegressionMatrix import regression_matrix
from Forecasting import frc_class

def main(file_name, line_indices, header):

    TRAIN_TEST_RATIO = 0.75
    N_PREDICTIONS = 10

    if not os.path.exists(file_name):
        load_raw = not os.path.exists(os.path.join("ProcessedData", "EnergyWeather_orig_train.pkl"))
        ts_struct = load_time_series.load_all_time_series(datasets='EnergyWeather', load_raw=load_raw,
                                                          name_pattern="orig_train")[0]
        write_data_to_iot_format.write_ts(ts_struct, file_name)


    data, metric_ids, host_ids, header_names = get_iot_data.get_data(file_name, line_indices, header)

    dataset = host_ids.keys()[0]
    ts = load_time_series.from_iot_to_struct(data, host_ids[dataset], dataset)

    data = regression_matrix.RegMatrix(ts)
    # Create regression matrix
    data.create_matrix(nsteps=1, norm_flag=True)

    frc_model = Lasso(alpha=0.01)

    # Split data for training and testing
    data.train_test_split(TRAIN_TEST_RATIO)
    model = data.train_model(frc_model=frc_model, generator=None,
                             selector=None)  # model parameters are changed inside

    print("Features before generation:", data.feature_dict)
    print("Features after generation:", model.named_steps["gen"].feature_dict)
    print("Features after generation:", model.named_steps["sel"].feature_dict)
    frc_class.print_pipeline_pars(model)

    # data.forecasts returns model obj, forecasted rows of Y matrix and a list [nts] of "flat"/ts indices of forecasted points
    data.forecast(model, replace=True)

    train_mae = data.mae(idx_rows=data.idx_train, out=None)  # , out="Training")
    train_mape = data.mape(idx_rows=data.idx_train, out=None)  # , out="Training")
    test_mae = data.mae(idx_rows=data.idx_test, out=None)  # , out="Test")
    test_mape = data.mape(idx_rows=data.idx_test, out=None)  # , out="Test")

    res1 = pd.DataFrame(train_mae, index=[t.name for t in ts.data], columns=[("MAE", "train")])
    res2 = pd.DataFrame(train_mape, index=[t.name for t in ts.data], columns=[("MAPE", "train")])
    res3 = pd.DataFrame(test_mae, index=[t.name for t in ts.data], columns=[("MAE", "test")])
    res4 = pd.DataFrame(test_mape, index=[t.name for t in ts.data], columns=[("MAPE", "test")])
    res = pd.concat([res1, res2, res3, res4], axis=1)
    print(res)

    data.plot_frc(n_frc=N_PREDICTIONS)

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
