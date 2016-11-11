import os
import optparse

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



def safe_read_iot_data(file_name, line_indices, header, default="EnergyWeather"):
    """
    If the data can't be read from file_name, first write it to iot format, then read from it.
    """

    if not os.path.exists(file_name):
        if default.lower() == "poisson":
            ts_struct = random_data.create_iot_data_poisson(n_ts=3, n_req=10, n_hist=20, max_length=5000, min_length=2000,
                                                    slope=0.0001, trend_noise=0.001, non_zero_ratio=0.001)
        elif default.lower() == "random":
            ts_struct = random_data.create_iot_data(n_ts=3, n_req=10, n_hist=20, max_length=5000,
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