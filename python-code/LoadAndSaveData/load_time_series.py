from __future__ import division
import os.path
import glob
import re

from sklearn.externals import joblib
from raw_time_series import TsStruct

import load_energy_weather_data
DIRNAME = 'ProcessedData' # directory to store data (.pkl) in

# Define a dict of func names for data loading
LOAD_FUNCS_DICT = {'EnergyWeather': load_energy_weather_data,
              'NNcompetition': 'load_energy_weather_data', # FIXIT
              'Artifical':'load_energy_weather_data'} #FIXIT

# Define a dict for raw data directories
RAW_DIRS_DICT = {'EnergyWeather': '../code/data/EnergyWeatherTS/orig',
              'NNcompetition': '../code/data/NNcompetition'} #FIXIT




def load_all_time_series(datasets=None, load_funcs=None, name_pattern='', load_raw=True):
    """
    Data loader

    :param datasets: contains names datasets to download
    :type datasets: list
    :param load_funcs: contains callables for each dataset. Used if load_raw=True
    :type load_funcs: list
    :param name_pattern: expression to look for in loaded file names
    :type name_pattern: string
    :param load_raw: If set to True, the raw data is reloaded first
    :type load_raw: boolean
    :return:
    :rtype:
    """



    if datasets is None:
        datasets = 'EnergyWeather'#['NNcompetition', 'EnergyWeather']
    # make it a list of datasets
    if not datasets is list:
        datasets = [datasets]

    if load_funcs is None:
        load_funcs = [LOAD_FUNCS_DICT[x] for x in datasets]

    if load_raw:
        load_raw_data(load_funcs)

    # find all .pkl files in DIRNAME directory
    filenames = glob.glob(os.path.join(DIRNAME, '*.pkl'))

    all_ts = []
    for fn in filenames:
        # ts is a namedtuple tsStruct
        ts = joblib.load(fn)

        # check if the name of time series matches the pattern
        match_pattern = len(re.findall(name_pattern, ts.name)) > 0
        # and select only those from the data sets, listed in 'datasets'
        match_dataset = ts.name.split('_')[0] in datasets #FIXIT
        if match_dataset and match_pattern:
            all_ts.append(ts)

    return all_ts

def load_raw_data(load_funcs):
    """
    Loads and saves raw data in .pkl format

    :param load_funcs: Each function (callable) is load_funcs loads some dataset
    :type load_funcs: list
    :return:
    :rtype:  None
    """
    # dirnames is a (list of) names of directory with raw data, passed to load_ts func
    # DIRNAME is the common directory for saving processed data

    for func in load_funcs:
        ts_list, names = getattr(func, 'load_ts')()
        for ts, name in zip(ts_list, names):
            save_ts_to_dir(ts, name, DIRNAME)


def save_ts_to_dir(ts, tsname, dirname):
    """
    Saves time series into specified directory

    :param ts: time series
    :type ts: list of TsStruct
    :param tsname: Filename. Data will be saved as .pkl, do not specify any extensions
    :type tsname: string
    :param dirname: Directory that stores processed data
    :type dirname: string
    :return:
    :rtype: None
    """
    # save time series under the name dirname/tsname
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    tsname = os.path.join(dirname, tsname) + '.pkl'
    joblib.dump(ts, tsname)

def from_iot_to_struct(ts_list, idx, dataset):
    """
    Converts data from IoT output to tsStruct. Request is single point for every ts and history is unknown

    :param ts_list: stores data in pandas.Series format
    :type ts_list: list
    :param idx: indices of time series in ts_list correspondent to specific host/dataset
    :type idx: list
    :param dataset: host/dataset name
    :type dataset: string
    :return: data structure with selected time series
    :rtype: TsStruct
    """

    request, ts = [], []
    for i in idx:
        request.append(ts_list[i].index[1] - ts_list[i].index[0])
        ts.append(ts_list[i])

    return TsStruct(ts, None, None, dataset, "")


if __name__ == '__main__':
    load_all_time_series()
