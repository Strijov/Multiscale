from __future__ import division
import os.path
import glob
from sklearn.externals import joblib
import re
from collections import namedtuple

import load_energy_weather_data

tsStruct = namedtuple('tsStruct', 'data request history name readme')

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
    % Data loader
    % Input: datasets - cell array, contains names of the folders inside data/
    % directory
    %
    % Output: a cell array of ts structures
    % Description of the time series structure:
    %   t [T,1]  - Time in milliseconds since 1/1/1970 (UNIX format)
    %   x [T, N] - Columns of the matrix are time series; missing values are NaNs
    %   legend {1, N}  - Time series descriptions ts.x, e.g. ts.legend={?Consumption, ?Price?, ?Temperature?};
    %   nsamples [1] - Number of samples per time series, relative to the target time series
    %   deltaTp [1] - Number of local historical points
    %   deltaTr [1] - Number of points to forecast
    %   readme [string] -  Data information (source, formation time etc.)
    %   dataset [string] - Reference name for the dataset
    %   name [string] - Reference name for the time series
    %   Optional:
    %   type [1,N] (optional) Time series types ts.x, 1-real-valued, 2-binary, k ? k-valued
    %   timegen [T,1]=func_timegen(timetick) (optional) Time ticks generator, may
    %   contain the start (or end) time in UNIX format and a function to generate the vector t of the size [T,1]
    """


    if datasets is None:
        datasets = 'EnergyWeather'#['NNcompetition', 'EnergyWeather']
    # make it a list of datasets
    if not datasets is list:
        datasets = [datasets]

    if load_funcs is None:
        load_funcs = [LOAD_FUNCS_DICT[x] for x in datasets]

    if load_raw:
        dirnames = [RAW_DIRS_DICT[x] for x in datasets]
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
    # dirnames is a (list of) names of directory with raw data, passed to load_ts func
    # DIRNAME is the common directory for saving processed data

    for func in load_funcs:
        ts_list, names = getattr(func, 'load_ts')()
        for ts, name in zip(ts_list, names):
            save_ts_to_dir(ts, name, DIRNAME)


def save_ts_to_dir(ts, tsname, dirname):
    # save time series under the name dirname/tsname
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    tsname = os.path.join(dirname, tsname) + '.pkl'
    joblib.dump(ts, tsname)


if __name__ == '__main__':
    load_all_time_series()
