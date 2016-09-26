from __future__ import division
import os
import glob
import joblib
import pandas as pd
from collections import namedtuple

tsStruct = namedtuple('tsStruct', 'data request history name readme')
DATASET = "EnergyWeather"

def load_ts(dirname):
    """
    Loads a set of time series and returns the cell array of ts structres.
    :param dirname: [string] named of folder with the loaded data.
    :return: time series, namedtuple tsStruct
    """


    dirname = os.path.abspath(dirname)
    folders = os.path.split(dirname) #, os.sep
    folder_name = folders[-1]
    if len(folder_name) == 0:
        folder_name = folders[-2]

    readme = {'orig':'Original time energy-weather series',
        'missing_value':'Energy-weather time series with artificially inserted missing values',
        'varying_rates':'Energy-weather time series with varying sampling rate'}
    readme = readme[folder_name]

    filename_train, filename_test, filename_weather = read_missing_value_dir(dirname)

    ts = []
    names = []
    for i in range(len(filename_train)):
        train_ts, test_ts, time, train_weather, test_weather = load_train_test_weather(
            filename_train[i], filename_test[i], filename_weather[i])

        request =time[24] - time[0] # by default, forecasts are requested for one day ahead
        history =time[7*24] - time[0] # by default, ts history is one week


        train_ts = [train_ts]
        train_ts.extend(train_weather)
        test_ts = [test_ts]
        test_ts.extend(test_weather)

        name_train, _ = os.path.splitext(os.path.split(filename_train[i])[1])
        name_train = DATASET + '_' + folder_name + '_' + name_train
        name_test, _ = os.path.splitext(os.path.split(filename_test[i])[1])
        name_test = DATASET + '_' + folder_name + '_' + name_test
        names.extend([name_train, name_test])
        ts.append(tsStruct(train_ts, request, history, name_train, readme))
        ts.append(tsStruct(test_ts, request, history, name_test, readme))




    return ts, names



def load_train_test_weather(train, test, weather):
    """

    :param train, test, weather: filenames
    :return: [train_ts, test_ts, time_train, time_test, train_weather, test_weather]
    """


    _, extension = os.path.splitext(train)


    weather_data = pd.read_csv(weather, sep=',', encoding='latin1')  # csvread(weather, 1, 1);
    weather_data = weather_data[['Date', 'Max Temperature', 'Min Temperature',
                                 'Precipitation', 'Wind', 'Relative Humidity', 'Solar']]
    weather_data['Date'] = pd.to_datetime(weather_data['Date'])
    test_ts, test_time,_ = process_csv_output(test, weather_data['Date'][1096], 1096)
    train_ts, train_time,_ = process_csv_output(train, weather_data['Date'][0], 1096)
    hourly_time =  train_time + test_time # for pd.time use # train_time.append(test_time)
    weather_data['Date'] = range(0, 24*1096*2, 24) #pd.to_numeric(weather_data['Date'])

    weather_train_ts = []
    weather_test_ts = []

    for k in ['Max Temperature', 'Min Temperature', 'Precipitation', 'Wind', 'Relative Humidity', 'Solar']:
        series = pd.Series(weather_data[k][:1096], index=weather_data["Date"][:1096], name=k)
        weather_test_ts.append(series)
        series = pd.Series(weather_data[k][1096:], index=weather_data["Date"][1096:], name=k)
        weather_train_ts.append(series)


    if not len(weather_data) == 2192:
        print weather, ': Data size: ', str(len(weather_data)),' does not match the expected size 2192 = 2*1096'




    return train_ts, test_ts, hourly_time, weather_train_ts, weather_test_ts

def process_csv_output(filename, start_date, ndays):

    ts = pd.read_csv(filename, sep=',', encoding='latin1') #csvread(filename, 1, 0);

    # generate hourly time stamps:
    #time_stamps = add_hours_to_dates(start_date, ndays, 24)
    time_stamps = range(24*1096)
    keys = ['Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5', 'Hour 6', 'Hour 7', 'Hour 8',
       'Hour 9', 'Hour 10', 'Hour 11', 'Hour 12', 'Hour 13', 'Hour 14',
       'Hour 15', 'Hour 16', 'Hour 17', 'Hour 18', 'Hour 19', 'Hour 20',
       'Hour 21', 'Hour 22', 'Hour 23', 'Hour 24']

    other = ts[['Day of the week', '1-workday, 2-Saturday, 3-Sunday, >4-untypical']]
    ts = pd.Series(ts[keys].as_matrix().reshape(24*1096), index=time_stamps)


    """
    if not len(ts) == 24*1096:
        print filename, ': Data size: ', str(len(ts)), ' does not match the expected size 26304 = 24x1096'
    """

    return ts, time_stamps, other


def add_hours_to_dates(start_date, ndays, hours):

    nticks = ndays*hours
    date_vec = pd.date_range(start_date, periods=nticks, freq='H')
    date_vec = pd.to_numeric(date_vec)

    return date_vec


def read_missing_value_dir(dirname):

    train_fns = glob.glob(os.path.join(dirname, 'train*'))
    test_fns = glob.glob(os.path.join(dirname, 'test*'))
    weather_fns = glob.glob(os.path.join(dirname, 'weatherdata*'))

    return train_fns, test_fns, weather_fns

'''
dirname = "../../code/data/EnergyWeatherTS/orig"
load_ts(dirname)
'''