from __future__ import division
import os
import glob
import csv

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

from .raw_time_series import TsStruct

DATASET = "EnergyWeather"
DIRNAME = "{}/../../code/data/EnergyWeatherTS".format(os.path.dirname(os.path.abspath(__file__)))


def load_ts():
    """
    Loads energy consumption and weather time series

    :return: TsStructures with loaded data and reference names
    :rtype: list, list
    """

    dirnames = ["orig", "missing_value", "varying_rates"]

    ts_list = []
    name_list = []
    for dname in dirnames:
        ts, names = load_ts_by_dirname(DIRNAME, dname)
        ts_list.extend(ts)
        name_list.extend(names)

    return ts_list, name_list


def load_ts_by_dirname(dirname, folder_name):
    """
    Load time series of specific type from EnergyWeather dataset

    :param dirname: path to directory with input data, ends with "orig", "missing_vale" or "varying_rates"
    :type dirname: string
    :param folder_name:
    :type folder_name:
    :return: sStructures with loaded data and reference names
    :rtype: list, list
    """
    dirname = os.path.abspath(dirname + os.sep + folder_name)

    # folders = os.path.split(dirname) #, os.sep
    # folder_name = folders[-1]
    # if len(folder_name) == 0:
    #     folder_name = folders[-2]

    readme = {'orig': 'Original time energy-weather series',
              'missing_value': 'Energy-weather time series with artificially inserted missing values',
              'varying_rates': 'Energy-weather time series with varying sampling rate'}
    readme = readme[folder_name]

    filename_train, filename_test, filename_weather = read_missing_value_dir(dirname)

    ts = []
    names = []
    for i in range(len(filename_train)):
        train_ts, test_ts, train_weather, test_weather = _load_train_test_csv(filename_train[i], filename_test[i],
                                                                             filename_weather[i])
        # train_ts, test_ts, train_weather, test_weather = load_train_test_weather(
        #     filename_train[i], filename_test[i], filename_weather[i])

        request = 1  #train_ts.index[24] - train_ts.index[0]  # by default, forecasts are requested for one day ahead
        history = 7  #train_ts.index[7 * 24] - train_ts.index[0]  # by default, ts history is one week

        train_ts = [train_ts]
        train_ts.extend(train_weather)
        test_ts = [test_ts]
        test_ts.extend(test_weather)

        name_train, _ = os.path.splitext(os.path.split(filename_train[i])[1])
        name_train = DATASET + '_' + folder_name + '_' + name_train
        name_test, _ = os.path.splitext(os.path.split(filename_test[i])[1])
        name_test = DATASET + '_' + folder_name + '_' + name_test
        names.extend([name_train, name_test])
        ts_train, ts_test = TsStruct(train_ts, request, history, name_train, readme), \
                            TsStruct(test_ts, request, history, name_test, readme)
        ts.append(ts_train)
        ts.append(ts_test)

    return ts, names

def _load_train_test_csv(train, test, weather):
    """
    Loads raw data from csv. The dataset contains 3-year daily observations for weather data and horly observations for
    electricity consumption data.
    Weather data headers: 'Date', 'Max Temperature', 'Min Temperature', 'Precipitation', 'Wind', 'Relative Humidity', 'Solar'
    Electricity data headers: 'Day of the week', '1-workday, 2-Saturday, 3-Sunday, >4-untypical', 'Hour 1', ..., 'Hour 24'

    :param train: filename for train data
    :type train: string
    :param test: filename for test data
    :type test: string
    :param weather: filename for weather data, both train and test
    :type weather:
    :return: train_set, test_set, train_weather, test_weather
    :rtype: pandas.Series
    """
    # reading CSV file
    reader = csv.reader(open(train, 'r'), delimiter=',')
    train = np.array(list(reader))
    reader = csv.reader(open(test, 'r'), delimiter=',')
    test = np.array(list(reader))
    reader = csv.reader(open(weather, 'r'), delimiter=',')
    weather = np.array(list(reader))

    labels = test[0, :]

    train = np.delete(train, [0], 0) # remove labels
    train_time = train[:, 0]
    # weekday_train = train[:, 1]
    # day_type_train = train[:, 2]
    train_set = np.delete(train, [0, 1, 2], 1) # remove date, weekday and day type columns

    test = np.delete(test, [0], 0)
    test_time = test[:, 0]
    # weekday_test = test[:, 1]
    # day_type_test = test[:, 2]
    test_set = np.delete(test, [0, 1, 2], 1)


    train_set = np.array(train_set, dtype='float32')
    train_set = np.reshape(train_set, (1096*24))
    test_set = np.array(test_set, dtype='float32')
    test_set = np.reshape(test_set, (1096*24))

    #print weather[0, :]
    if "Longitude" in weather[0, :]:
        weather_labels = list(weather[0, 4:])
        idx_del = range(4)
    else:
        weather_labels = list(weather[0, 1:])
        idx_del = [0]
    weather = np.delete(weather, [0], 0)
    weather_time = weather[:, 0]
    weather = np.delete(weather, idx_del, 1) # discard 'Date' 'Longitude' 'Latitude' 'Elevation'
    weather = np.array(weather, dtype="float32")

    # convert time to unix format:
    if "/" in weather_time[0]:
        weather_time = [datetime.strptime(str_d, "%m/%d/%Y") for str_d in weather_time]
        train_time = [datetime.strptime(str_d, "%Y%m%d.0") for str_d in train_time]
        test_time = [datetime.strptime(str_d, "%Y%m%d.0") for str_d in test_time]
    else:
        weather_time = [datetime.strptime(str_d, "%d.%m.%Y") for str_d in weather_time]
        train_time = [datetime.strptime(str_d, "%Y%m%d") for str_d in train_time]
        test_time = [datetime.strptime(str_d, "%Y%m%d") for str_d in test_time]


    min_date = min(weather_time + train_time + test_time)
    #weather_time = [(dt - min_date).total_seconds() for dt in weather_time]
    train_weather_time = weather_time[:1096]
    test_weather_time = weather_time[1096:]

    train_time = _add_hours_to_dates(train_time)
    test_time = _add_hours_to_dates(test_time)
    #test_time = [(dt - min_date).total_seconds() for dt in test_time]
    #train_time = [(dt - min_date).total_seconds() for dt in train_time]


    train_set = pd.Series(train_set, index=train_time, name="Energy")
    test_set = pd.Series(test_set, index=test_time, name="Energy")

    train_weather, test_weather = [0]*len(weather_labels), [0]*len(weather_labels)
    for i in range(len(weather_labels)):
        train_weather[i] = pd.Series(weather[:1096, i], index=train_weather_time, name=weather_labels[i])
        test_weather[i] = pd.Series(weather[1096:, i], index=test_weather_time, name=weather_labels[i])
        #print "nans:", weather_labels[i], np.sum(np.isnan(weather[:, i])), np.sum(train_weather[i].isnull()), np.sum(test_weather[i].isnull())

    return train_set, test_set, train_weather, test_weather

def _add_hours_to_dates(dates):
    """
    Adds hourly time ticks to daily ticks

    :param dates: observed dates
    :type dates: list
    :return:
    :rtype: list
    """

    date_vec = []
    for date in dates:
        date_vec.extend([date + timedelta(hours=x) for x in range(24)])

    return date_vec



def _load_train_test_weather(train, test, weather):
    """

    :param train, test, weather: filenames
    :return: [train_ts, test_ts, time_train, time_test, train_weather, test_weather]
    """


    _, extension = os.path.splitext(train)


    weather_data = pd.read_csv(weather, sep=',', encoding='latin1')  # csvread(weather, 1, 1);
    weather_data = weather_data[['Date', 'Max Temperature', 'Min Temperature',
                                 'Precipitation', 'Wind', 'Relative Humidity', 'Solar']]
    weather_data['Date'] = pd.to_datetime(weather_data['Date'])
    test_ts, test_time,_ = _process_csv_output(test, weather_data['Date'][1096], 1096)
    train_ts, train_time,_ = _process_csv_output(train, weather_data['Date'][0], 1096)
    hourly_time =  train_time + test_time # for pd.time use # train_time.append(test_time)
    weather_data['Date'] = range(0, 24*1096*2, 24) #pd.to_numeric(weather_data['Date'])

    weather_train_ts = []
    weather_test_ts = []

    for k in ['Max Temperature', 'Min Temperature', 'Precipitation', 'Wind', 'Relative Humidity', 'Solar']:
        series = pd.Series(weather_data[k][:1096], index=weather_data["Date"][:1096], name=k)
        weather_train_ts.append(series)
        series = pd.Series(weather_data[k][1096:], index=weather_data["Date"][1096:], name=k)
        weather_test_ts.append(series)


    # if not len(weather_data) == 2192:
    #     print weather, ': Data size: ', str(len(weather_data)),' does not match the expected size 2192 = 2*1096'




    return train_ts, test_ts, weather_train_ts, weather_test_ts

def _process_csv_output(filename, start_date, ndays):

    ts = pd.read_csv(filename, sep=',', encoding='latin1') #csvread(filename, 1, 0);

    # generate hourly time stamps:
    #time_stamps = add_hours_to_dates(start_date, ndays, 24)
    time_stamps = range(24*1096)
    keys = ['Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5', 'Hour 6', 'Hour 7', 'Hour 8',
       'Hour 9', 'Hour 10', 'Hour 11', 'Hour 12', 'Hour 13', 'Hour 14',
       'Hour 15', 'Hour 16', 'Hour 17', 'Hour 18', 'Hour 19', 'Hour 20',
       'Hour 21', 'Hour 22', 'Hour 23', 'Hour 24']

    other = ts[['Day of the week', '1-workday, 2-Saturday, 3-Sunday, >4-untypical']]
    ts = pd.Series(ts[keys].as_matrix().reshape(24*1096), index=time_stamps, name="Energy")



    return ts, time_stamps, other



def read_missing_value_dir(dirname):

    train_fns = glob.glob(os.path.join(dirname, 'train*'))
    test_fns = glob.glob(os.path.join(dirname, 'test*'))
    weather_fns = glob.glob(os.path.join(dirname, 'weatherdata*'))

    return train_fns, test_fns, weather_fns

