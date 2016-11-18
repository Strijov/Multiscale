import pandas as pd
import numpy as np
import datetime

from LoadAndSaveData.load_time_series import TsStruct

PERIOD = 15
def create_sine_ts(n_ts=3, n_req=10, n_hist=20, max_length=5000, min_length=200, period=PERIOD, dt_index=False, allow_empty=True):
    """
    Creates artificial "Multiscale" data (noised sines)

    :param n_ts: number of time series in the set
    :type n_ts: int
    :param n_req: Request (time interval)
    :type n_req: time delta#FIXIT
    :param n_hist: History (time interval)
    :type n_hist: time delta#FIXIT
    :param max_length: maximum length of time series in the set
    :type max_length:  int
    :param min_length: minimum length of time series in the set
    :type min_length: int
    :return: Data structure
    :rtype: TsStruct
    """

    end_time = np.random.randint(min_length, max_length + 1)

    ts = [0] * n_ts
    for n in range(n_ts):
        index = create_index(0, end_time, npoints_per_step=1, dt_index=dt_index)
        ts[n] = pd.Series(np.sin(index*2*np.pi/period) + np.random.randn(len(index))*0.2, index=index, name=str(n))

    ts = TsStruct(ts, n_req, n_hist, 'Sine', 'Artificial data for testing purposes', allow_empty)
    return ts


def create_random_ts(n_ts=3, n_req=None, n_hist=20, max_length=5000, min_length=200, max_freq=1, time_delta=None,
                     dt_index=False, allow_empty=True):
    """
        Creates random "Multiscale" data

        :param n_ts: number of time series in the set
        :type n_ts: int
        :param n_req: Request (time interval)
        :type n_req: time delta#FIXIT
        :param n_hist: History (time interval)
        :type n_hist: time delta#FIXIT
        :param max_length: maximum length of time series in the set
        :type max_length:  int
        :param min_length: minimum length of time series in the set
        :type min_length: int
        :param max_freq: specifies maximum frequency, times of basis time delta (1)
        :type max_freq: int
        :return: Data structure
        :rtype: TsStruct
        """


    end_time = np.random.randint(min_length, max_length + 1, n_ts)
    if not time_delta is None:
        n_ts = len(time_delta)
    if time_delta is None:
        time_delta = np.random.randint(1, max_freq + 1, n_ts)
    ts = [0] * n_ts
    for n in range(n_ts):
        index = create_index(0, end_time[n], time_delta[n], dt_index)   #list(range(0, end_time, time_delta[n]))
        ts[n] = pd.Series(np.random.randn(len(index))*2, index=index, name=str(n))

    ts = TsStruct(ts, n_req, n_hist, 'Sine', 'Artificial data for testing purposes', allow_empty)
    return ts

def create_index(start_time=0, end_time=500, npoints_per_step=1, dt_index=False):

    step_size = 1.0
    if npoints_per_step < 1.0:
        step_size = 1.0 / npoints_per_step
        npoints_per_step = 1.0

    step = np.linspace(0, step_size, npoints_per_step + 1)[:-1]
    index = []
    for i in np.arange(start_time, end_time, step_size):
        index.append(step + i)

    index = np.hstack(index)
    index = np.around(index, decimals=5)
    if dt_index:
        index = _index_to_datetime(index)

    if np.unique(index).size < len(index):
        print("Time stamps are not unique")
    #index = np.arange(start_time, end_time, 1.0 / npoints_per_step)
    return index

def _index_to_datetime(index):

    #print("Type of indx {}".format(type(index[0])))
    if isinstance(index[0], datetime.timedelta):
        return index

    new_index = []
    for i, t in enumerate(index):
        new_index.append(datetime.datetime.fromtimestamp(t)) #.strftime('%Y-%m-%d %H:%M:%S')

    return new_index


def create_linear_ts(n_ts=3, n_req=10, n_hist=20, max_length=5000, min_length = 200, slope=1.0, dt_index=False, allow_empty=True):
    """
        Creates artificial "Multiscale" data, linear ts

        :param n_ts: number of time series in the set
        :type n_ts: int
        :param n_req: Request (time interval)
        :type n_req: time delta#FIXIT
        :param n_hist: History (time interval)
        :type n_hist: time delta#FIXIT
        :param max_length: maximum length of time series in the set
        :type max_length:  int
        :param min_length: minimum length of time series in the set
        :type min_length: int
        :return: Data structure
        :rtype: TsStruct
        """

    end_time = np.random.randint(min_length, max_length + 1)
    ts = [0] * n_ts
    for n in range(n_ts):
        index = create_index(0, end_time, npoints_per_step=1, dt_index=dt_index)
        ts[n] = pd.Series(np.arange(end_time) * slope, index=index, name=str(n))

    ts = TsStruct(ts, n_req, n_hist, 'Sine', 'Artificial data for testing purposes', allow_empty)
    return ts


def create_iot_data(n_ts=3, n_req=10, n_hist=20, max_length=5000, min_length=200, slope=0.001,
                    non_zero_ratio=0.01, signal_to_noize=5, trend_noise=0.1, dt_index=False, allow_empty=True):

    ts_struct = create_linear_ts(n_ts, n_req, n_hist, max_length, min_length, slope, dt_index, allow_empty)
    for i, ts in enumerate(ts_struct.data):
        ts += np.random.randn(ts.shape[0])* trend_noise
        signal = np.zeros_like(ts)
        non_zero_idx = np.random.rand(ts.shape[0]) < non_zero_ratio
        signal[non_zero_idx] = (np.random.rand() + 1) * signal_to_noize
        ts += signal
        ts_struct.data[i] = ts

    ts_struct.readme = "Artificial Iot-like data"
    ts_struct.name = "IoT"

    return ts_struct

def create_iot_data_poisson(n_ts=3, n_req=10, n_hist=20, max_length=10000, min_length=2000, slope=0.001,
                    non_zero_ratio=0.001, signal_to_noize=5, trend_noise=0.1, dt_index=False, allow_empty=True):


    ts_struct = create_linear_ts(n_ts, n_req, n_hist, max_length, min_length, slope, dt_index, allow_empty)

    for i, ts in enumerate(ts_struct.data):
        ts += np.random.randn(ts.shape[0])* trend_noise
        lambda_ = int(len(ts) * non_zero_ratio)
        intervals = np.random.poisson(lam=lambda_, size=len(ts))
        non_zero_idx = np.cumsum(intervals)
        non_zero_idx = non_zero_idx[non_zero_idx < len(ts)]
        signal = np.zeros_like(ts)
        signal[non_zero_idx] = (np.random.rand() + 1) * signal_to_noize
        ts += signal
        ts_struct.data[i] = ts

    ts_struct.readme = "Artificial Iot-like data, Poisson-based"
    ts_struct.name = "IoT"

    return ts_struct




