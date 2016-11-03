import pandas as pd
import numpy as np

from LoadAndSaveData.load_time_series import TsStruct

PERIOD = 15
def create_sine_ts(n_ts=3, n_req=10, n_hist=20, max_length=5000, min_length = 200, period=PERIOD):
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

    index = np.arange(end_time)  # frequency is the same for all FIXIT
    ts = [0] * n_ts
    for n in range(n_ts):
        ts[n] = pd.Series(np.sin(index*2*np.pi/period) + np.random.randn(len(index))*0.2, index=index, name=str(n))

    ts = TsStruct(ts, n_req, n_hist, 'Sine', 'Artificial data for testing purposes')
    return ts


def create_random_ts(n_ts=3, n_req=10, n_hist=20, max_length=5000, min_length = 200):
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
        :return: Data structure
        :rtype: TsStruct
        """

    end_time = np.random.randint(min_length, max_length + 1)
    index = range(end_time)  # frequency is the same for all FIXIT
    ts = [0] * n_ts
    for n in range(n_ts):
        ts[n] = pd.Series(np.random.randn(len(index))*2, index=index, name=str(n))

    ts = TsStruct(ts, n_req, n_hist, 'Sine', 'Artificial data for testing purposes')
    return ts

def create_linear_ts(n_ts=3, n_req=10, n_hist=20, max_length=5000, min_length = 200, slope=1.0):
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
    index = range(end_time)  # frequency is the same for all FIXIT
    ts = [0] * n_ts
    for n in range(n_ts):
        ts[n] = pd.Series(np.arange(end_time) * slope, index=index, name=str(n))

    ts = TsStruct(ts, n_req, n_hist, 'Sine', 'Artificial data for testing purposes')
    return ts


def create_iot_data(n_ts=3, n_req=10, n_hist=20, max_length=5000, min_length=200, slope=0.001,
                    non_zero_ratio=0.01, signal_to_noize=5, trend_noise=0.1):

    ts_struct = create_linear_ts(n_ts, n_req, n_hist, max_length, min_length, slope)
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

def create_iot_data_poisson(n_ts=3, n_req=10, n_hist=20, max_length=5000, min_length=200, slope=0.001,
                    non_zero_ratio=0.001, signal_to_noize=5, trend_noise=0.1):


    ts_struct = create_linear_ts(n_ts, n_req, n_hist, max_length, min_length, slope)

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




