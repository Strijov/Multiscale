import pandas as pd
import numpy as np

from LoadAndSaveData.load_time_series import TsStruct

PERIOD = 15
def create_sine_ts(n_ts = 3, n_req = 10, n_hist=20, max_length=5000, min_length = 200, period=PERIOD):
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


def create_random_ts(n_ts = 3, n_req = 10, n_hist=20, max_length=5000, min_length = 200):
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

def create_linear_ts(n_ts = 3, n_req = 10, n_hist=20, max_length=5000, min_length = 200):
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
        ts[n] = pd.Series(np.arange(end_time), index=index, name=str(n))

    ts = TsStruct(ts, n_req, n_hist, 'Sine', 'Artificial data for testing purposes')
    return ts