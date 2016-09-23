import pandas as pd
import numpy as np
from collections import namedtuple
tsStruct = namedtuple('tsStruct', 'time data norm_div norm_subt request history name readme')

def create_sine_ts(n_ts = 3, n_req = 10, n_hist=20, max_length=5000, min_length = 200):

    end_time = np.random.randint(min_length, max_length + 1)
    index = range(end_time)  # frequency is the same for all FIXIT
    ts = [0] * n_ts
    for n in range(n_ts):
        ts[n] = pd.Series(np.sin(index) + np.random.randn(len(index))*0.2, index=index, name=str(n))

    ts = tsStruct(index, ts, 1, 0, n_req, n_hist, 'Sine', 'Artificial data for testing purposes')
    return ts


def create_random_ts(n_ts = 3, n_req = 10, n_hist=20, max_length=5000, min_length = 200):

    end_time = np.random.randint(min_length, max_length + 1)
    index = range(end_time)  # frequency is the same for all FIXIT
    ts = [0] * n_ts
    for n in range(n_ts):
        ts[n] = pd.Series(np.random.randn(len(index))*2, index=index, name=str(n))

    ts = tsStruct(index, ts, 1, 0, n_req, n_hist, 'Sine', 'Artificial data for testing purposes')
    return ts

def create_linear_ts(n_ts = 3, n_req = 10, n_hist=20, max_length=5000, min_length = 200):

    end_time = np.random.randint(min_length, max_length + 1)
    index = range(end_time)  # frequency is the same for all FIXIT
    ts = [0] * n_ts
    for n in range(n_ts):
        ts[n] = pd.Series(np.arange(end_time), index=index, name=str(n))

    ts = tsStruct(index, ts, 1, 0, n_req, n_hist, 'Sine', 'Artificial data for testing purposes')
    return ts