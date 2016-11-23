from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd

TOL = pow(10, -10)

class TsStruct():
    """ This structure stores input data. The fields are:

    :param data: input time series, each is pandas.Series
    :type data: list
    :param request: Number of one-step intervals requested for forecast
    :type request: int
    :param history: Number of one-step intervals to forecast.
    :type history: int
    :param name: Dataset name
    :type name: string
    :param readme: Dataset info
    :type readme: string
    """
    def __init__(self, data, request, history, name, readme, allow_empty=False):
        self.data = data

        if not allow_empty:
            if len(data) == 0:
                raise ValueError("TsStruct.__init__: Data is an empty list")
            for ts in data:
                if ts.size == 0:
                    raise ValueError("TsStruct.__init__: ts {} is empty".format(ts.name))

        self.intervals = np.around(self.ts_frequencies(), decimals=5)
        self.one_step = assign_one_step_requests(self.intervals, isinstance(self.data[0].index[0], pd.tslib.Timestamp))

        if request is None:
            request = 1

        self.request = request
        self.history = history
        self.name = name
        self.readme = readme





    def ts_frequencies(self):

        if isinstance(self.data[0].index[0], pd.tslib.Timestamp):
            freqs = []
            for ts in self.data:
                index = [ts.index[i].value for i in range(len(ts))]
                freqs.append(min(np.diff(index)))
            return freqs

        freqs = [min(np.diff(ts.index)) for ts in self.data]

        return freqs

    def train_test_split(self, train_test_ratio=0.75):
        """
        Splits time series sequentially into train and test time series

        :param train_test_ratio: ratio of train objects to original ts length
        :type train_test_ratio: float
        :return: TsStructs with train and test time series
        :rtype: tuple
        """

        max_freq = np.argmin(self.intervals) #
        n_train = int(len(self.data[max_freq]) * train_test_ratio)
        max_train_index = self.data[max_freq].index[n_train]

        train_ts, test_ts = [], []
        for ts in self.data:
            train_idx = ts.index <= max_train_index
            test_idx = ts.index > max_train_index
            train_ts.append(ts[train_idx])
            test_ts.append(ts[test_idx])
        train = TsStruct(train_ts, self.request, self.history, self.name, self.readme)
        test = TsStruct(test_ts, self.request, self.history, self.name, self.readme)

        return train, test


    def replace_nans(self):
        for i, ts in enumerate(self.data):
            if not np.isnan(ts).any():
                continue

            print("Filling NaNs for TS", ts.name)
            if np.isnan(ts).all():
                print("All inputs are NaN", "replacing with zeros")
                self.data[i] = pd.Series(np.zeros_like(ts), index=ts.index, name=ts.name)
                continue

            ts_prop = pd.Series(ts).fillna(method="pad")
            ts_back = pd.Series(ts_prop).fillna(method="bfill")
            self.data[i] = ts_back  # (ts_back + ts_prop)[pd.isnull(ts)] / 2




    def truncate(self, max_history=50000, max_total = None):
        """
        Truncate time series so that number of observations in any time series or\and in toltal do not exceed the given values

        :param max_history: Max number of observations per time series
        :type max_history: int
        :param max_history: Max total number of observations in the set (ignored!)
        :type max_history: int
        :return:
        :rtype:
        """

        for i, ts in enumerate(self.data):
            self.data[i] = ts.iloc[-max_history:]

        self.align_time_series()


    def align_time_series(self, max_history=None):
        """
        Truncates time series in self.data so that the end points of all times series belong to the same requested interval

        :return: truncated time series in pd.Series format
        :rtype: list
        """


        #min_end_T = min([ts.index[-1] for ts in self.data]) # find earliest end-point index
        #max_start_T = max([ts.index[0] for ts in self.data]) # find latest start-point index

        if not max_history is None:
            self.truncate(max_history)
            return self.data

        common_T = set(np.around(self.data[0].index, decimals=5))
        common_T.add(np.around(self.data[0].index[-1] + self.intervals[0], decimals=5))
        for i, ts in enumerate(self.data[1:]):
            index_plus_1 = set(np.around(ts.index, decimals=5))
            index_plus_1.add(np.around(ts.index[-1] + self.intervals[i+1], decimals=5))
            #print(max(index_plus_1) - max(common_T))
            common_T = common_T.intersection(index_plus_1)

            #ending_T = ending_T.intersection(set(ts.index[np.logical_and(ts.index <= min_end_T, ts.index >= min_end_T - self.request)]))
            #self.data[i] = ts.iloc[np.logical_and(ts.index < min_end_T + self.request, ts.index >= max_start_T)]
            #self.data[i] = ts.iloc[ts.index < min_end_T + self.request]



        min_end_T = max(common_T)
        max_start_T = min(common_T)

        for i, ts in enumerate(self.data):
            self.data[i] = ts.iloc[np.logical_and(ts.index < min_end_T, ts.index >= max_start_T)]



        return self.data



    def summarize_ts(self, latex=False):
        """
        Returns basic statistics for each time series in self.data
        :param latex: if True, returns latex string with results in table
        :type latex: bool
        :return: pd.DataFrame or latex string with ts statistics
        """

        column_names = ["N. obs.", "Min", "Max", "T. min", "T.max", "T. delta", "Nans %"]
        res = []
        names = []
        for ts in self.data:
            names.append(ts.name)
            if isinstance(ts.index[0], pd.tslib.Timestamp):
                ts_min = ts.index[0]
                ts_max = ts.index[-1]
                ts_delta = min(ts.index[1:] - ts.index[:-1]) - pd.to_datetime("1970-01-01")
            else:
                ts_min = pd.to_datetime(ts.index, unit="s")[0]
                ts_max = pd.to_datetime(ts.index, unit="s")[-1]
                ts_delta = pd.to_datetime(min(np.diff(ts.index)), unit="s") - pd.to_datetime("1970-01-01")
            res.append([ts.count(), ts.min(), ts.max(), ts_min, ts_max, ts_delta, np.mean(np.isnan(ts.as_matrix()))*100])


        res = pd.DataFrame(res, index=names, columns=column_names)
        if latex:
            res = res.to_latex()
            res = "{\\tiny \n " + res + "\n }\n \\bigskip \n"

        return res

def assign_one_step_requests(intervals, as_timedelta=False):
    """
    Assigns request as the lowest common multiple of one step requests for each time series

    :param intervals: time deltas in unix format
    :type intervals: ndarray
    :return: common request
    :rtype: time delta, unix format
    """

    request = 1
    min_interval = min(intervals)
    intervals = intervals / min_interval

    for intv in intervals:
        request = lcm(intv, request)

    request *= min_interval

    if as_timedelta:
        request = pd.to_timedelta(request)

    return request


def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)

def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b > TOL:
        a, b = b, a % b
    return a