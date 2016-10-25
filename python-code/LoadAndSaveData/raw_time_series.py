import numpy as np
import pandas as pd

class TsStruct():
    """ This structure stores input data. The fields are:

    :param data: input time series, each is pandas.Series
    :type data: list
    :param request: Time interval requested for forecast
    :type request: int\ time delta ? #FIXIT
    :param history: Time interval,  to define number of historical points.
    :type history: int\ time delta ? #FIXIT
    :param name: Dataset name
    :type name: string
    :param readme: Dataset info
    :type readme: string
    """
    def __init__(self, data, request, history, name, readme):
        self.data = data
        self.request = request
        self.history = history
        self.name = name
        self.readme = readme
        self.intervals = self.ts_frequencies()

        if request is None:
            self.request = assign_one_step_requests(self.intervals)


    def ts_frequencies(self):
        freqs = [min(np.diff(ts.index)) for ts in self.data]

        return freqs

    def train_test_split(self, train_test_ratio=0.75):

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


    def align_time_series(self):
        """
        Truncates time series in self.data so that the end points of all times series belong to the same requested interval
        :return: truncated time series in pd.Series format
        :rtype: list
        """


        min_end_T = min([ts.index[-1] for ts in self.data]) # find earliest end-point index
        max_start_T = max([ts.index[0] for ts in self.data]) # find latest start-point index
        for i, ts in enumerate(self.data):
            ts = ts[ts.index >= max_start_T]
            self.data[i] = ts[ts.index < min_end_T + self.request]

        return self.data



    def summarize_ts(self, latex=False):
        """
        Returns basic statistics for each time series in self.data
        :param latex: if True, returns latex string with results in table
        :type latex: bool
        :return: pd.DataFrame or latex string with ts statistics
        """

        column_names = ["N. obs.", "Min", "Max", "T. min", "T.max", "T. delta", "Nans"]
        res = []
        names = []
        for ts in self.data:
            stats = pd.DataFrame(ts).describe()
            names.append(ts.name)
            res.append([ts.count(), ts.min(), ts.max(), ts.index.to_datetime()[0], ts.index.to_datetime()[-1],
                       min(np.diff(ts.index.to_datetime())), sum(np.isnan(ts.as_matrix()))])


        res = pd.DataFrame(res, index=names, columns=column_names)
        if latex:
            res = res.to_latex()
            res = "{\\tiny \n " + res + "\n }\n \\bigskip \n"

        return res

def assign_one_step_requests(intervals):

    request = 1
    for intv in intervals:
        request = lcm(intv, request)

    return request


def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)

def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:
        a, b = b, a % b
    return a