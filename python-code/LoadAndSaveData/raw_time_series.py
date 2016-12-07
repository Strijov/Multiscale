from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline

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
    def __init__(self, data, request=1, history=1, name='', readme='', allow_empty=False,
                 as_floats=False, max_one_step=30):
        
        # Check inputs:
        # Some tests might need to process empty data without errors
        if not allow_empty:
            if len(data) == 0:
                raise ValueError("TsStruct.__init__: Data is an empty list")
            for ts in data:
                if ts.size == 0:
                    raise ValueError("TsStruct.__init__: ts {} is empty".format(ts.name))

        if request is None:
            request = 1

        if not isinstance(request, int) and not isinstance(request, float):
            raise TypeError("request should be int or float; got {}, type {}".format(request, type(request)))

        if history is not None and not isinstance(history, int) and not isinstance(history, float):
            raise TypeError("history should be int or float; got {}, type {}".format(history, type(history)))
                
        self.as_floats = as_floats
        self.max_one_step = max_one_step
        self.data, self.index_type = data_index_type(data, as_floats)
        self.original_index = [ts.index for ts in self.data]

        self.intervals = np.around(self.ts_frequencies(), decimals=5)
        self.one_step = assign_one_step_requests(self.intervals, self.as_floats, self.index_type, self.max_one_step)

        self.request = request
        self.history = history
        self.name = name
        self.readme = readme
        
    def to_floats(self):
        """
        Forces TsStruct instance to as_floats format.

        :return: None
        """
        if self.as_floats:
            return None
        request, history = self.request, self.history
        if not isinstance(request, int) and not isinstance(request, float):
            raise TypeError("request should be int or float; got {}, type {}".format(request, type(request)))

        if history is not None and not isinstance(history, int) and not isinstance(history, float):
            raise TypeError("history should be int or float; got {}, type {}".format(history, type(history)))
        
        self.as_floats = True
        if not self.index_type == 'int':
            self.data, self.index_type = data_index_type(self.data, self.as_floats)
            self.intervals = np.around(self.ts_frequencies(), decimals=5)
            self.one_step = assign_one_step_requests(self.intervals, as_floats=self.as_floats,
                                                     index_type=self.index_type, max_request=self.max_one_step)
            for i, idx in enumerate(self.original_index):
                self.original_index[i], _ = pd_time_stamps_to_floats(idx, self.index_type)
        

    def ts_frequencies(self):
        """
        :return: time intervals for each time series; intervals are floats
        :rtype: list
        """
        if self.as_floats:  #self.index_type in ["ns", "s", "h"]:
            return [min(np.diff(ts.index)) for ts in self.data]

        freqs = []
        if not isinstance(self.data[0].index[0], pd.tslib.Timestamp) and not self.index_type=="int":
            raise TypeError("Unexpected type of time indices; expected pd.tslib.Timestamp, got {}. "
                            "Index_type is {}"
                            .format(type(self.data[0].index[0]), self.index_type))
        for ts in self.data:
            index, _ = pd_time_stamps_to_floats(ts.index, self.index_type)
            freqs.append(min(np.diff(index)))

        return freqs

    def train_test_split(self, train_test_ratio=0.75):
        """
        Splits time series sequentially into train and test time series.
        ! Values of one_step are the same as those in the splitted TsStruct instance

        :param train_test_ratio: ratio of train objects to original ts length
        :type train_test_ratio: float
        :return: TsStructs with train and test time series
        :rtype: tuple
        """

        max_freq = np.argmin(self.intervals) #
        n_train = int(len(self.data[max_freq]) * train_test_ratio)
        max_train_index = self.data[max_freq].index[n_train]

        train_ts, test_ts = [], []
        original_index_train, original_index_test = [], []
        for i, ts in enumerate(self.data):
            train_idx = ts.index <= max_train_index
            test_idx = ts.index > max_train_index
            train_ts.append(ts[train_idx])
            test_ts.append(ts[test_idx])
            original_index_test.append(self.original_index[i][self.original_index[i] > max_train_index])
            original_index_train.append(self.original_index[i][self.original_index[i] <= max_train_index])

        train = TsStruct(train_ts, self.request, self.history, self.name, self.readme)
        train.one_step = self.one_step
        test = TsStruct(test_ts, self.request, self.history, self.name, self.readme)
        test.one_step = self.one_step

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

        if not max_history is None:
            self.truncate(max_history)
            return self.data

        # for index_type in ['h', 'ns', 's'] returns pd.Timestamps
        intervals = from_floats_to_index_type(self.intervals, self.index_type)
        if self.index_type == "int":
            common_t = set(np.around(self.data[0].index, decimals=5))
            common_t.add(np.around(self.data[0].index[-1] + intervals[0], decimals=5))
        else:
            common_t = set(self.data[0].index)
            common_t.add(self.data[0].index[-1] + intervals[0])

        for i, ts in enumerate(self.data[1:]):
            if self.index_type == "int":
                index_plus_1 = set(np.around(ts.index, decimals=5))
                index_plus_1.add(np.around(ts.index[-1] + intervals[i+1], decimals=5))
            else:
                index_plus_1 = set(ts.index)
                index_plus_1.add(ts.index[-1] + intervals[i + 1])
            common_t = common_t.intersection(index_plus_1)
            if len(common_t) == 0:
                self.interpolate_data()
                return self.align_time_series(max_history)

        min_end_T = max(common_t)
        max_start_T = min(common_t)

        for i, ts in enumerate(self.data):
            self.data[i] = ts.iloc[np.logical_and(ts.index < min_end_T, ts.index >= max_start_T)]

        return self.data

    def interpolate_data(self):
        """
        Is used when original indices do not intersect, or the intersection is too small.
        Defines a new uniform grid of time indices, which includes time steps from all the
        time series in the set. Overwrites data and intervals fields, but leaves history, request
        and one_step as they are

        :return: None
        """
        index = set(np.array(self.data[0].index))
        for ts in self.data:
            index.update(set(np.array(ts.index)))
        index = np.sort(list(index))
        common_step = np.min(np.diff(index))
        index = np.arange(index[0], index[-1], common_step)

        for i, ts in enumerate(self.data):
            s = InterpolatedUnivariateSpline(np.array(ts.index), np.array(ts.T), k=1)
            self.data[i] = pd.Series(s(index), index=index, name=ts.name)

        self.intervals = np.around(self.ts_frequencies(), decimals=5)


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
    
def data_index_type(data, as_float):
    # should be handled with care, pd.Timedelta tends to lose nanoseconds for som reason
    numeric_type_dict = {"int": 0, "ns": 1, "s": 2, "h": 3}
    index_type, numeric_type = [], []
    for ts in data:
        index_type.append(infer_frequency(ts.index))
        numeric_type.append(numeric_type_dict[index_type[-1]])

    if "int" in index_type and sum(numeric_type) > 0:
        raise TypeError("Data contains time series with both numeric and pd.Timestamp time indices")

    preferred_type = index_type[np.argmin(numeric_type)]

    if not as_float:
        return data, preferred_type

    for i, ts in enumerate(data):
        ts_index, _ = pd_time_stamps_to_floats(ts.index, preferred_type)
        data[i].index = ts_index

    return data, preferred_type
        
        
def infer_frequency(ts_index):
    if len(ts_index) < 2:
        raise ValueError(("ts.index is too short!  len(ts.index) = {}".format(len(ts_index))))
    
    if not isinstance(ts_index[0], pd.tslib.Timestamp):  # isinstance(ts.index, pd.DatetimeIndex):  #
        return "int"
    
    delta = ts_index[1] - ts_index[0]
    if delta._ms > 0 or delta._ns > 0:
        frequency = 'ns'
    elif delta._s > 0:
        frequency = 's'
    else:
        frequency = 'h'
        
    return frequency
    
    
def pd_time_stamps_to_floats(time_stamps_array, frequency):

    if len(time_stamps_array) < 2:
        print("ts.index is too short!")
        raise ValueError

    if not isinstance(time_stamps_array[0], pd.tslib.Timestamp):  # isinstance(ts.index, pd.DatetimeIndex):  #
        return np.array(time_stamps_array), frequency

    time_stamps_array = (time_stamps_array.to_datetime() - np.datetime64('1970-01-01T00:00:00Z')) \
                        / np.timedelta64(1, frequency)
    time_stamps_array = np.array(time_stamps_array)

    return time_stamps_array, frequency


def general_time_delta_to_float(td, freq):
    if freq == "int":
        if isinstance(td, pd.tslib.Timedelta):
            raise TypeError
        return td

    total = td.total_seconds()
    if freq == "ns":
        total *= 1e9
    elif freq == "s":
        pass
    elif freq == "h":
        total /= 3600.00
    else:
        print("Unsupported frequency type, should be 'ns, s, h or int, got {}".format(freq))
        raise ValueError

    return total


def from_floats_to_index_type(floats, freq):
    if freq == "int":
        if not isinstance(floats, float) and not isinstance(floats, int):
            raise TypeError("Inputs should be of type float or int, got {}".format(type(floats)))
        return floats

    time_deltas = pd.to_datetime(floats, unit=freq) - np.datetime64('1970-01-01T00:00:00Z')

    return time_deltas


def multiply_pd_time_delta(time_delta, multipl):
    res = pd.Timedelta(days=time_delta.days * multipl,
                       # hours=time_delta.hours * multipl,
                       seconds=time_delta.seconds * multipl,
                       microseconds=time_delta.microseconds * multipl,
                       nanoseconds=time_delta.nanoseconds * multipl)

    return res

def assign_one_step_requests(intervals, as_floats=True, index_type="s", max_request=1000):
    """
    Assigns request as the lowest common multiple of one step requests for each time series

    :param intervals: time deltas in unix format
    :type intervals: ndarray
    :param as_floats: if True, return request as floats, otherwise - a pd.Timedelta
    :type as_floats: bool
    :param index_type: defines frequency of the timedelta request (if as_floats is false)
    :type index_type: str
    :return: common request
    :rtype: pd.tslib.Timedelta, float
    """


    min_interval = min(intervals)
    intervals = intervals / min_interval

    # Define the length of the least common interval
    request = 1.0
    for intv in intervals:
        request = lcm(intv, request)

    request *= min_interval
    request = min([request, max_request])

    if as_floats or index_type=="int":
        return request

    return pd.to_timedelta(request, index_type)


def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)

def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b > TOL:
        a, b = b, a % b
    return a