import unittest
import pandas as pd
import datetime

from RegressionMatrix import random_data
from LoadAndSaveData.load_time_series import TsStruct
from LoadAndSaveData import raw_time_series

class TestTsStruct(unittest.TestCase):

    def test_truncation(self):
        """ Check that time series start from the same point after truncation """

        input_ts = random_data.create_random_ts(time_delta=[1, 1, 10]) # this returns time series that start from 0

        # shift time series
        input_ts.data[0] = input_ts.data[0][2:]
        input_ts.data[2] = input_ts.data[2][5:]
        input_ts.align_time_series() # align and truncate time series


        self.assertTrue(input_ts.data[0].index[0] == input_ts.data[1].index[0])
        self.assertTrue(input_ts.data[1].index[0] == input_ts.data[2].index[0])




    def test_double_truncation(self):
        """ Check that re-truncation does not have any effect """

        for i in range(10):
            input_ts = random_data.create_random_ts(time_delta=[1, 1, 10])

            # Truncate time series and remember resultant sizes
            input_ts.align_time_series()
            sizes1 = [ts.size for ts in input_ts.data]

            # Repeat:
            input_ts.align_time_series()
            input_ts.align_time_series()
            sizes2 = [ts.size for ts in input_ts.data]

            # Compare results
            self.assertEqual(sizes1, sizes2)


    def test_empty_input(self):
        """ Check response to empty input """
        input_ts = random_data.create_random_ts()
        with self.assertRaises(ValueError) as e:
            TsStruct([], input_ts.request, input_ts.history, input_ts.name, input_ts.readme)

        self.assertTrue('empty list' in e.exception.message)

        data = input_ts.data[0][:0]
        with self.assertRaises(ValueError) as e:
            TsStruct([data], input_ts.request, input_ts.history, input_ts.name, input_ts.readme)


    def test_request_assignment(self):
        """ Tests for 'assign_one_step_requests' method with time in floats"""

        # time deltas greater than one:
        input_ts = random_data.create_random_ts(time_delta=[1000, 1000, 2000], dt_index=False)
        intervals = input_ts.intervals
        self.assertTrue((intervals <= input_ts.request).all(),
                        "Request is smaller than at least one of the time intervals (td < 1.0)")
        self.assertTrue(input_ts.request == 0.001, "Request should be equal to 0.001, got {}".format(input_ts.request))

        # time deltas less than one:
        input_ts = random_data.create_random_ts(time_delta=[1.0, 0.5, 0.2], dt_index=False)
        intervals = input_ts.intervals
        self.assertTrue((intervals <= input_ts.request).all(),
                        "Request is smaller than at least one of the time intervals (td > 1.0)")
        self.assertTrue(input_ts.request == 10.0, "Request should be equal to 10.0, got {}".format(input_ts.request))

        # time deltas less than one:
        input_ts = random_data.create_random_ts(time_delta=[10, 1.0, 0.1], dt_index=False)
        intervals = input_ts.intervals
        self.assertTrue((intervals <= input_ts.request).all(),
                        "Request is smaller than at least one of the time intervals (mixed case)")
        self.assertTrue(input_ts.request == 10.0, "Request should be equal to 10.0, got {}".format(input_ts.request))

    def test_request_assignment_td(self):
        """ Tests for 'assign_one_step_requests' method with time in timedelta format"""

        # time deltas greater than one:
        input_ts = random_data.create_random_ts(time_delta=[1000, 1000, 2000], dt_index=True)
        intervals = input_ts.intervals
        fl_request = raw_time_series.assign_one_step_requests(intervals, as_timedelta=False)
        expected_fl_request = 0.001
        expected_td_request = _float_to_td_value(expected_fl_request)
        self.assertTrue((intervals <= fl_request).all(),
                        "Request is smaller than at least one of the time intervals (td > 1.0)")
        # self.assertTrue(input_ts.request == expected_td_request,
        #                 "Request should be equal to {}, got {}".format(expected_td_request, input_ts.request))
        self.assertTrue(fl_request == expected_td_request,
                        "Fl. Request should be equal to {}, got {}".format(expected_td_request, fl_request))

        # time deltas less than one:
        input_ts = random_data.create_random_ts(time_delta=[1.0, 0.5, 0.2], dt_index=True)
        intervals = input_ts.intervals
        fl_request = raw_time_series.assign_one_step_requests(intervals, as_timedelta=False)
        expected_fl_request = 10.0
        expected_td_request = _float_to_td_value(expected_fl_request)
        self.assertTrue((intervals <= fl_request).all(),
                        "Request is smaller than at least one of the time intervals (td < 1.0)")
        # self.assertTrue(input_ts.request == expected_td_request,
        #                 "Request should be equal to {}, got {}".format(expected_td_request, input_ts.request))
        self.assertTrue(fl_request == expected_td_request,
                        "Fl. Request should be equal to {}, got {}".format(expected_td_request, fl_request))

        # time deltas less than one:
        input_ts = random_data.create_random_ts(time_delta=[10, 1.0, 0.1], dt_index=True)
        intervals = input_ts.intervals
        fl_request = raw_time_series.assign_one_step_requests(intervals, as_timedelta=False)
        expected_fl_request = 10.0
        expected_td_request = _float_to_td_value(expected_fl_request)
        self.assertTrue((intervals <= fl_request).all(),
                        "Request is smaller than at least one of the time intervals mixed case)")
        # self.assertTrue(input_ts.request == expected_td_request,
        #                 "Request should be equal to {}, got {}".format(expected_td_request, input_ts.request))
        self.assertTrue(fl_request == expected_td_request,
                        "Fl. Request should be equal to {}, got {}".format(expected_td_request, fl_request))


def _float_to_td_value(expected_fl_request):
    return pd.to_datetime(datetime.datetime.fromtimestamp(2 * expected_fl_request)).value - pd.to_datetime(
        datetime.datetime.fromtimestamp(expected_fl_request)).value

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTsStruct)
    unittest.TextTestRunner(verbosity=2).run(suite)