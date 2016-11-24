import unittest
import os
import numpy as np

from RegressionMatrix import random_data
from LoadAndSaveData import write_data_to_iot_format, get_iot_data, load_time_series

FILE_NAME = "TestIot.csv"
TOL = pow(10, -10)


class TestIotData(unittest.TestCase):

    def test_read_lines(self):
        """
        Check that reading by line works correctly: write random data into file, then read by line and compare
        results to original time series
        """
        print("\nRunning test_read_lines\n")
        input_ts = random_data.create_random_ts(n_ts=3, n_req=1, n_hist=5, max_length=2000, min_length=200)
        write_data_to_iot_format.write_ts(input_ts, FILE_NAME)

        for i, ts in enumerate(input_ts.data):
            data, metric_ids, host_ids, header_names = get_iot_data.get_data(FILE_NAME, [i+2], True)
            dataset = host_ids.keys()[0]
            converted_ts = load_time_series.from_iot_to_struct(data, host_ids[dataset], dataset)
            self.assertTrue((abs(ts - converted_ts.data[0]) < TOL).all(),
                            "Maximum difference {} between ts values exceeded tolerance {}".
                            format(max(abs(ts - converted_ts.data[0])), TOL))
            self.assertTrue(ts.name == converted_ts.data[0].name)
            self.assertTrue(input_ts.name == dataset)

        os.remove(FILE_NAME)

    def test_read_and_write(self):
        """ Writes random data into file, then reads from it and compares results """
        print("\nRunning read_and_write\n")
        input_ts = random_data.create_random_ts(n_ts=3, n_req=2, n_hist=3, max_length=2000, min_length=200)

        # write data to file in IoT format
        write_data_to_iot_format.write_ts(input_ts, FILE_NAME)

        data, metric_ids, host_ids, header_names = get_iot_data.get_data(FILE_NAME, "all", True)
        os.remove(FILE_NAME)


        converted_ts = load_time_series.iot_to_struct_by_dataset(data, host_ids, dataset_idx=[0])[0]

        for tsi, tsc in zip(input_ts.data, converted_ts.data):
            # print(max(abs(np.array(tsi.T) - np.array(tsc.T))))
            # print(max(abs(tsi.index - tsc.index)))
            self.assertTrue((abs(np.array(tsi.T) - np.array(tsc.T)) < TOL).all(),
                            "Maximum difference {} between ts values exceeded tolerance {}".
                            format(max(abs(np.array(tsi.T) - np.array(tsc.T))), TOL))
            self.assertTrue((abs(tsi.index - tsc.index) < TOL).all(),
                            "Maximum difference {} between ts indices exceeded tolerance {}".
                            format(max(abs(np.array(tsi.T) - np.array(tsc.T))), TOL))

        self.assertTrue(input_ts.name == converted_ts.name)

    def test_read_empty_lines(self):
        """ Checks results of reading empty lines """
        input_ts = random_data.create_random_ts(n_ts=5, n_req=3, n_hist=7, max_length=2000, min_length=200)

        # write data to file in IoT format
        write_data_to_iot_format.write_ts(input_ts, FILE_NAME)

        data, metric_ids, host_ids, header_names = get_iot_data.get_data(FILE_NAME, [], True)
        os.remove(FILE_NAME)

        with self.assertRaises(IndexError) as e:
            ts_list = load_time_series.iot_to_struct_by_dataset(data, host_ids, dataset_idx=[0])

        #self.assertTrue('This is broken' in e.exception)

    def test_read_empty_dataset(self):
        """ Checks results of reading empty dataset """
        input_ts = random_data.create_random_ts(n_ts=1, n_req=2, n_hist=5, max_length=2000, min_length=200)
        input_ts.data = []


        write_data_to_iot_format.write_ts(input_ts, FILE_NAME)
        data, metric_ids, host_ids, header_names = get_iot_data.get_data(FILE_NAME, "all", True)
        os.remove(FILE_NAME)

        with self.assertRaises(IndexError) as e:
            ts = load_time_series.iot_to_struct_by_dataset(data, host_ids, dataset_idx=[0])[0]

        #self.assertTrue('This is broken' in e.exception)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIotData)
    unittest.TextTestRunner(verbosity=2).run(suite)