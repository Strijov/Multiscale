import unittest
import os

from RegressionMatrix import regression_matrix, random_data
from LoadAndSaveData import write_data_to_iot_format, get_iot_data

FILE_NAME = "TestIot.csv"
TOL = pow(10, -10)

class TestIotData(unittest.TestCase):
    def test_read_lines(self):
        input_ts = random_data.create_random_ts(n_ts=3, n_req=7, n_hist=23, max_length=2000, min_length=200)
        write_data_to_iot_format.write_ts(input_ts, FILE_NAME)
        for i, ts in enumerate(input_ts.data):
            data, metric_ids, host_ids, header_names = get_iot_data.get_data(FILE_NAME, [i+2], True)
            dataset = host_ids.keys()[0]
            converted_ts = write_data_to_iot_format.from_iot_to_struct(data, host_ids[dataset], dataset)
            self.assertTrue((abs(ts - converted_ts.data[0]) < TOL).all())
            self.assertTrue(ts.name == converted_ts.data[0].name)
            self.assertTrue(input_ts.name == dataset)

        os.remove(FILE_NAME)


    def read_and_write(self):
        input_ts = random_data.create_random_ts(n_ts=3, n_req=7, n_hist=23, max_length=2000, min_length=200)
        #data_original = regression_matrix.RegMatrix(input_ts)
        #data_original.create_matrix()

        # write data to file in IoT format
        write_data_to_iot_format.write_ts(input_ts, FILE_NAME)

        data, metric_ids, host_ids, header_names = get_iot_data.get_data(FILE_NAME, "all", True)
        os.remove(FILE_NAME)


        dataset = host_ids.keys()[0]
        converted_ts = write_data_to_iot_format.from_iot_to_struct(data, host_ids[dataset], dataset)
        #converted_data = regression_matrix.RegMatrix(converted_ts)
        #converted_data.create_matrix()

        for tsi, tsc in zip(input_ts.data, converted_ts.data):
            self.assertTrue((abs(tsi - tsc) < TOL).all())
            self.assertTrue((abs(tsi.index - tsc.index) < TOL).all())

        self.assertTrue(input_ts.name == converted_ts)








suite = unittest.TestLoader().loadTestsFromTestCase(TestIotData)
unittest.TextTestRunner(verbosity=2).run(suite)