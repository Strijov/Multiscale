import unittest

from Forecasting import frc_class
from RegressionMatrix import regression_matrix, random_data
from LoadAndSaveData import write_data_to_iot_format, get_iot_data

FILE_NAME = "TestIot.csv"
TOL = pow(10, -10)

class TestIotData(unittest.TestCase):
    def test_read_lines(self):
        pass

    def read_and_write(self):
        input_ts = random_data.create_random_ts(n_ts=3, n_req=10, n_hist=20, max_length=200, min_length=200)
        data_original = regression_matrix.RegMatrix(input_ts)
        #data_original.create_matrix()

        # write data to file in IoT format
        write_data_to_iot_format.write_ts(input_ts, FILE_NAME)

        data, metric_ids, host_ids, header_names = get_iot_data.get_data(FILE_NAME, "all", True)

        dataset = host_ids.keys()[0]
        converted_ts = write_data_to_iot_format.from_iot_to_struct(data, host_ids[dataset], dataset)
        converted_data = regression_matrix.RegMatrix(converted_ts)
        #converted_data.create_matrix()

        for tsi, tsc in zip(input_ts.data, converted_ts.data):
            self.assertTrue((abs(tsi - tsc) < TOL).all())







suite = unittest.TestLoader().loadTestsFromTestCase(TestIotData)
unittest.TextTestRunner(verbosity=2).run(suite)