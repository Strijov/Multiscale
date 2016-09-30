import unittest

from Forecasting import frc_class
from RegressionMatrix import regression_matrix, random_data

TOL = pow(10, -10)

class TestRegMatrix(unittest.TestCase):

    def test_identity(self):
        input_ts = random_data.create_random_ts(n_ts = 3, n_req = 10, n_hist=20, max_length=200, min_length = 200)
        data = regression_matrix.RegMatrix(input_ts)


        # for n, ts in  enumerate(input_ts):
        #     self.assertTrue((ts==data.ts[n].data).all())

        data.create_matrix()

        data.X = data.Y  # for identity frc
        data.train_test_split(0.25)

        model = frc_class.IdenitityFrc()
        model = data.train_model(model)  # model parameters are changed inside

        frc, idx_frc = data.forecast(model, data.idx_train, replace=True)
        self.assertTrue((frc == data.trainY).all()) # first check  that identity frc works

        frc, idx_frc = data.forecast(model, data.idx_test, replace=True)
        self.assertTrue((frc == data.testY).all()) # ones again, check  that identity frc works

        # now check forecats:
        #print data.mae(), data.mape()
        self.assertTrue((data.mae() < TOL).all())
        self.assertTrue((data.mape() < TOL).all())
        #self.assertTrue((data.mape() < TOL*np.array([sum(ts.s) for ts in data.ts])).all())




suite = unittest.TestLoader().loadTestsFromTestCase(TestRegMatrix)
unittest.TextTestRunner(verbosity=2).run(suite)