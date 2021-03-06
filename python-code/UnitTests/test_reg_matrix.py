import unittest
import copy

from Forecasting import frc_class
from RegressionMatrix import regression_matrix, random_data


TOL = pow(10, -10)

class TestRegMatrix(unittest.TestCase):

    def test_identity(self):
        input_ts = random_data.create_random_ts(n_ts=3, n_req=1, n_hist=2, max_length=200, min_length=200)

        data = regression_matrix.RegMatrix(input_ts)
        data.create_matrix()
        data.X = data.Y  # for identity frc
        data.train_test_split(0.25)

        model = frc_class.PipelineModel(frc_mdl=frc_class.IdentityFrc())
        model.train_model(data.trainX, data.trainY)  # model parameters are changed inside

        frc, idx_frc = data.forecast(model, idx_rows=data.idx_train, replace=True)
        self.assertTrue((frc == data.trainY).all()) # first check  that identity frc works

        frc, idx_frc = data.forecast(model, idx_rows=data.idx_test, replace=True)
        self.assertTrue((frc == data.testY).all()) # ones again, check  that identity frc works

        # now check forecats:
        #print data.mae(), data.mape()
        self.assertTrue((data.mae() < TOL).all())
        self.assertTrue((data.mape() < TOL).all())
        #self.assertTrue((data.mape() < TOL*np.array([sum(ts.s) for ts in data.ts])).all())



    def test_y_slicing_args(self):
        """ Check that individual forecasts are the same if sliced in init or at create_matrix """

        input_ts = random_data.create_random_ts(n_ts=3, n_req=2, n_hist=13, max_length=500, min_length=200)

        # include all ts explicitly
        data = regression_matrix.RegMatrix(input_ts,  y_idx=range(len(input_ts.data)))
        data.create_matrix()
        data.train_test_split(0.25)

        model = frc_class.PipelineModel(frc_mdl=frc_class.MartingalFrc())
        model.train_model(data.trainX, data.trainY)  # model parameters are changed inside
        frc1, idx_frc = data.forecast(model)

        # let the model define y_idx
        data = regression_matrix.RegMatrix(input_ts)
        data.create_matrix()
        data.train_test_split(0.25)

        model = frc_class.PipelineModel(frc_mdl=frc_class.MartingalFrc())
        model.train_model(data.trainX, data.trainY)

        frc2, idx_frc = data.forecast(model)

        data = regression_matrix.RegMatrix(input_ts)
        data.create_matrix(y_idx=range(len(input_ts.data)))
        data.train_test_split(0.25)

        model = frc_class.PipelineModel(frc_mdl=frc_class.MartingalFrc())
        model.train_model(data.trainX, data.trainY)  # model parameters are changed inside
        frc3, idx_frc = data.forecast(model)

        self.assertTrue((frc1 == frc2).all())
        self.assertTrue((frc3 == frc2).all())
        self.assertTrue((frc1 == frc3).all())


    def test_frc_by_one_2(self):
        """ Check that individual forecasts do not depend on the rest of the matrix """

        input_ts = random_data.create_random_ts(n_ts=3, n_req=11, n_hist=23, max_length=200, min_length=200)
        # create the data object for all ts
        data = regression_matrix.RegMatrix(input_ts, y_idx=range(len(input_ts.data)))
        # then construct the matrix just for one ts:
        data.create_matrix(y_idx=0, x_idx=0)
        data.train_test_split(0.25)
        model = frc_class.PipelineModel(frc_mdl=frc_class.MartingalFrc())
        model.train_model(data.trainX, data.trainY)  # model parameters are changed inside
        frc0, idx_frc = data.forecast(model)
        # Remember the first ts:
        ts0 = input_ts.data[0]

        for i in xrange(5):
            # generate new data
            input_ts = random_data.create_random_ts(n_ts=3, n_req=11, n_hist=23, max_length=200, min_length=200)
            # keep the first ts the same
            new_ts = [ts0]
            new_ts.extend(input_ts.data[1:])
            input_ts.data = new_ts
            data = regression_matrix.RegMatrix(input_ts)
            data.create_matrix(y_idx=0, x_idx=0)
            data.train_test_split(0.25)
            model = frc_class.PipelineModel(frc_mdl=frc_class.MartingalFrc())
            model.train_model(data.trainX, data.trainY)  # model parameters are changed inside
            frc, idx_frc = data.forecast(model)
            self.assertTrue((frc0 == frc).all())


    def test_normalization(self):
        """ Check that errors with normalization are the same errors for normalized ts"""
        pass

    def test_frc_by_one(self):
        """Check that individual forecasts are the same as frc for a set of one ts"""

        input_ts = random_data.create_random_ts(n_ts=5, n_req=11, n_hist=23, max_length=500, min_length=200)

        for i_ts, ts in enumerate(input_ts.data):
            data = regression_matrix.RegMatrix(input_ts, y_idx=i_ts)
            data.create_matrix()
            data.train_test_split(0.25)

            model = frc_class.PipelineModel(frc_mdl=frc_class.MartingalFrc())
            model.train_model(data.trainX, data.trainY)  # model parameters are changed inside
            frc1, idx_frc = data.forecast(model)
            Y1 = data.Y

            input_ts2 = copy.deepcopy(input_ts)
            input_ts2.data = input_ts.data[i_ts:i_ts + 1]
            data = regression_matrix.RegMatrix(input_ts2)
            data.create_matrix()
            data.train_test_split(0.25)

            model = frc_class.PipelineModel(frc_mdl=frc_class.MartingalFrc())
            model.train_model(data.trainX, data.trainY) # model parameters are changed inside
            frc2, idx_frc = data.forecast(model)
            Y2 = data.Y

            self.assertTrue((frc1 == frc2).all())
            self.assertTrue((Y1 == Y2).all())

        return None


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRegMatrix)
    unittest.TextTestRunner(verbosity=2).run(suite)