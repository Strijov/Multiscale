# coding: utf-8
from __future__ import division
from __future__ import print_function

import pandas as pd
import utils_

from sklearn.linear_model import Lasso

from RegressionMatrix import regression_matrix
from Forecasting import frc_class
# from Forecasting import  LSTM, GatingEnsemble

def main(file_name, line_indices, header):
    """
    Forecast data simultaneously and separately and compare errors

    :param file_name: file name (.csv) with data in IoT format
    :type file_name: str
    :param line_indices: indices of lines to read from file.  Lines are enumerated from 1. If "all", read the whole file
    :param header: Specifies if the file contains a header row
    :type header: bool
    :return: forecasting errors
    :rtype: pandas.DataFrame
    """

    TRAIN_TEST_RATIO = 0.75
    N_PREDICTIONS = 10
    N_EXPERTS = 4


    ts = utils_.safe_read_iot_data(file_name=file_name, line_indices=line_indices, header=header, default="poisson")
    data = regression_matrix.RegMatrix(ts)
    # Create regression matrix
    data.create_matrix(nsteps=1, norm_flag=True)

    #frc_model = frc_class.CustomModel(Lasso, name="Lasso", alpha=0.01)
    frc_model = frc_class.CustomModel(Lasso, name="Lasso", alpha=0.001)
    #frc_model = frc_class.CustomModel(GatingEnsemble.GatingEnsemble,
    #                                  estimators = [LinearRegression() for i in range(N_EXPERTS)])#(LSTM.LSTM, name="LSTM")

    # Split data for training and testing
    data.train_test_split(TRAIN_TEST_RATIO)
    model, _, _, _ = data.train_model(frc_model=frc_model, generator=None,
                             selector=None)  # model parameters are changed inside

    print("Features before generation:", data.feature_dict)
    print("Features after generation:", model.named_steps["gen"].feature_dict)
    print("Features after generation:", model.named_steps["sel"].feature_dict)


    # data.forecasts returns model obj, forecasted rows of Y matrix and a list [nts] of "flat"/ts indices of forecasted points
    data.forecast(model, replace=True)

    train_mae = data.mae(idx_rows=data.idx_train)
    train_mape = data.mape(idx_rows=data.idx_train)
    test_mae = data.mae(idx_rows=data.idx_test)
    test_mape = data.mape(idx_rows=data.idx_test)

    res1 = pd.DataFrame(train_mae, index=[t.name for t in ts.data], columns=[("MAE", "train")])
    res2 = pd.DataFrame(train_mape, index=[t.name for t in ts.data], columns=[("MAPE", "train")])
    res3 = pd.DataFrame(test_mae, index=[t.name for t in ts.data], columns=[("MAE", "test")])
    res4 = pd.DataFrame(test_mape, index=[t.name for t in ts.data], columns=[("MAPE", "test")])
    res = pd.concat([res1, res2, res3, res4], axis=1)
    print(res)

    data.plot_frc(n_frc=N_PREDICTIONS)

    return res






if __name__ == '__main__':
    file_name, line_indices, header, _ = utils_.parse_options()
    main(file_name, line_indices, header)
