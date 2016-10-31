from __future__ import division
from __future__ import print_function

import os
import pandas as pd
from collections import namedtuple
from RegressionMatrix import regression_matrix, random_data
from sklearn.linear_model import Lasso
from LoadAndSaveData import load_time_series
from Forecasting import frc_class
import my_plots


tsStruct = namedtuple('tsStruct', 'data request history name readme')


def demo_compare_forecasts(ts_struct_list=None, model=None, generators=None, feature_selection_mdl=None, verbose=False):
    #[testError, trainError, bias, model] = demoCompareForecasts(tsStructArray, model, generators, feature_selection_mdl, verbose)
    """
    Script demoCompareForecasts runs one forecasting experiment.
    It applies several competitive models to single dataset.
    :param ts_struct_list: list of namedtuples tsStruct
    :param model: list of dictionaries which specify model structure
    :param generators: list of dictionaries which specify feature generators
    :param feature_selection_mdl: list of dictionaries which specify feature selection strategies
    :param verbose:
    :return: testError, trainError, bias, model
    """

    # Experiment settings.
    TRAIN_TEST_RATIO = 0.75
    VERBOSE = True

    # Check arguments:
    if generators is None:
        generator = frc_class.IdentityGenerator(name="Identity generator")

    if feature_selection_mdl is None:
        selector = frc_class.IdentityModel(name="Identity selector")

    frc_model = frc_class.CustomModel(Lasso, name="Lasso", alpha=0.01)


    # Load and prepare dataset.
    load_raw = not os.path.exists(os.path.join("ProcessedData", "EnergyWeather_orig_train.pkl"))
    try:
        ts_struct_list = load_time_series.load_all_time_series(datasets=['EnergyWeather'], load_raw=load_raw, name_pattern="", verbose=VERBOSE)
    except:
        ts_struct_list = load_time_series.load_all_time_series(datasets=['EnergyWeather'], load_raw=True, name_pattern="", verbose=VERBOSE)

    print(ts_struct_list)


    results = []
    res_text = []
    TS_IDX = [0,1,2,4,5,6] # exclude precipitation from the list of time series
    for ts in ts_struct_list:
        data = regression_matrix.RegMatrix(ts, x_idx=TS_IDX, y_idx=TS_IDX)

        # Create regression matrix
        data.create_matrix(nsteps=1, norm_flag=True)

        # Split data for training and testing
        data.train_test_split(TRAIN_TEST_RATIO)
        model, frc, gen, sel = data.train_model(frc_model=frc_model, generator=generator, selector=selector) # model parameters are changed inside
        if VERBOSE:
            frc_class.print_pipeline_pars(model)
            VERBOSE = False

        data.forecast(model)
        data.plot_frc(n_frc=5, n_hist=10)

        train_mae = data.mae(idx_rows=data.idx_train, out=None)
        train_mape = data.mape(idx_rows=data.idx_train, out=None)

        test_mae = data.mae(idx_rows=data.idx_test, out=None)
        test_mape = data.mape(idx_rows=data.idx_test, out=None)

        index = [ts.data[i].name for i in TS_IDX]
        res1 = pd.DataFrame(train_mae, index=index, columns=[("MAE", "train")])
        res2 = pd.DataFrame(train_mape, index=index, columns=[("MAPE", "train")])
        res3 = pd.DataFrame(test_mae, index=index, columns=[("MAE", "test")])
        res4 = pd.DataFrame(test_mape, index=index, columns=[("MAPE", "test")])
        res = pd.concat([res1, res2, res3, res4], axis=1)
        print(res)

        results.append(res)
        res_text.append("Time series {0} forecasted with {1} + '{2}' feature generation model and '{3}' feature selection model \n \\\\".
                        format(ts.name, frc.name, gen.name, sel.name))


    my_plots.save_to_latex(results, df_names=res_text)



    return res



if __name__ == '__main__':
    demo_compare_forecasts()
