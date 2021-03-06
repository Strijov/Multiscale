from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import numpy as np
from RegressionMatrix import regression_matrix
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
# from sklearn.decomposition import PCA
from LoadAndSaveData import load_time_series
from Forecasting import frc_class
from Features import generation_models as gnt_class
from Features import quadratic_feature_selection as sel_class
# from Forecasting.GatingEnsemble import GatingEnsemble
# from Forecasting.LSTM import LSTM
import my_plots

# Experiment data
DATASET = 'EnergyWeather'
TS_IDX = [0, 1, 2, 4, 5, 6] # We exclude precipitation from the list of time series

# Example of partition. You may use all missing_values time series and both orig time series to test you model.
# The final quality will be assesed on varying_rates time series, so don't look at them until you are finished
TRAIN_FILE_NAMES = ['missing_value_train']
TEST_FILE_NAMES = ['missing_value_test']
HIDDEN_TEST = ['varying'] #

feature_gnt_names = [None, 'univariate_transformation',
                       # 'bivariate_transformation', this one has additional argument
                       'simple_statistics',
                       'haar_transformations',
                       'monotone_linear',
                       'monotone_polinomial_rate',
                       'monotone_sublinear_polinomial_rate',
                       'monotone_logarithmic_rate',
                       'monotone_slow_convergence',
                       'monotone_fast_convergence',
                       'monotone_soft_relu',
                       'monotone_sigmoid',
                       'monotone_soft_max',
                       'monotone_hyberbolic_tangent',
                       'monotone_softsign',
                       'centroids',
                       'all']

# output and saving parameters
VERBOSE = False
SAVE_DIR = "results"
FNAME_PREFIX = ""

TRAIN_TEST_RATIO = 0.75
N_STEPS = 1 # forecast only one requested interval


def main():
    """
    Provides an example of usage of the system.

    The model consists of three main components: feature generation, feature selection and forecasting model.
    Feature generation and selection may be empty:
    generation = None
    selection = None

    which is the same as
    generator = gnt_class.FeatureGeneration(name="Identity generator")
    selector = sel_class.FeatureSelection(name="Identity selector", on=False)

    Other options for feature generation:
    generator = gnt_class.FeatureGeneration(name="univariate", replace=False, norm=True
                                            transformations=["univariate_transformation", "centroids"])
    generator = gnt_class.Nonparametric()
    generator = gnt_class.Monotone()

    Examples of using sklearn solutions:
    frc_class.CustomModel(PCA, name="Randomized PCA", svd_solver="randomized")
    frc_class.CustomModel(PCA, name="PCA")

    Examples of custom models:
    * Mixture of experts:
    frc_model = frc_class.CustomModel(GatingEnsemble, name="Mixture", estimators=[Lasso(alpha=0.01), Lasso(alpha=0.001)])
    * LSTM network:
    frc_model = frc_class.CustomModel(LSTM.LSTM, name="LSTM")

    """
    # Load and prepare dataset.
    ts_list = load_energy_weather_data()

    generator = gnt_class.FeatureGeneration(transformations='centroids') #gnt_class.Monotone()

    # feature selection model can be defined in the same way. If you don't use any, just leave as is
    selector = sel_class.FeatureSelection(on=False) #
    # first argument is your model class, then follow optional parameters as keyword arguments
    frc_model = frc_class.CustomModel(RandomForestRegressor, name="RF")
    #frc_class.CustomModel(Lasso, name="Lasso", alpha=0.001)

    # train your model:
    model = demo_train(ts_list, frc_model=frc_model, fg_mdl=generator, fs_mdl=selector, verbose=VERBOSE)

    # evaluate errors on the test set
    train_error, train_std = competition_errors(model=model, names=TRAIN_FILE_NAMES, y_idx=TS_IDX)
    test_error, test_std = competition_errors(model=model, names=TEST_FILE_NAMES, y_idx=TS_IDX)


    print("Average MAPE across time series: train = {} with std {}, test = {} with std {}".
          format(train_error, train_std, test_error, test_std))

    return train_error, test_error


def load_energy_weather_data(load_raw=None, fnames=TRAIN_FILE_NAMES):
    """Load data from the EnergyWeather dataset """
    if load_raw is None:
        load_raw = not os.path.exists(os.path.join("..", "data", "ProcessedData", "EnergyWeather_orig_train.pkl"))

    load_time_series.load_all_time_series(datasets=[DATASET], load_raw=load_raw, verbose=VERBOSE)
    ts_list = []
    for name in fnames:
        ts_list.extend(
            load_time_series.load_all_time_series(datasets=[DATASET], load_raw=False, name_pattern=name,
                                                  verbose=False)
        )
        print(name)
        print(ts_list[-1].summarize_ts())

    return ts_list


def demo_train(ts_struct_list, frc_model=None, fg_mdl=None, fs_mdl=None, verbose=False,
               return_model=False, rewrite=True):
    """
    Train and save the model.

    :param ts_struct_list: list of namedtuples tsStruct
    :param frc_model: forecasting model, instance of CustomModel
    :param fg_mdl: feature generation model, instance of FeatureGeneration
    :param fs_mdl: feature selection model, instance of FeatureSelection
    :param verbose: controls the output
    :return: testError, trainError, bias, model
    """

    # Check arguments:
    if fg_mdl is None:
        fg_mdl = frc_class.IdentityGenerator(name="Identity generator", on=False)

    if fs_mdl is None:
        fs_mdl = gnt_class.FeatureGeneration()  # IdentityModel(name="Identity selector")

    if frc_model is None:
        frc_model = frc_class.CustomModel(Lasso, name="Lasso", alpha=0.01)

    model = frc_class.PipelineModel(gen_mdl=fg_mdl, sel_mdl=fs_mdl, frc_mdl=frc_model)
    results = []
    res_text = []

    for ts in ts_struct_list:
        data = regression_matrix.RegMatrix(ts, x_idx=TS_IDX, y_idx=TS_IDX)

        # Create regression matrix
        data.create_matrix(nsteps=N_STEPS, norm_flag=True) # this creates data.Y, data.X and some other fields

        # Split data for training and testing
        data.train_test_split(TRAIN_TEST_RATIO)

        # train the model. This returns trained pipeline and its steps
        model, frc, gen, sel = model.train_model(data.trainX, data.trainY)

        selection_res = "\n Feature selection results: problem status {}, selected {} from {} \\\\ \n".\
            format(sel.status, len(sel.selected), sel.n_vars)

        frcY, _ = data.forecast(model) # returns forecasted matrix of the same shape as data.Y
        # frcY, idx_frc = data.forecast(model, idx_rows=data.idx_test) # this would return forecasts only for data.testY

        data.plot_frc(n_frc=5, n_hist=10, folder=SAVE_DIR) #this saves figures into SAVE_DIR

        train_mae = data.mae(idx_rows=data.idx_train, idx_original=data.original_index)
        train_mape = data.mape(idx_rows=data.idx_train, idx_original=data.original_index)

        test_mae = data.mae(idx_rows=data.idx_test, idx_original=data.original_index)
        test_mape = data.mape(idx_rows=data.idx_test, idx_original=data.original_index)

        index = [ts.data[i].name for i in TS_IDX]
        res1 = pd.DataFrame(train_mae, index=index, columns=[("MAE", "train")])
        res2 = pd.DataFrame(train_mape, index=index, columns=[("MAPE", "train")])
        res3 = pd.DataFrame(test_mae, index=index, columns=[("MAE", "test")])
        res4 = pd.DataFrame(test_mape, index=index, columns=[("MAPE", "test")])
        res = pd.concat([res1, res2, res3, res4], axis=1)

        configuration_str = "\n Time series {} forecasted with {} + '{}' feature generation model and  " \
                            "'{}' feature selection model \\\\ \n".format(ts.name, frc.name, gen.name, sel.name)
        if verbose:
            print(configuration_str)
            print(selection_res)
            print(res)

        results.append(res)
        res_text.append(configuration_str)
        res_text.append(selection_res)

    saved_mdl_fname = model.save_model(file_name=FNAME_PREFIX, folder=SAVE_DIR) # saving in not an option yet
    # model = frc_class.PipelineModel().load_model(file_name=fname)

    # write results into a latex file
    my_plots.save_to_latex(results, df_names=res_text, folder=SAVE_DIR, rewrite=rewrite)
    print("Results saved to folder {}".format(SAVE_DIR))

    if return_model:
        return model, saved_mdl_fname

    return saved_mdl_fname


def competition_errors(model, names, y_idx=None):
    """
    Returns MAPE, averaged over a set of multivariate time series, specified by names

    :param model: trained forecasting model
    :type model: PipelineModel
    :param names: (parts of) names of time series in the set
    :type names: list
    :param y_idx:
    :type y_idx:
    :return:
    :rtype:
    """

    if isinstance(model, str):
        model = frc_class.PipelineModel().load_model(model)

    mape = []
    for name in names:
        ts = load_time_series.load_all_time_series(datasets=['EnergyWeather'], load_raw=False, name_pattern=name, verbose=False)[0]

        data = regression_matrix.RegMatrix(ts, y_idx=y_idx)
        data.create_matrix()
        data.forecast(model)

        mape.append(data.mape())

    return np.mean(mape), np.std(mape)


def feature_generation_demo():

    ts_list = load_energy_weather_data(load_raw=False, fnames=TRAIN_FILE_NAMES)
    frc_model = frc_class.CustomModel(Lasso, name="Lasso", alpha=0.0001)
    selector = sel_class.FeatureSelection(name="Katrutsa")
    rewrite = True
    for fg_name in feature_gnt_names[:-2]:  #:["all"]
        generator = gnt_class.FeatureGeneration(name=fg_name, replace=False,
                                                transformations=[fg_name], norm=True)
        model, _ = demo_train(ts_list, frc_model=frc_model, fg_mdl=generator, fs_mdl=selector,
                              verbose=True, return_model=True, rewrite=rewrite)
        rewrite = False

        train_error, train_std = competition_errors(model=model, names=TRAIN_FILE_NAMES, y_idx=TS_IDX)
        test_error, test_std = competition_errors(model=model, names=TEST_FILE_NAMES, y_idx=TS_IDX)

        res_text = "\n Average MAPE across time series: train = {} with std {}, test = {} with std {} \\\\ \n".\
            format(train_error, train_std, test_error, test_std)

        print(res_text)
        my_plots.save_to_latex(text=res_text, folder=SAVE_DIR, rewrite=rewrite)


if __name__ == '__main__':
    #feature_generation_demo()
    main()