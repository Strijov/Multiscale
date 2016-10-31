from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import numpy as np
from RegressionMatrix import regression_matrix
from sklearn.linear_model import Lasso
from LoadAndSaveData import load_time_series
from Forecasting import frc_class
import my_plots

# Experiment data
DATASET = 'EnergyWeather'
TS_IDX = [0,1,2,4,5,6] # We exclude precipitation from the list of time series

# Example of partition. You may use all missing_values time series and both orig time series to test you model.
# The final quality will be assesed on varying_rates time series, so don't look at them until you are finished
TRAIN_FILE_NAMES = ['missing_value_train']
TEST_FILE_NAMES = ['missing_value_test']
HIDDEN_TEST = ['varying'] #

# output and saving parameters
VERBOSE = False
SAVE_DIR = "results"
FNAME_PREFIX = ""


TRAIN_TEST_RATIO = 0.75
N_STEPS = 1 # forecast only one requested interval

#model = frc_class.PipelineModel().load_model("results\\frc_sel_gen.pkl")

def main():
    # Load and prepare dataset.
    load_raw = not os.path.exists(os.path.join("ProcessedData", "EnergyWeather_orig_train.pkl"))
    load_time_series.load_all_time_series(datasets=[DATASET], load_raw=load_raw, verbose=VERBOSE)
    ts_list = []
    for name in TRAIN_FILE_NAMES:
        ts_list.append(load_time_series.load_all_time_series(datasets=[DATASET], load_raw=False, name_pattern=name, verbose=False)[0])


    generator = frc_class.IdentityGenerator(name="Identity generator")
    # Example: define a transformation function for feature generation
    def transform(X):
        return np.hstack((X, np.power(X, 2)))
    generator.transform = transform

    # feature selection model can be defined in the same way. If you don't use any, just leave as is
    selector = frc_class.IdentityModel(name="Identity selector")

    # first argument is your model class, then follow optional parameters as keyword arguments
    frc_model = frc_class.CustomModel(Lasso, name="Lasso", alpha=0.01)

    # train your model:
    model = demo_train(ts_list, frc_model=frc_model, fg_mdl=generator, fs_mdl=selector, verbose=VERBOSE)

    # evaluate errors on the test set
    train_error, train_std = competition_errors(model=model, names=TRAIN_FILE_NAMES, y_idx=TS_IDX)
    test_error, test_std = competition_errors(model=model, names=TEST_FILE_NAMES, y_idx=TS_IDX)

    print("Mean error across time series: train = {} with std {}, test = {} with std {}".format(train_error, train_std, test_error, test_std))

    return train_error, test_error


def demo_train(ts_struct_list, frc_model=None, fg_mdl=None, fs_mdl=None, verbose=False):
    """
    Train and save the model.

    :param ts_struct_list: list of namedtuples tsStruct
    :param model: list of dictionaries which specify model structure
    :param generators: list of dictionaries which specify feature generators
    :param feature_selection_mdl: list of dictionaries which specify feature selection strategies
    :param verbose: controls the output
    :return: testError, trainError, bias, model
    """

    # Check arguments:
    if fg_mdl is None:
        fg_mdl = frc_class.IdentityGenerator(name="Identity generator")

    if fs_mdl is None:
        fs_mdl = frc_class.IdentityModel(name="Identity selector")

    if frc_model is None:
        frc_model = frc_class.CustomModel(Lasso, name="Lasso", alpha=0.01)



    results = []
    res_text = []

    for ts in ts_struct_list:
        data = regression_matrix.RegMatrix(ts, x_idx=TS_IDX, y_idx=TS_IDX)

        # Create regression matrix
        data.create_matrix(nsteps=N_STEPS, norm_flag=True) # this creates data.Y, data.X and some other fields

        # Split data for training and testing
        data.train_test_split(TRAIN_TEST_RATIO)

        # train the model. This returns trained pipeline and its steps
        model, frc, gen, sel = data.train_model(frc_model=frc_model, generator=fg_mdl, selector=fs_mdl)


        if verbose:
            model.print_pipeline_pars()
            verbose = False

        frcY, _ = data.forecast(model) # returns forecasted matrix of the same shape as data.Y
        # frcY, idx_frc = data.forecast(model, idx_rows=data.idx_test) # this would return forecasts only for data.testY

        data.plot_frc(n_frc=5, n_hist=10, folder=SAVE_DIR) #this saves figures into SAVE_DIR

        train_mae = data.mae(idx_rows=data.idx_train)
        train_mape = data.mape(idx_rows=data.idx_train)

        test_mae = data.mae(idx_rows=data.idx_test)
        test_mape = data.mape(idx_rows=data.idx_test)

        index = [ts.data[i].name for i in TS_IDX]
        res1 = pd.DataFrame(train_mae, index=index, columns=[("MAE", "train")])
        res2 = pd.DataFrame(train_mape, index=index, columns=[("MAPE", "train")])
        res3 = pd.DataFrame(test_mae, index=index, columns=[("MAE", "test")])
        res4 = pd.DataFrame(test_mape, index=index, columns=[("MAPE", "test")])
        res = pd.concat([res1, res2, res3, res4], axis=1)

        if verbose:
            print(res)

        results.append(res)
        res_text.append("Time series {0} forecasted with {1} + '{2}' feature generation model and '{3}' feature selection model \n \\\\".
                        format(ts.name, frc.name, gen.name, sel.name))

    #saved_fname = model.save_model(file_name=FNAME_PREFIX, folder=SAVE_DIR) # saving in not an option yet

    # write results into a latex file
    my_plots.save_to_latex(results, df_names=res_text, folder=SAVE_DIR)


    return model


def competition_errors(model, names, y_idx=None):

    # if isinstance(model, str):
    #     model = frc_class.PipelineModel().load_model(model) # this doesn't work yet

    mape = []
    for name in names:
        ts = load_time_series.load_all_time_series(datasets=['EnergyWeather'], load_raw=False, name_pattern=name, verbose=False)[0]

        data = regression_matrix.RegMatrix(ts, y_idx=y_idx)
        data.create_matrix()
        data.forecast(model)

        mape.append(data.mape())

    return np.mean(mape), np.std(mape)




if __name__ == '__main__':
    main()
