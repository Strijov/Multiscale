from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import namedtuple
from itertools import product
from sklearn.linear_model import Lasso
from statsmodels.tsa.stattools import adfuller
from sklearn.externals import joblib

import my_plots
from RegressionMatrix import regression_matrix
from LoadAndSaveData import load_time_series
from Forecasting import frc_class


def main(frc_model=None, selector=None, generator=None):
    # Experiment settings.
    MAX_NOIZE = 2
    N_PREDICTIONS = 10  # plotting par

    # Load and prepare dataset.
    load_raw = not os.path.exists(os.path.join("ProcessedData", "EnergyWeather_orig_train.pkl"))
    ts_struct_list = load_time_series.load_all_time_series(datasets='EnergyWeather', load_raw=load_raw, name_pattern="orig_train")

    if generator is None:
        generator = frc_class.CustomModel(frc_class.IdentityGenerator, name="No generation")
    if selector is None:
        selector = frc_class.IdentityModel(name="No selection")

    if frc_model is None:
        frc_model = frc_class.CustomModel(Lasso, name="Lasso", alpha=0.01)  # frc_class.IdenitityFrc() #LinearRegression()
    # Create regression matrix

    results = []
    res_text = []
    ts0 = ts_struct_list[0]
    noise_ratio_list = list(np.arange(0.1, 2.1, 0.1))
    horizon_list = range(1, MAX_HORIZON)
    results = {}
    res_matrix = [np.zeros((len(noise_ratio_list), len(horizon_list)))] * len(ts0.data)
    for nsteps, noise_ratio in product(horizon_list, noise_ratio_list):
        # Be sure to modify the original time series
        ts = add_normal_noise(ts0, noise_ratio)

        data = regression_matrix.RegMatrix(ts)
        # Create regression matrix
        data.create_matrix(nsteps=nsteps, norm_flag=True)

        # Split data for training and testing
        data.train_test_split(TRAIN_TEST_RATIO)
        model, _, _, _ = data.train_model(frc_model=frc_model, generator=generator,
                                 selector=selector, from_scratch=True)  # model parameters are changed inside

        # data.forecasts returns model obj, forecasted rows of Y matrix and a list [nts] of "flat"/ts indices of forecasted points
        data.forecast(model, replace=True)

        mae_train = data.mae(idx_rows=data.idx_train, out=None)  # , out="Training")
        mape_train = data.mape(idx_rows=data.idx_train, out=None)  # , out="Training")
        mae_test = data.mae(idx_rows=data.idx_test, out=None)  # , out="Test")
        mape_test = data.mape(idx_rows=data.idx_test, out=None)


        idx = data.matrix_to_flat(data.idx_test)
        test_rsd = [0] * data.nts
        for i, ts in enumerate(data.ts):
            test_rsd[i] = ts.s[idx[i]] - data.forecasts[i][idx[i]]
            results[(nsteps, noise_ratio, ts.name, "ADF p-value, test")] = check_stationarity(test_rsd[i], ts.name)
            results[(nsteps, noise_ratio, ts.name, "residues, test")] = ts.s[idx[i]] - data.forecasts[i][idx[i]]

            results[(nsteps, noise_ratio, ts.name, "MAE, train")] = mae_train[i]
            results[(nsteps, noise_ratio, ts.name, "MAPE, train")] = mape_train[i]
            results[(nsteps, noise_ratio, ts.name, "MAE, test")] = mae_test[i]
            results[(nsteps, noise_ratio, ts.name, "MAPE, test")] = mape_test[i]

    joblib.dump(results, "results_structure")
    error_by_ts(results, errors=["MAPE, test"])

    return results


def error_by_ts(results, errors=None):
    keys = results.keys()
    steps, noises, ts_names, err_names = zip(*keys)

    steps = np.unique(steps)
    noises = np.unique(noises)
    err_names = np.unique(err_names)
    ts_names = np.unique(ts_names)

    if errors is None:
        errors = err_names


    for ts_name in ts_names:
        err_matrix = [np.zeros((len(noises), len(steps)))] * len(ts_names)
        plt.figure(figsize=(7, 5))
        for i, err_name in enumerate(errors):
            for j, k in product(range(len(steps)), range(len(noises))):
                err_matrix[i][k, j] = results[steps[j], noises[k], ts_name, err_name]

            plt.plot(err_matrix[i], color='blue', label=err_name)
        plt.legend(loc='best')
        plt.title(ts_name)
        plt.show(block=False)


def errors_by_horizon(results):
    pass



def check_stationarity(ts, name=None):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(ts, window=12)
    rolstd = pd.rolling_std(ts, window=12)

    # Plot rolling statistics:
    # plt.figure(figsize=(7, 5))
    # orig = plt.plot(ts, color='blue', label='Original')
    # mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    # std = plt.plot(rolstd, color='black', label='Rolling Std')
    # plt.legend(loc='best')
    # plt.title('Rolling Mean & Standard Deviation for residuals of '+ name)
    # plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

    return dfoutput['p-value']


def add_normal_noise(ts_struct, noise_ratio):
    new_ts = [0] * len(ts_struct.data)
    for i, ts in enumerate(ts_struct.data):
        range_ts = max(ts) - min(ts)
        new_ts[i] = ts + np.random.randn()*np.sqrt(range_ts)*noise_ratio

    ts_struct = tsStruct(new_ts, ts_struct.request, ts_struct.history, ts_struct.name, ts_struct.readme)
    return ts_struct


if __name__ == '__main__':
    main()
    #results = joblib.load("results_structure")
    #error_by_ts(results, ["MAE, test"])

