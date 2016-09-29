from __future__ import division
from __future__ import print_function


import pandas as pd
from collections import namedtuple
from RegressionMatrix import regression_matrix
from sklearn.linear_model import Lasso
from LoadAndSaveData import load_time_series
from Forecasting import frc_class
import my_plots

tsStruct = namedtuple('tsStruct', 'data request history name readme')


def drop_cols(model, X):
    return X[:, :100]

def main(frc_model=None, generator=None, selector=None):


    # Experiment settings.
    TRAIN_TEST_RATIO = 0.75
    N_PREDICTIONS = 10 # plotting par

    # Load and prepare dataset.
    ts_struct_list = load_time_series.load_all_time_series(datasets='EnergyWeather', load_raw=True, name_pattern="")

    if generator is None:
        generator = frc_class.CustomModel(name='Poly', fitfunc=None, predictfunc=None, replace=True, ndegrees=3)
    if selector is None:
        selector = frc_class.CustomModel(name="Identity", fitfunc=None, predictfunc=drop_cols)

    if frc_model is None:
        frc_model = Lasso(alpha=0.01) #frc_class.IdenitityFrc() #LinearRegression()
    # Create regression matrix

    results = []
    res_text = []
    for ts in ts_struct_list:
        data = regression_matrix.RegMatrix(ts)
        # Create regression matrix
        data.create_matrix(nsteps=1, norm_flag=True)

        # Split data for training and testing
        data.train_test_split(TRAIN_TEST_RATIO)
        model = data.train_model(frc_model=frc_model, generator=generator, selector=selector) # model parameters are changed inside

        # data.forecasts returns model obj, forecasted rows of Y matrix and a list [nts] of "flat"/ts indices of forecasted points
        frc, idx_frc = data.forecast(model, data.idx_test, replace=True)
        frc, idx_frc = data.forecast(model, data.idx_train, replace=True)


        train_mae = data.mae(idx_rows=data.idx_train, out=None)#, out="Training")
        train_mape = data.mape(idx_rows=data.idx_train, out=None)#, out="Training")
        test_mae = data.mae(idx_rows=data.idx_test, out=None)#, out="Test")
        test_mape = data.mape(idx_rows=data.idx_test, out=None)#, out="Test")

        res1 = pd.DataFrame(train_mae, index=[t.name for t in ts.data], columns=[("MAE", "train")])
        res2 = pd.DataFrame(train_mape, index=[t.name for t in ts.data], columns=[("MAPE", "train")])
        res3 = pd.DataFrame(test_mae, index=[t.name for t in ts.data], columns=[("MAE", "test")])
        res4 = pd.DataFrame(test_mape, index=[t.name for t in ts.data], columns=[("MAPE", "test")])
        res = pd.concat([res1, res2, res3, res4], axis=1)
        print(res)

        results.append(res)
        res_text.append(ts.name + ": Lasso, feature generation: " + generator.name + ", " + selector.name)

        data.plot_frc(n_frc=N_PREDICTIONS)

    my_plots.save_to_latex(results, df_names=res_text)



    return results




if __name__ == '__main__':
    main()
