from __future__ import print_function
import numpy as np
import pandas as pd
import os
import time
import my_plots

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from LoadAndSaveData import get_iot_data, load_time_series
from RegressionMatrix import regression_matrix
from Forecasting import frc_class, arima_model



def main():
    # Init string for latex results:
    latex_str = ""
    folder = os.path.join("fig", str(int(time.time())))
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Load data in IoT format
    file_name = os.path.join('..', 'code','data', 'IotTemplate', 'data.csv')
    line_indices = range(2, 9) # FIXIT
    data, metric_ids, host_ids, header_names = get_iot_data.get_data(file_name, line_indices, True)

    # Select only data from first dataset in host_ids:
    dataset = host_ids.keys()[0] # Dataset name
    ts = load_time_series.from_iot_to_struct(data, host_ids[dataset], dataset) #

    # split time series into train and validation
    train, test = ts.train_test_split(train_test_ratio=0.75)

    # Infer periodicity:
    period, msg = arima_model.decompose(ts.data[0], nhist=500, folder=os.path.join(folder, "decompose"), nsplits=50)
    latex_str += msg
    latex_str += arima_model.make_report(os.path.join(folder, "decompose"), write=False)


    # select number of trees and history parameter:
    n_req, nr_tree, best_train_mse = train_model_CV(train, n_fold=2, windows=[5, 10, 25, 50, 75, 100, 150])
    #n_req, nr_tree, best_train_mse = 10, 500, 0.006
    latex_str += "Estimated parameters: history = {0}, n. trees = {1} \\\\ \n".format(n_req, nr_tree)


    # use selected parameters to forecast trainning data:
    ts.history = n_req * ts.request
    data = regression_matrix.RegMatrix(ts)
    data.create_matrix()
    data.train_test_split()

    model = frc_class.CustomModel(RandomForestRegressor, n_estimators=nr_tree, n_jobs=24)
    model, _,_,_ = data.train_model(frc_model=model)

    frc, _ = data.forecast(model, idx_rows=data.idx_train)
    train_mse = mean_squared_error(frc, data.trainY)

    frc, _ = data.forecast(model, idx_rows=data.idx_test)
    test_mse = mean_squared_error(frc, data.testY)

    latex_str += "Best CV error: {0}, train error for estimated parameters: {1}, " \
                 "test error with estimated parameters {2} \\\\ \n".format(best_train_mse, train_mse, test_mse)

    err_all = forecasting_errors(data)
    column_names = [("MAE", "train"), ("MAPE", "train"), ("MAE", "test"), ("MAPE", "test")]
    res_all = data_frame_res(err_all, column_names, ts)
    print(res_all)

    latex_str += res_all.to_latex()

    data.plot_frc(n_frc=10, n_hist=10, folder=folder)
    latex_str += my_plots.include_figures_from_folder(folder)

    my_plots.print_to_latex(latex_str, check=False, file_name="IoT_trees", folder=folder)

    return train_mse, test_mse





def data_frame_res(columns, column_names, ts):
    res = []
    for col, name in zip(columns, column_names):
        res.append(pd.DataFrame(col, index=[t.name for t in ts.data], columns=[name]))

    res = pd.concat(res, axis=1)
    return res


def forecasting_errors(data):
    train_mae = data.mae(idx_rows=data.idx_train)
    train_mape = data.mape(idx_rows=data.idx_train)
    test_mae = data.mae(idx_rows=data.idx_test)
    test_mape = data.mape(idx_rows=data.idx_test)

    return train_mae, train_mape, test_mae, test_mape


def train_model_CV(data, n_fold=5, windows=[5, 10, 25, 50, 75, 100, 150],
                       n_trees=[500, 1000, 2000, 3000], f_horizon=1):

        scores = np.zeros((len(windows), len(n_trees), n_fold))

        for w_ind in range(0, len(windows)):
            # obtain the matrix from  the time series data with a given window-size
            data.history = windows[w_ind] * data.request
            mat = regression_matrix.RegMatrix(data)
            mat.create_matrix(f_horizon)
            w_train = mat.X
            y_wtrain = mat.Y
            #(w_train, y_wtrain) = windowize(data, windows[w_ind], f_horizon=f_horizon)

            # cross-validation
            r, c = w_train.shape
            kf = KFold(r, n_folds=n_fold)
            for tree_ind in range(0, len(n_trees)):
                reg = RandomForestRegressor(n_estimators=n_trees[tree_ind], n_jobs=24)
                n = 0
                for train_index, val_index in kf:
                    # getting training and validation data
                    X_train, X_val = w_train[train_index, :], w_train[val_index, :]
                    y_train, y_val = y_wtrain[train_index], y_wtrain[val_index]
                    # train the model and predict the MSE
                    reg.fit(X_train, y_train)
                    pred_val = reg.predict(X_val)
                    scores[w_ind, tree_ind, n] = mean_squared_error(pred_val, y_val)
                    n += 1
        m_scores = np.average(scores, axis=2)
        mse = m_scores.min()

        # select best window_size and best n_tree with smallest MSE
        (b_w_ind, b_tree_ind) = np.where(m_scores == mse)
        window_size, nr_tree = windows[b_w_ind], n_trees[b_tree_ind]

        return (window_size, nr_tree, mse)



def mean_squared_error(f, y):
    return np.mean((f-y)**2)


if __name__ == '__main__':
    main()
    # for period in range(10, 100, 10):
    #     ts = random_data.create_sine_ts(n_ts=1, period=period, min_length=2000, max_length=2000).data[0]
    #     ts = np.log(ts + np.arange(ts.shape[0])*abs(np.max(ts))/ts.shape[0])
    #     ts[np.isnan(ts)] = np.mean(ts)
    #     decompose(ts)
    #     find_fft_period(ts)

