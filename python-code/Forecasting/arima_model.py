from __future__ import print_function
import numpy as np
import pandas as pd
import os

from scipy import fft
from scipy import signal
import my_plots

import matplotlib.pyplot as plt
import statsmodels.tsa as tsa
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import product
from RegressionMatrix import random_data



def adjust_arima_model(sd, max_p_lags=20, max_q_lags=20, nsplits=0, nhist=100, nsteps=1, folder="fig"):
    # Compute ACF with alpha-confidence intervals
    acf, confint, qstat, p = tsa.stattools.acf(sd.resid, nlags=100, alpha=alpha, qstat=True)
    confint = confint - confint.mean(1)[:, None]

    # Find non-zero lags, where ACF  exceeds upper alpha-confidence level (candidates for q)
    q_lags = np.nonzero(acf > confint[:, 1])[0][1:]
    q_lags = q_lags[q_lags <= max_q_lags]

    # Compute ACF with alpha-confidence intervals
    pacf, pconfint = tsa.stattools.pacf(sd.resid, nlags=100, alpha=alpha)
    pconfint = pconfint - pconfint.mean(1)[:, None]

    # Find non-zero lags, where PACF  exceeds upper alpha-confidence level (candidates for p)
    p_lags = np.nonzero(pacf > pconfint[:, 1])[0][
             1:]  # p = np.nonzero(np.vstack((acf > confint[:, 1], acf < confint[:, 0])).any(axis=0))
    p_lags = p_lags[p_lags <= max_p_lags]
    rss = []
    # tscv = TimeSeriesSplit(n_splits=nsplits)
    predicted = np.zeros_like(ts)
    split_rss = []
    # train, _ = tscv.split(ts)
    # n_train = len(train)
    for p, q in product(p_lags, q_lags):
        try:
            for train, test in random_split_time_series(ts, nhist, nsteps, nsplits):  # train, test in tscv.split(ts):
                # train = np.hstack((train[-len(test):], test[:-1]))
                # test = test[-1:]
                model = tsa.arima_model.ARIMA(ts[train], order=(p, 0, q))
                results_AR = model.fit(disp=-1)
                # predicted[test] = model.predict(results_AR.params, start=len(train), end=len(test) + len(train)-1)
                predicted[test], err, intvl = results_AR.forecast(steps=len(test))
                split_rss.append(np.mean((predicted[test] - ts[test]) ** 2))

        except:
            continue
        rss.append((p, q, np.mean(split_rss)))  # sum((predicted - ts) ** 2)))

        if (predicted > 0).any():
            plt.plot(ts)
            plt.plot(predicted, color='red')
            plt.title('p: %.0f, q: %.0f, RSS: %.4f' % rss[-1])
            plt.savefig(os.path.join(folder, 'arima_p' + str(p) + 'q' + str(q) + '.png'))
            plt.close()

    _, _, rss_values = zip(*rss)
    best_pars = np.argmin(rss_values)
    p, q, rss = rss[best_pars]

    return p, q, rss

def decompose(ts, log_detrend=False, try_arima=False, alpha=0.01, max_p_lags=20, max_q_lags=20, nsplits=0, nhist=100, nsteps=1, folder="fig"):

    if isinstance(ts, pd.Series):
        ts = ts.as_matrix()
    if isinstance(ts, list):
        ts = np.array(ts)

    ts = np.squeeze(ts)
    detrended = signal.detrend(ts)

    # Compute ACF with alpha-confidence intervals
    acf, confint, qstat, p = tsa.stattools.acf(detrended, nlags=100, alpha=alpha, qstat=True)

    # Infer periodicity from ACF
    period = find_fft_period(acf, window_size=len(acf), folder=folder)
    msg = "Estimated period {0} with std {1} \\\\ \n".format(np.mean(period), np.std(period))
    period = np.mean(period)
    period = max([1, int(period)])

    sd = seasonal_decompose(ts, freq=period, model="additive")

    sd.resid = signal.detrend(ts - sd.seasonal)
    sd.trend = ts - sd.resid - sd.seasonal

    my_plots.plot_seasonal_trend_decomposition(ts, sd.trend, sd.seasonal, sd.resid, folder)

    plot_acf_pacf(ts, "Original", folder)
    plot_acf_pacf(sd.trend, "Trend", folder)
    plot_acf_pacf(sd.seasonal, "Seasonal", folder)
    plot_acf_pacf(sd.resid, "Residuals", folder)
    plt.clf()

    if try_arima:
        adjust_arima_model(sd, max_p_lags=max_p_lags, max_q_lags=max_q_lags, nsplits=nsplits, nhist=nhist, nsteps=nsteps, folder=folder)


    # deseasoned = ts[period:] - ts[:-period]
    # plt.plot(ts[period:] - ts[:-period])

    return period, msg





def find_fft_period(ts, window_size=1024, step = 100, folder=None):

    i = 0
    period = []
    while i + window_size <= len(ts):
        X = fft(ts[i:i+window_size], window_size)
        max_freq = abs(X[1:window_size//2]).argmax() + 1
        period.append(window_size/max_freq - 1)
        i += step
        plt.plot(np.divide(window_size, range(1, window_size // 2)) - 1, abs(X[1:window_size // 2]))


    #period = np.mean(period)

    if not folder is None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(os.path.join(folder, "fft.png"))
    else:
        plt.close()

    return period


def split_time_series(ts, nhist, nsteps):

    for i in range(0, len(ts)-nsteps-nhist, nsteps):
        yield range(i, i+nhist), range(i+nhist, i+nsteps+nhist)

def random_split_time_series(ts, nhist, nsteps, nsplits):

    ts_idx = range(len(ts)-nsteps-nhist)
    np.random.shuffle(ts_idx)
    ts_idx = ts_idx[:nsplits]

    for i in ts_idx:
        yield range(i, i+nhist), range(i+nhist, i+nsteps+nhist)


def plot_acf_pacf(ts, title="_", folder="fig"):
    if not os.path.exists(folder):
        os.mkdir(folder)

    ts = pd.Series(ts)
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title)
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts.values.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts, lags=40, ax=ax2)
    fig.savefig(os.path.join(folder, "acf_"+title+".png"))
    plt.close()



def make_report(folder="fig/sine+trend/", fname="ts_arima_analisys", write=True):
    import glob

    latex_str = ""
    folder0 = folder.split(os.path.sep)[-1]
    # add decomposition plot
    latex_str += "\n"
    latex_str += "\\begin{figure}\n"
    latex_str += "\includegraphics[width=0.9\\textwidth]{"+folder0+"/decomposition.png}\n"
    latex_str += "\caption{Decomposition of time series into trend and seasonal component).} \label{fg:decomp}\n"
    latex_str += "\end{figure}\n"

    # add acf plots
    latex_str += "\n"
    latex_str += "\\begin{figure}\n"
    latex_str += "\subfloat{\includegraphics[width=0.5\\textwidth]{"+folder0+"/acf_Original.png}}\n"
    latex_str += "\subfloat{\includegraphics[width=0.5\\textwidth]{"+folder0+"/acf_Trend.png}}\\\ \n"
    latex_str += "\subfloat{\includegraphics[width=0.5\\textwidth]{"+folder0+"/acf_Seasonal.png}}\n"
    latex_str += "\subfloat{\includegraphics[width=0.5\\textwidth]{"+folder0+"/acf_Residuals.png}}\n"
    latex_str += "\caption{ACF and PACF for componets of the time series.} \label{fg:acf}\n"
    latex_str += "\end{figure}\n"

    # for all p-q pairs add plots of arma forecasts
    fnames = glob.glob(os.path.join(folder, "arima_p*.png"))
    for figname in fnames:
        #figname = "p"+str(p)+"q"+str(q)
        figname = figname.split(os.path.sep)[-1] #"/".join(figname.split(os.path.sep))
        latex_str += "\n"
        latex_str += "\\begin{figure}\n"
        latex_str += "\includegraphics[width=0.7\\textwidth]{" + figname + "}\n"
        latex_str += "\caption{Random forecasts with ARMA($p, q$).} \label{fg:"+figname+"}\n"
        latex_str += "\end{figure}\n"

    if write:
        my_plots.print_to_latex(latex_str, os.path.join(folder, fname), check=False)

    return latex_str



# def decompose(ts, log_detrend=False, alpha=0.01, max_p_lags=50, max_q_lags=50):
#     ts = ts.as_matrix()
#     detrended = signal.detrend(ts)
#     trend = ts - detrended
#     acf, confint, qstat, p = statsmodels.tsa.stattools.acf(detrended, nlags=100, alpha=alpha, qstat=True)
#
#     period = arima_model.find_fft_period(acf, window_size=len(acf))
#     confint = confint - confint.mean(1)[:, None]
#
#     #p = np.nonzero(np.vstack((acf > confint[:, 1], acf < confint[:, 0])).any(axis=0))
#     q_lags = np.nonzero(acf > confint[:, 1])[0][1:]
#     q_lags = q_lags[q_lags < 50]
#
#     pacf, pconfint = statsmodels.tsa.stattools.pacf(detrended, nlags=100, alpha=alpha)
#     pconfint = pconfint - pconfint.mean(1)[:, None]
#     p_lags = np.nonzero(pacf > pconfint[:, 1])[0][1:]
#     p_lags = p_lags[p_lags < 50]
#
#     for p, q in product(p_lags, q_lags):
#         model = ARIMA(ts, order=(p, 0, q))
#         results_AR = model.fit(disp=-1)
#         plt.plot(ts)
#         plt.plot(results_AR.fittedvalues, color='red')
#         plt.title('p: %.0f, q: %.0f, RSS: %.4f' % (p, q, sum((results_AR.fittedvalues - ts) ** 2)))
#         plt.savefig('fig/p'+str(p)+'q'+str(q)+'.png')
#         plt.close()
#
#     period = arima_model.find_fft_period(acf, window_size=len(acf))
#
#     sd = seasonal_decompose(ts, freq=int(period))
#     if np.isnan(sd.trend).any():
#         sd.resid = signal.detrend(ts - sd.seasonal)
#         sd.trend = ts - sd.resid - sd.seasonal
#
#     my_plots.plot_seasonal_trend_decomposition(ts, sd.trend, sd.seasonal, sd.resid)
#
#     arima_model.plot_acf_pacf(ts).show()
#     arima_model.plot_acf_pacf(sd.trend).show()
#     arima_model.plot_acf_pacf(sd.seasonal).show()
#     arima_model.plot_acf_pacf(sd.resid).show()
#     # mdl = statsmodels.tsa.ar_model.AR(ts)
#     # nlags = mdl.select_order(100, 'aic', method='cmle')
#     #mdl = sm.tsa.ARMA(ts, (2,0)).fit()
#
#     dw_trend = sm.stats.durbin_watson(sd.resid)
#
#     period = arima_model.find_fft_period(ts)
#     arima_model.plot_acf_pacf(ts)
#
#     return None




if __name__ == '__main__':
    ts = random_data.create_sine_ts(n_ts=1, period=24, min_length=2000, max_length=2000).data[0]
    ts = np.log(ts + np.arange(ts.shape[0])*abs(np.max(ts))/ts.shape[0])
    ts[np.isnan(ts)] = np.mean(ts)
    adjust_arima_model(ts, nhist = 500, nsteps=1, nsplits=50)
    # for period in range(10, 100, 10):
    #     ts = random_data.create_sine_ts(n_ts=1, period=period, min_length=2000, max_length=2000).data[0]
    #     ts = np.log(ts + np.arange(ts.shape[0])*abs(np.max(ts))/ts.shape[0])
    #     ts[np.isnan(ts)] = np.mean(ts)
    #     decompose(ts)
    #     find_fft_period(ts)
