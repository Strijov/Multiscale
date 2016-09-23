import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

tsMiniStruct = namedtuple('tsMiniStruct', 's norm_div norm_subt name index')


def plot_forecast(ts, forecasts, idx_ts=None, idx_frc=None):
    if idx_frc is None:
        idx_frc = range(len(ts.s))
    if idx_ts is None:
        idx_ts = range(len(ts.s))



    plt.figure(figsize=(7, 5))
    idx_time = range(len(ts.s))
    idx_frcb = [i in idx_frc for i in idx_time]
    idx_tsb = [i in idx_ts for i in idx_time]
    plt.plot(np.compress(idx_tsb, idx_time, axis=0), ts.s[idx_ts], label='Time series', linewidth=2, ls='-', c='b')
    plt.plot(np.compress(idx_frcb, idx_time, axis=0), np.compress(idx_frcb, forecasts, axis=0), label='Forecast', linewidth=2, ls='-', c='k')
    plt.axvline(idx_time[idx_frc[0]], linewidth=1, ls='--', c='k')

    plt.legend(loc='best', prop={'size': 16})
    plt.xlabel('Time')
    plt.ylabel(ts.name)
    plt.title('Forecasting results')
    plt.xscale('linear')
    plt.show()


