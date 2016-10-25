import matplotlib.pyplot as plt
import numpy as np

import re
import os
import glob




def formatted_plot(y, x=None, ls=None, title=None, legend=None, xlabel=None, ylabel=None):
    if ls is None:
        ls = "-"
    if legend is None:
        label = ""
    else:
        label = legend
    if x is None:
        plt.plot(y, ls=ls, label=label)
    else:
        plt.plot(x, y, ls=ls, label=label)

    if not title is None:
        plt.title(title)
    if not legend is None:
        plt.legend(loc='best', prop={'size': 16})
    if not xlabel is None:
        plt.xlabel(xlabel)
    if not ylabel is None:
        plt.ylabel(ylabel)

    return plt

def plot_forecast(ts, forecasts, idx_ts=None, idx_frc=None, filename=None, folder=""):
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
    if filename is None:
        plt.show()
    else:
        if not os.path.exists(folder):
            os.mkdir(folder)

        filename = os.path.join(folder, filename)
        plt.savefig(filename)
        plt.close()


def imagesc(matrix, xlabel="", ylabel="", xticks=None, yticks=None):

    plt.imshow(matrix, interpolation='none')
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not xticks is None:
        plt.gca().set_xticklabels(xticks)
    if not yticks is None:
        plt.gca().set_yticklabels(yticks)


    return plt


def input_latex_headers():
    latex_header = '\\documentclass[12pt]{article}\n' + \
                   '\\extrafloats{100}\n' + \
                   '\\usepackage{a4wide}\n' + \
                   '\\usepackage{booktabs}\n' + \
                   '\\usepackage{multicol, multirow}\n' + \
                   '\\usepackage[cp1251]{inputenc}\n' + \
                   '\\usepackage[russian]{babel}\n' + \
                   '\\usepackage{amsmath, amsfonts, amssymb, amsthm, amscd}\n' + \
                   '\\usepackage{graphicx, epsfig, subfig, epstopdf}\n' + \
                   '\\usepackage{longtable}\n' + \
                   '\\graphicspath{ {../fig/} {../} {fig/} {decompose/}}\n' + \
                   '\\begin{document}\n\n'
    latex_end = "\\end{document}"

    return latex_header, latex_end

def save_to_latex(df_list, df_names=None, file_name=None):

    if file_name is None:
        file_name = "test_latex_output"
    file_name = file_name + ".tex"

    latex_header, latex_end = input_latex_headers()

    if df_names is None:
        df_names = ["Table" + str(i + 1) for i in range(len(df_list))]

    with open(file_name, "w+") as f:
        f.write(latex_header)
        for i, df in enumerate(df_list):
            f.write("\n"+ check_text_for_latex(df_names[i])  + "\\\ \n")
            f.write(df.to_latex())
            f.write("\n")

        f.write(latex_end)
        f.close()



def check_text_for_latex(text):

    text = re.sub("_", "\_", text)
    return text


def plot_seasonal_trend_decomposition(ts, trend, seasonal=None, residual=None, folder="fig"):

    if not os.path.exists(folder):
        os.makedirs(folder)

    if seasonal is None:
        seasonal = np.zeros_like(ts)
    if residual is None:
        residual = ts - trend - seasonal
    plt.subplot(411)
    plt.plot(ts, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "decomposition.png"))
    plt.close()


def print_to_latex(inputs, file_name=None, check=True, folder=""):
    latex_header, latex_end = input_latex_headers()
    if check:
        inputs = check_text_for_latex(inputs)
    latex_str = latex_header + "\n" + inputs + "\n" + latex_end

    if file_name is None:
        file_name = "test_latex_output"
    file_name = file_name + ".tex"
    file_name = os.path.join(folder, file_name)

    with open(file_name, "w+") as f:
        f.write(latex_str)
        f.close()


def include_figures_from_folder(folder):

    if not os.path.exists(folder):
        os.mkdir(folder)

    fnames = glob.glob(os.path.join(folder, "*.png"))
    latex_str = ""
    for figname in fnames:
        #figname = "p"+str(p)+"q"+str(q)
        figname = "/".join(figname.split(os.path.sep)[1:]) #join(figname.split(os.path.sep))
        if " " in figname:
            figname = "\"" + figname + "\""
        latex_str += "\n"
        latex_str += "\\begin{figure}\n"
        latex_str += "\includegraphics[width=0.7\\textwidth]{" + figname + "}\n"
        latex_str += "\caption{"+ check_text_for_latex(figname) + ".} \label{fg:"+figname+"}\n"
        latex_str += "\end{figure}\n"

    return latex_str