# coding: utf-8
"""
Created on 30 September 2016
@author: Parantapa Goswami, Yagmur Gizem Cinar
"""
import os
import pandas as pd
import linecache

from collections import defaultdict

'''
get_data: method to read certain metrics from the data file.
@param: FILE_NAME is the path to the data file
@param: line_indices is the list of line numbers (indices) corresponding to time series to be retrieved
@return: data matrix is in the size of [number of instances (n) , number of time series (length of line_indices)]
@return: metric_ids, host_ids, header_names
'''


def get_data(file_name, line_indices="all", header=True):
    """
    Read data from InternetOfThings dataset.

    :param file_name:  .csv filename with raw data
    :type file_name: string
    :param line_indices: indices of lines to read from file.  Lines are enumerated from 1. If "all", read the whole file
    :param header: Specifies if the file contains a header row
    :type header: bool
    :return: data, metric_ids_dict, host_ids, header_names
    :rtype: tuple
    """

    if line_indices=="all":
        # read the whole file
        return read_all_lines(file_name, header)

    # otherwise, only specified rows
    return read_random_lines(file_name, line_indices, header)


def read_all_lines(file_name, header):
    """
    Read (all lines) from file in InternetOfThings format.

    :param file_name:  .csv filename with raw data
    :type file_name: string
    :param header: Specifies if the file contains a header row
    :type header: bool
    :return: data - list of pandas.Series; metric_dict - dictionary, maps ts numbers device names; host_ids - dictionary,
    maps host names to devices; header_names stores columns names, read from the header row
    :rtype: tuple
    """
    header_names = []

    if header:
        a = linecache.getline(file_name, 1)
        b = a.split(',')
        header_names = b[0:7]

    metric_ids = []  # stores time series id's against line numbers
    host_ids = defaultdict(list)  # stores list of corresponding line numbers for each dataset

    data = [] # empty matrix to store data
    nline = 1 + header
    new_line = linecache.getline(file_name, nline)
    while len(new_line) > 0:
        # retrieve the fields of a line
        b = new_line.split(',')
        metric_ids.append(b[0])
        host_ids[b[1]].append(nline - (1 + header) )
        # values of the current metric, v1..vn
        V, T = [], []
        if "\n" in b:
            b.remove("\n")
        for i in range(8, len(b)):
            c = b[i]
            vst = c.split(":")  # value:status:time
            v, s = vst[0], vst[1]
            t = ":".join(vst[2:])

            if ":" in t or "-" in t or "/" in t:
                T.append(pd.to_datetime(t,infer_datetime_format=True))
            else:
                T.append(float(t))
            V.append(float(v))

        # append current values to the data matrix
        data.append(pd.Series(V, index=T, name=b[0]))
        nline += 1
        new_line = linecache.getline(file_name, nline)

    metric_ids_dict = {k:v for k, v in enumerate(metric_ids)}
    return data, metric_ids_dict, host_ids, header_names


def read_random_lines(file_name, line_indices, header):
    """
    Read specific lines from file in InternetOfThings format.

    :param file_name:  .csv filename with raw data
    :type file_name: str
    :param line_indices: indices of lines to read from file.  Lines are enumerated from 1
    :type line_indices: list
    :param header: Specifies if the file contains a header row
    :type header: bool
    :return: data - list of pandas.Series; metric_ids_dict - dictionary, maps ts numbers device names; host_ids - dictionary,
    maps host names to devices; header_names stores columns names, read from the header row
    :rtype: tuple
    """
    # This block processes the header line, if it exits
    # header = True means the first line of the csv file in the columns 1 to 8 are variable names
    header_names = []
    if header:
        a = linecache.getline(file_name, 1)
        b = a.split(',')
        header_names = b[0:7]

    host_ids = defaultdict(list)  # stores list of corresponding line numbers for each dataset
    metric_ids = {}  # stores time series id's against line numbers

    data = []  # empty matrix to store data
    nline = 0
    for line_index in line_indices: # line_indices: input the time series correspond to the same device
        # retrieve the fields with metadata
        a = linecache.getline(file_name, line_index)
        b = a.split(',')

        metric_ids[line_index] = b[0]
        host_ids[b[1]].append(nline)
        nline += 1
        # values of the current metric, v1..vn
        V, T, = [], []
        for i in range(8, len(b)):
            c = b[i]
            v, s, t = c.split(":")  # value:status:time
            V.append(float(v))
            T.append(float(t)) # time is in unix format
        # append current values to the data matrix
        data.append(pd.Series(V, index=T, name=b[0]))

    return data, metric_ids, host_ids, header_names