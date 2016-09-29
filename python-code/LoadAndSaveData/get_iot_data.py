# required packages
import numpy as np
import linecache

'''
get_data: method to read certain metrics from the data file.
@param: FILE_NAME is the path to the data file
@param: line_indices is the list of line numbers (indices) corresponding to time series to be retrieved
@return: data matrix is in the size of [number of instances (n) , number of time series (length of line_indices)]
@return: metric_ids, host_ids, header_names
'''


def get_data(file_name, line_indices=(14, 15)):
    # This block processes the header line, if it exits
    header = True  # True means the first line of the csv file in the columns 1 to 8 are variable names
    #if header == True:
    a = linecache.getline(file_name, 1)
    b = a.split(',')
    header_names = b[0:7]
    # dictionaries to store metric ids and host ids against the line indices
    metric_ids = dict.fromkeys(line_indices)
    host_ids = dict.fromkeys(line_indices)

    # empty matrix to store data
    data = []

    # line_indices: input the time series correspond to the same device
    for line_index in line_indices:
        # retrieve  different fields of a line
        a = linecache.getline(file_name, line_index)
        b = a.split(',')

        # stores the metricID and hostID against line numbers
        #if header == True:
        metric_ids[line_index] = b[0]
        host_ids[line_index] = b[1]
        # values of the current metric, v1..vn
        V = []
        for i in range(8, len(b)):
            c = b[i]
            v, s, t = c.split(":")  # value:status:time
            V.append(float(v))
        # append current values to the data matrix
        data.append(V)

    # convert data to numpy format to be used later by sk-learn mathods
    data = np.array(data)
    data = np.transpose(data)
    # returned data matrix is in the size of [number of instances (n) , number of time series (length of line_indices)]
    # each column contains the sequence of a time series
    return (data, metric_ids, host_ids, header_names)