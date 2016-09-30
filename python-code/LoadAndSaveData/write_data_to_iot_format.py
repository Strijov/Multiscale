import csv
import os

from LoadAndSaveData import load_time_series


HEADERS = ["TS_id", "Device_id", "controlPointId", "n", "firstTime", "lastTime", "warn", "crit"]

def write_ts(ts_struct=None, file_name="data.csv"):
    """
    Write time series into .csv file in IoT format

    :param ts_struct: data to be written into file
    :type ts_struct: TsStruct
    :param file_name: output filename
    :type file_name: string
    :return:
    :rtype: None
    """

    if ts_struct is None:
        load_raw = not os.path.exists(os.path.join("ProcessedData", "EnergyWeather_orig_train.pkl"))
        ts_struct = load_time_series.load_all_time_series(datasets='EnergyWeather', load_raw=load_raw,
                                                               name_pattern="orig_train")[0]

    with open(file_name, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        rows = [HEADERS]
        for ts in ts_struct.data:
            rows.append(ts_to_csv_row(ts, ts_struct.name))

        csvwriter.writerows(rows) # expects a list of rows. Each row is also represented with a list
        csvfile.close()

def ts_to_csv_row(ts, dataset):
    """
    Converts time series into csv strings. Status is replaced with "nan"s. Device name is replaced with ts.name

    :param ts: input time series
    :type ts: TsStruct
    :param dataset: dataset name, replaces host name in IoT format
    :type dataset: string
    :return: strings to write into file as a csv row
    :rtype: list
    """
    row_list = [ts.name, dataset, "nan", str(len(ts)), ts.index[0], ts.index[-1], "nan", "nan"]
    time = ts.index
    ts = ts.as_matrix()
    for i, tsi in enumerate(ts):
        row_list.append(str(tsi)+":"+"nan"+":"+str(time[i]))
    return row_list



if __name__ == '__main__':
    write_ts()