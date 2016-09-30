import csv
import os

from collections import namedtuple

from LoadAndSaveData import load_time_series

tsStruct = namedtuple('tsStruct', 'data request history name readme')
HEADERS = ["TS_id", "Device_id", "controlPointId", "n", "firstTime", "lastTime", "warn", "crit"]

def write_ts(ts_struct=None, file_name="data.csv"):

    if ts_struct is None:
        load_raw = not os.path.exists(os.path.join("ProcessedData", "EnergyWeather_orig_train.pkl"))
        ts_struct = load_time_series.load_all_time_series(datasets='EnergyWeather', load_raw=load_raw,
                                                               name_pattern="orig_train")[0]

    with open(file_name, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        rows = [HEADERS]
        for ts in ts_struct.data:
            rows.append(ts_to_csv_row(ts, ts_struct.name))

        csvwriter.writerows(rows)
        csvfile.close()

def ts_to_csv_row(ts, dataset):
    row_list = [ts.name, dataset, "nan", str(len(ts)), ts.index[0], ts.index[-1], "nan", "nan"]
    time = ts.index
    ts = ts.as_matrix()
    for i, tsi in enumerate(ts):
        row_list.append(str(tsi)+":"+"nan"+":"+str(time[i]))
    return row_list


def from_iot_to_struct(ts_list, idx, dataset):
    """ Converts data from IoT output to tsStruct. Request is single point for every ts and horizon is unknown"""
    request, ts = [], []
    for i in idx:
        request.append(ts_list[i].index[1] - ts_list[i].index[0])
        ts.append(ts_list[i])

    return tsStruct(ts, max(request), None, dataset, "")


if __name__ == '__main__':
    write_ts()