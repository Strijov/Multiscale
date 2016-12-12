# Introduciing gaps/missing values

# ARG1: input file (original data)
# ARG2: output data file (data with NAN values as missing)
# ARG3: output mask file (mask file whose size is equal to the data and is zero in the indices for which we will have missing values in the data and one otherwise, i.e. it is a binary vector)
# ARG4: missig or gaps. This argument should be of type string, either "missing" or "gaps"
# ARG5: percentage of the values to be replaced by NANs, i.e. the amount of data that should dissapear(!) either in the form of missing values or gaps. An integer between 1 and 100 is expected!
# ARG6: log file (for gaps only)

# note that the gaps are of a random size between 5 and 100. We continue introducing gaps untill we get enough NAN values (equal to ARG5)
# note that the if we choose to have more than 90% of missing values as gaps, it will take a while to have them!

import numpy as np
import csv
import random
import sys
import cPickle as pickle

# function to read the original electricity data in the CSV format
def load_pse_data(datafile):
    with open(datafile, 'rU') as f:
        mycsv = csv.reader(f, delimiter=';')
        mycsv = list(mycsv)
    mycsv = mycsv[1:]
    data = np.zeros(len(mycsv))
    for idx in range(0,len(mycsv)):
        data[idx] = mycsv[idx][3]
    return data

# function to introduce the missing values and save the output file as well as the mask file.
def introduce_missing(data,perc):
    missing_data = data.copy()
    data_size = data.shape[0]
    mask = np.ones(data_size)
    nb_missing = int(data_size*perc/100.0)
    print "data_size",data_size
    print "nb missing ",nb_missing
    i = 0;
    while i < nb_missing:
        r = np.random.randint(0,data_size)
        if mask[r] == 0:
            continue
        mask[r] = 0
        missing_data[r] = np.NAN
        i = i + 1
    pickle.dump(missing_data, open(sys.argv[2],"wb"), pickle.HIGHEST_PROTOCOL)
    pickle.dump(mask, open(sys.argv[3],"wb"), pickle.HIGHEST_PROTOCOL)
    

# function to introduce the gaps and save the output file as well as the mask file.
def introduce_gaps(data,perc):
    missing_data = data.copy()
    data_size = data.shape[0]
    mask = np.ones(data_size)
    nb_missing = int(data_size*perc/100.0)
    print "data_size",data_size
    print "nb values to be missed",nb_missing
    sum_gap_len = 0.0
    gap_count = 0.0
    i = 0;
    while i < nb_missing:
        r = np.random.randint(0,data_size)
        l = np.random.randint(5,101)
        if r+l >= data_size or mask[r:r+l].sum()<>l:
            continue
        mask[r:r+l] = 0
        missing_data[r:r+l] = np.NAN
        i = i + l
        gap_count = gap_count + 1
        sum_gap_len = sum_gap_len + l
    print "actuall missing",sum_gap_len
    print "actuall missing percentage",sum_gap_len/data_size
    print "average len of gaps",sum_gap_len/gap_count
    print "nb gaps",gap_count
    log_file = open(sys.argv[6],"wb")
    log_file.write("nb gaps: " + str(sum_gap_len/gap_count)+'\n')
    log_file.write("average len of gaps: " + str(gap_count))
    log_file.close()
    pickle.dump(missing_data, open(sys.argv[2],"wb"), pickle.HIGHEST_PROTOCOL)
    pickle.dump(mask, open(sys.argv[3],"wb"), pickle.HIGHEST_PROTOCOL)
    
def main():
    data = load_pse_data(sys.argv[1])
    perc = int(sys.argv[5])
    if sys.argv[4].lower() == "missing":
        introduce_missing(data,perc)
    elif sys.argv[4].lower() == "gaps":
        introduce_gaps(data,perc)
    else:
        print "The fourth arguement should be either MISSING or GAPS."
        
if __name__ == '__main__':
    main()   
