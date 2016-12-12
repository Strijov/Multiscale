This directory contains the polish electricity data.

“original” subdirectory
==========================
In this subdirectory you will find the 12-year data (from 1/1/2002 to 31/7/2014) of Polish electricity.
The file is named as “pse_12years_2hrs.csv” and contains a subsample of the original data.
The subsampling rate is 2 hours so we have 12 data points per day. Note that the original (not subsampled) data has a 96 data points per day (one data point every 15 minutes)

“gap_5” subdirectory
==========================
Using the data in the original subdirectory, we introduce some gaps such that, in total, 5% of the data being missed.
The length of the gaps are selected randomly between 5 and 100.
To introduce the gaps, we select a data point randomly, consider it as the starting point of the gap; then we select a random integer between 5 and 100 as the length of the gap. If there is not already a gap in that area, we introduce the gap, otherwise wi select another point and another integer between 5 and 100. We continue this process until we reach 5% of missing data.
There are 3 files in this directory: 1) “missing_pse_12years_2hrs.pkl” which contains the data where some values are NAN. Those NAN values are actually the missing values. 2) “mask_pse_12years_2hrs.pkl” is a binary vector of the same size as the data. It contains values of 1 when the data is not missing and 0 otherwise. 3) “log.txt” contains the average length of the gaps as well as the number of gaps.

“gap_10” subdirectory
==========================
The same as gap_5 but with 10% of missing.

“gap_15” subdirectory
==========================
The same as gap_5 but with 15% of missing.

“missing_5” subdirectory
==========================
Using the data in the original subdirectory, we introduce some missing values such that, in total, 5% of the data being missed. The missing values are randomly selected and are represented as NAN values. There are 2 files in this directory: 1) “missing_pse_12years_2hrs.pkl” which contains the data where some values are NAN. Those NAN values are actually the missing values. 2) “mask_pse_12years_2hrs.pkl” is a binary vector of the same size as the data. It contains values of 1 when the data is not missing and 0 otherwise.

“missing_10” subdirectory
==========================
The same as missing_5 but with 10% of missing.

“missing_10” subdirectory
==========================
The same as gmissing_5 but with 15% of missing.