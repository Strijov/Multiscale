The “original” directory contains the original (not noisy) data: a subsampled version of Polish electricity data for 12 years (1/1/2002 to 31/7/2014). The subsampling is done such that we have 12 data points per day. Note that the initial data is recorded on a 15-minute basis.




The “gap5” directory contains a missing file and a mask file. The missing file is the original data (taken form the “original” directory) with 5% of missing values AS GAPS. The mask file is a binary vector which has zeros where we have a missing value and ones otherwise. The log file shows the number of gaps and the average length of the gaps.

The “gap10” and “gap15” are similar to “gap5” with 10% and 15% of missing values AS GAPS respectively.


The “missing5”, “missing10” and “missing15” are similar to gaps except that the missing values are not gaps anymore but random points.