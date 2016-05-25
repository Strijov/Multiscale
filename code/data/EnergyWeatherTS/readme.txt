The set consists of energy consumption time series, recorded per hour 
 and weather time series, measured daily. Energy time series are stored in 
'train*' and 'test*' files, weather time series - in 'weatherdata*' file.
 The dataset is present in original form (orig/), with artificially added 
 missing values(missing_value/) and varying sampling rate (varying_rate/).

Headers: 
1) Energy data: 1-Date, 2-Day of the week, 3-Indiator of the untypical day, 4-27 - hourly data
Fields 2--3 are unused.
2) Weather data: 1-Date, 2-Longitude, 3-Latitude, 4-Elevation,	5-Max Temperature (C), 6-Min Temperature (C), 
7-Precipitation (mm), 8-Wind (m/s), 9-Relative Humidity, 10-Solar (MJ/m^2). Fields 2--4 are not used for prediction.

The data is plit into training and testing folds:
Training: Time series of the hourly loads of the local power system from the period 1999-2001 - SL2.xls file - (1096x24=26304 data points)
Test: time series of the hourly loads of the Polish power system from the period 2002-2004 - PL.xls file - (1096x24=26304 data points)

The weather data can be separated into training and test  sets as:
Training: weather data for the period of 1999-2001: (1 - 1096 data points)
Test: weather data for the period of 2002-2004: (1097 - 2192 data points)
