# Multiscale time series forecasting

The main goal is to construct a forecasting model to predict the outputs of set of multiscale time series simultaneously.
The input data is a set of time series. Each time series contains records of some signal and has its own number of observations. The task is to predict future values of the signals within a given time range. 

The foreasting problem is reformulated into the problem of multivariate regression. The input time series comprise a design matrix with a set of columns corresponding to the independent variables and a set of columns corresponding to the target variables. Independent variables may also include feature transformations.


* For more information on the forecasting problem and the proposed method, see 'doc'. It contains the paper draft, presentations and reports for the project.
* Directories 'code' and 'python-code' contain matlab and python code for the project.