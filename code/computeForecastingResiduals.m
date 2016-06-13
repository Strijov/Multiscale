function [testRes, trainRes, model] = computeForecastingResiduals(ts, ...
                                                model, ...
                                                idxTrain,...
                                                idxTest)


% The function fits the foreasting model to the target time series on the
% train set and applies it to foreast the test set
% Input:
% ts - basic time series structure (for more details see createRegMatrix.m)
%       ts.X [m x n] and ts.Y [m x ts.deltaTr] store the independent and
%        the target variables of the design matrix
% model - basic model structure (see Systemdocs.doc)
%       model.handle: handle to the forecasting model, 
%       model.params: model parameters
%       model.testErrors, model.trainErrors: forecasting errors (symMAPE)
%       model.forecasted_y: [(T- (n - ts.deltaTr) x 1)] forcasted values of the
%       original time series ts.x [T x 1]
% idxTrain - indices of train objects from design matrix
% idxTest   - test (validation) indices of test objects. FIXIT Validation set is used as test
%        set...


if nargin == 2
    idxTrain = 1:size(ts.X, 1);
    idxTest = [];
end
if nargin == 3
    trainTestValRatio = idxTrain;
    [idxTrain, idxTest, ~] = TrainTestSplit(size(ts.X, 1), trainTestValRatio);
end

[forecastY, matTrainForecastY, model] = feval(model.handle, ts.X(idxTest, :), model, ...
                            ts.X(idxTrain, :), ts.Y(idxTrain, :)); 

% Remember that forecasts, ts.X and ts.Y are normalized, while ts.x are not!
vecForecasts = zeros(size(ts.Y));
vecForecasts(idxTrain, :) = matTrainForecastY;
vecForecasts(idxTest, :) = forecastY;

% Unravel forecasts from matrices to vecors and denormalize forecasts:
% unravel_target_var returns a cell array of size [1 x nTimeSeries]
model.forecasted_y = unravel_target_var(vecForecasts, ts.deltaTr, ts.norm_div, ts.norm_subt);

% compute frc residuals for each time series (cell array [1 x nTimeSeries])
residuals = calcResidualsByTs(model.forecasted_y, ts.x, ts.deltaTp);
% split them into 2 arrays of residuals, each of size [1 x nTimeSeries]
nPredictions = size(ts.Y, 2)/sum(ts.deltaTr);
[testRes, trainRes] = splitForecastsByTs(residuals, idxTrain, idxTest, ...
                                                  ts.deltaTr*nPredictions);

[testFrc, trainFrc] = splitForecastsByTs(model.forecasted_y, idxTrain, idxTest, ...
                                                  ts.deltaTr*nPredictions);                                              
[testY, trainY] = splitForecastsByTs(ts.x, idxTrain, idxTest, ...
                                                  ts.deltaTr*nPredictions);                                              
model.testError = cellfun(@calcSymMAPE, testY, testFrc);
model.trainError = cellfun(@calcSymMAPE, trainY, trainFrc);


end

% DISCUSS: Compute quality of forecast in extra function? Additional quality/error
% functions?



         
        
        
    