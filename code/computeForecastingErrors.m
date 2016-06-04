function [testMAPE, trainMAPE, model] = computeForecastingErrors(ts, ...
                                                model, ...
                                                alpha_coeff,...
                                                idxTrain,...
                                                idxVal)


% The function fits the foreasting model to the target time series on the
% train set and applies it to foreast the test set
% Input:
% ts - basic time series structure (for more details see createRegMatrix.m)
%       ts.matrix [m x n] stores the design matrix, where the last [m x
%       ts.deltaTr] columns store the target variable
% model - basic model structure (see Systemdocs.doc)
%       model.handle: handle to the forecasting model, 
%       model.params: model parameters
%       model.testErrors, model.trainErrors: forecasting errors (symMAPE)
%       model.forecasted_y: [(T- (n - ts.deltaTr) x 1)] forcasted values of the
%       original time series ts.x [T x 1]
% alpha_coeff - test to train ratio in train-test-validation split. Takes values in 
%       range [0, 1]. "0" coressponds to no test set, only validetion FIXIT
% idxTrain - row indices of ts.matrix
% idxVal   - row indices of ts.matrix. FIXIT Validation set is used as test
%        set...

if nargin < 4
    [idxTrain, ~, idxVal] = TrainTestSplit(size(ts.X, 1), alpha_coeff);
end

[forecastY, trainForecastY, model] = feval(model.handle, ts.X(idxVal, :), model, ...
                            ts.X(idxTrain, :), ts.Y(idxTrain, :)); 

% Remember that forecasts and ts.matrix are normalized
trainMAPE = calcSymMAPE(ts.Y(idxTrain, :), trainForecastY);
testMAPE = calcSymMAPE(ts.Y(idxVal, :), forecastY);
forecasts = zeros(size(ts.Y));
forecasts(idxTrain, :) = trainForecastY;
forecasts(idxVal, :) = forecastY;
forecasts = unravel_target_var(forecasts);
% Denormalize forecasts:
model.forecasted_y = forecasts*ts.norm_div + ts.norm_subt;
model.testError = testMAPE;
model.trainError = trainMAPE;




end

% DISCUSS: Compute quality of forecast in extra function? Additional quality/error
% functions?


function vecY = unravel_forecasts(trainY, testY, idxTrain, idxTest, deltaTr)

vecY = zeros((numel(idxTrain) + numel(idxTest))*deltaTr, 1);
idxTrain = repmat(idxTrain(:), 1, deltaTr)' + repmat(0:deltaTr-1, numel(idxTrain), 1)';
idxTrain = idxTrain(:);
idxTest = repmat(idxTest(:), 1, deltaTr)' + repmat(0:deltaTr-1, numel(idxTest), 1)';
idxTest = idxTest(:);

vecY(idxTrain) = reshape((trainY));
vecY(idxTest) = testY;
    
end


         
        
        
    