function [testMAPE, trainMAPE, model] = ComputeForecastingErrors(ts, ...
                                                alpha_coeff, model, ...
                                                idxTrain,...
                                                idxVal)



if nargin < 4
    [idxTrain, ~, idxVal, idxX, idxY] = FullSplit(size(ts.matrix, 1), ...
                            size(ts.matrix, 2), alpha_coeff, ts.deltaTr);
else
    idxX = 1:size(ts.matrix, 2) - ts.deltaTr;
    idxY = size(ts.matrix, 2) - ts.deltaTr + 1: size(ts.matrix, 2);
end

[forecastY, trainForecastY, model] = feval(model.handle, ts.matrix(idxVal, idxX), model, ...
                            ts.matrix(idxTrain, idxX), ts.matrix(idxTrain, idxY)); 

% Remember that forecasts and ts.matrix are normalized
trainMAPE = calcSymMAPE(ts.matrix(idxTrain, idxY), trainForecastY);
testMAPE = calcSymMAPE(ts.matrix(idxVal, idxY), forecastY);
forecasts = zeros(size(ts.matrix, 1), ts.deltaTr);
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


         
        
        
    