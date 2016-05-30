function [testMAPE, trainMAPE, model] = ComputeForecastingErrors(ts, K, alpha_coeff, model)



model.forecasted_y = zeros(size(ts.matrix, 1) * ts.deltaTr, ts.deltaTr*K);

trainMAPE = zeros(1, K);
testMAPE = zeros(1, K);
for n = 1:K
    [idxTrain, ~, idxVal, idxX, idxY] = FullSplit(size(ts.matrix, 1), ...
                                size(ts.matrix, 2), alpha_coeff, ts.deltaTr);
    
    [forecastY, trainForecastY, model] = feval(model.handle, ts.matrix(idxVal, idxX), model, ...
                                ts.matrix(idxTrain, idxX), ts.matrix(idxTrain, idxY)); 
    
    trainMAPE(n) = calcSymMAPE(ts.matrix(idxTrain, idxY), trainForecastY);
    testMAPE(n) = calcSymMAPE(ts.matrix(idxVal, idxY), forecastY);
    forecasts = zeros(size(ts.matrix, 1), ts.deltaTr);
    forecasts(idxTrain, :) = trainForecastY;
    forecasts(idxVal, :) = forecastY;
    model.forecasted_y = unravel_target_var(forecasts);
end
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


         
        
        
    