function [test_forecast, train_forecast, model] = MarginalForecast(testX, model,trainX, trainY)
% Compute forecast using VAR model with fixed parameters.
%
% Input:
% validationX [mtest x nx] feature row for the forecasted period
% model [struct] containing model and its parameters;
%   model.params stores deltaTr and deltaTp vectors 
% trainX, trainY stores training data:
%   trainX [mtrain x nx] stores features
%   trainY [mtrain x ny] stores target variables
%
% Output:
% test_forecast  [mtest x ny] forecasted values of test y (regression of x)
% train_forecast  [mtrain x ny] forecasted values of train y (regression of x)
% model - updated model structure


model.transform = @(X) transform(X, model.params.deltaTp, model.params.deltaTr);
train_forecast = model.transform(trainX);
test_forecast = model.transform(testX);

end

function Y = transform(X, deltaTp, deltaTr)

Y = zeros(size(X, 1), sum(deltaTr));
sumDeltaTr = [0, cumsum(deltaTr)];
deltaTp = [cumsum(deltaTp)];

for i = 1:numel(deltaTr)
    Y(:, sumDeltaTr(i)+1:sumDeltaTr(i+1)) = ...
        X(:, deltaTp(i) - deltaTr(i) + 1:deltaTp(i));
end

end