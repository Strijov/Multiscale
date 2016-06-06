function [test_forecast, train_forecast, model] = VarForecast(validationX, model,trainX, trainY)
% Compute forecast using VAR model with fixed parameters.
%
% Input:
% validationX [mtest x nx] feature row for the forecasted period
% model [struct] containing model and its parameters;
%   model.params stores AR parameters matrix W [nx x ny] 
% trainX, trainY stores training data:
%   trainX [mtrain x nx] stores features
%   trainY [mtrain x ny] stores target variables
%
% Output:
% test_forecast  [mtest x ny] forecasted values of test y (regression of x)
% train_forecast  [mtrain x ny] forecasted values of train y (regression of x)
% model - ipdated model structure

if model.unopt_flag
    model.params = inv(trainX'*trainX)*trainX'*trainY;
    model.unopt_flag = false; % AM 
end

W = model.params;
test_forecast = validationX*W;
train_forecast = trainX*W;

model.transform = @(x) x*W;

end