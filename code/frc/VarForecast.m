function [forecasted_y, train_forecast, model] = VarForecast(validation_x, model,trainX, trainY)
% Compute forecast using VAR model with fixed parameters.
%
% Input:
% validation_x [1 x nx] feature row for the forecasted period
% model [struct] containing model and its parameters;
%   model.params stores AR parameters matrix W [nx x ny] 
% trainX, trainY stores training data:
%   trainX [m x nx] stores features
%   trainY [m x ny] stores target variables
%
% Output:
% forecast_y  [1 x ny] forecasted values of y (regression of x)

if model.unopt_flag
    model.params = inv(trainX'*trainX)*trainX'*trainY;
    model.unopt_flag = false; % AM 
end

W = model.params;
forecasted_y = validation_x*W;
train_forecast = trainX*W;

end