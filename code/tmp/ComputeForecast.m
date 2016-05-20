function [forecast_y] = ComputeForecast(x, model, trainX, trainY)
% Compute forecast using model with fixed parameters.

% Input:
% x [1 x deltaTp] feature string for last period
% model [struct] containing model and its parameters
% trainX, trainY  training data:
% trainX [m x deltaTp] stores features
% trainY [m x deltaTr] stores target variables
%
% Output:
% forecast_y  [1 x deltaTr] forecasted values of y (regression of x)

switch model.name        
case 'VAR'
    W = model.params;
    forecast_y = x*W;
case 'Neural_network'
    forecast_y = model.tuned_func(x');
    forecast_y = forecast_y';
case 'SVR'
    forecast_y = feval(model.handle, trainX, trainY, x, model.params); %SVRMethod(X, Y, val_x);
end

end
% TODO reply this code by eval with model struct.