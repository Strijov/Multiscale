function [forecast_y] = ComputeForecast(val_x, model, X, Y)
% Compute forecast using model with fixed parameters.

% Input:
% x [1 x deltaTp] feature string for last period
% model [struct] containing model and its parameters
%
% Output:
% forecast_y  [1 x deltaTr] forecast string for las period

switch model.name        
case 'VAR'
    W = model.params;
    forecast_y = val_x*W;
case 'Neural_network'
    forecast_y = model.tuned_func(val_x');
    forecast_y = forecast_y';
case 'SVR'
    forecast_y = SVRMethod(X, Y, val_x);
end

end
% TODO reply this code by eval with model struct.