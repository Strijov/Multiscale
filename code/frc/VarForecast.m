function forecasted_y = VarForecast(validation_x, model,trainX, trainY)
% Compute forecast using VAR model with fixed parameters.
%
% Input:
% validation_x [1 x deltaTp] feature string for last period
% model [struct] containing model and its parameters;
%   model.params stores AR parameters matrix W [deltaTp x deltaTr] 
% trainX, trainY stores training data:
%   trainX [m x deltaTp] stores features
%   trainY [m x deltaTr] stores target variables
%
% Output:
% forecast_y  [1 x deltaTr] forecasted values of y (regression of x)

if model.unopt_flag
    model.params = inv(trainX'*trainX)*trainX'*trainY;
    model.tuned_func = [];
    model.unopt_flag = false; % AM 
end

W = model.params;
forecasted_y = validation_x*W;

end