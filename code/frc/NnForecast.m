function forecasted_y = NnForecast(validation_x, model, trainX, trainY)
% Compute forecast using VAR model with fixed parameters.
%
% Input:
% validation_x [1 x deltaTp] feature string for last period
% model [struct] containing NN model and its parameters:
%   model.params stores the tuned network (network object), which expects [deltaTp x M]
%   input and outputs [deltaTr x M] matrix
% trainX, trainY store training data:
%   trainX [m x deltaTp] stores features
%   trainY [m x deltaTr] stores target variables
% 
%
% Output:
% forecast_y  [1 x deltaTr] forecasted values of y (regression of x)

HIDDEN_LAYER_SIZE = 10;

if model.unopt_flag
    net = fitnet(HIDDEN_LAYER_SIZE);
    net.trainParam.showWindow = false;
    model.params = train(net, trainX',trainY');
    model.unopt_flag = true; %FIXIT Should be false, but can't find function to retrain NN using old parameters.
end

forecasted_y = model.params(validation_x');
forecasted_y = forecasted_y';

end