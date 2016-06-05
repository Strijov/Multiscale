function [forecasted_y, train_forecast, model] = NnForecast(validationX, model, trainX, trainY)
% Compute forecast using NN model with fixed parameters.
%
% Input:
% validationX [mtest x nx] feature row for the forecasted points
% model [struct] containing NN model and its parameters:
%   model.obj stores the tuned network (network object), which expects [nx x M]
%   input and outputs [ny x M] matrix
% trainX, trainY store training data:
%   trainX [mtrain x nx] stores features
%   trainY [mtrain x ny] stores target variables
% 
%
% Output:
% forecast_y  [mtest x ny] forecasted values of y (regression of x)

HIDDEN_LAYER_SIZE = 10;

if model.unopt_flag
    model.params = struct('nHiddenLayers', HIDDEN_LAYER_SIZE);
    net = fitnet(model.params.nHiddenLayers);
    net.trainParam.showWindow = false;
    model.obj = train(net, trainX',trainY');
    model.unopt_flag = true; %FIXIT Should be false, but can't find function to retrain NN using old parameters.
end

forecasted_y = model.obj(validationX');
forecasted_y = forecasted_y';
train_forecast = model.obj(trainX');
train_forecast = train_forecast';

end