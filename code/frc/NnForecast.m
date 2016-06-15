function [test_forecast, train_forecast, model] = NnForecast(validationX, model, trainX, trainY)
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
% test_forecast  [mtest x ny] forecasted values of test y (regression of x)
% train_forecast  [mtrain x ny] forecasted values of train y (regression of x)
% model - ipdated model structure

HIDDEN_LAYER_SIZE = 10;

if model.unopt_flag
    model.params = struct('nHiddenLayers', HIDDEN_LAYER_SIZE);
    model.unopt_flag = false; %FIXIT Should be false, but can't find function to retrain NN using old parameters.
end

net = fitnet(model.params.nHiddenLayers);
net.trainParam.showWindow = false;
trained_net = train(net, trainX',trainY');
model.transform = @(x) transform(x, trained_net);
test_forecast = feval(model.transform, validationX);
train_forecast = feval(model.transform, trainX);

end

function frc = transform(X, trained_net)

frc = trained_net(X');
frc = frc';

end