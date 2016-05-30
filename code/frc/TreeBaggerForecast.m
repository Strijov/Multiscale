function [forecasted_y, train_forecast, model] = TreeBaggerForecast(validation_x, model, trainX, trainY)
% Compute forecasts using 'Random Forest' model with fixed parameters.
%
% Input:
% validation_x [1 x deltaTp] feature string for last period
% model [struct] containing model and its parameters:
%   model.params stores the tuned network (network object), which expects [deltaTp x M]
%   input and outputs [deltaTr x M] matrix
% trainX, trainY store training data:
%   trainX [m x deltaTp] stores features
%   trainY [m x deltaTr] stores target variables
% 
%
% Output:
% forecast_y  [1 x deltaTr] forecasted values of y (regression of x)

ALPHA = 0.0005;
N_PREDICTORS = 20;
N_TREES = 50;

if model.unopt_flag
    model.params = struct('nTrees', N_TREES, 'nVars', N_PREDICTORS);
    tb = TreeBagger(model.params.nTrees, trainX, trainY(:, 1),...
                                  'Method', 'regression', ...
                                  'oobpred', 'on', ...
                                  'NVarToSample', model.params.nVars);
    oobFrcError = oobError(tb);
    [~, nTrees] = min(oobFrcError + ALPHA*(1:length(oobFrcError))');
    %plot([oobFrcError, oobFrcError + ALPHA*(1:length(oobFrcError))']);
    
    model.unopt_flag = false;
    model.obj = tb;
    model.params.nTrees = nTrees;
end

forecasted_y = zeros(size(validation_x, 1), size(trainY, 2));
train_forecast = zeros(size(trainY));
for j = 1:size(trainY, 2)
    tb = TreeBagger(nTrees, trainX, trainY(:, j),...
                                  'Method', 'regression', ...
                                  'NVarToSample', model.params.nVars);
    forecasted_y(:, j) = model.obj.predict(validation_x);
    train_forecast(:, j) = model.obj.predict(trainX);
                          
end


end