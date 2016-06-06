function [test_forecast, train_forecast, model] = TreeBaggerForecast(validationX, model, trainX, trainY)
% Compute forecasts using 'Random Forest' model with fixed parameters.
%
% Input:
% validationX [mtest x nx] feature row for the forecasted points
% model [struct] containing model handle and parameters:
%   model.obj stores the trained model
% trainX, trainY store training data:
%   trainX [mtrain x nx] stores features
%   trainY [mtrain x ny] stores target variables
% 
%
% Output:
% test_forecast  [mtest x ny] forecasted values of test y (regression of x)
% train_forecast  [mtrain x ny] forecasted values of train y (regression of x)
% model - ipdated model structure

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
    model.params.nTrees = nTrees;
    
    test_forecast = zeros(size(validationX, 1), size(trainY, 2));
    train_forecast = zeros(size(trainY));
    tb = zeros(1, size(trainY, 2));

    for j = 1:size(trainY, 2)
    tb(j) = TreeBagger(nTrees, trainX, trainY(:, j),...
                                  'Method', 'regression', ...
                                  'NVarToSample', model.params.nVars);
    test_forecast(:, j) = tb(j).predict(validationX);
    train_forecast(:, j) = tb(j).predict(trainX);                          
    end
    model.transform = @(x) transform(x, tb);
else
    train_forecast = fefal(model.transform, trainX);
    test_forecast = fefal(model.transform, validationX);
end


end


function frc = transform(X, tb)

frc = zeros(size(X, 1), numel(tb));
for j = 1:numel(tb)
    frc(:, j) = tb.predict(X);
end
    

end