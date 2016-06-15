function [testMeanRes, trainMeanRes, testStdRes, trainStdRes, model] = ...
                            calcErrorsByModel(ts, model, idxTrain, idxTest)

% This funtions returns forecasting errors on the test and train sets for
% each models and the models with updated fields.

% Input:
%   ts                            structure of time-series
%   model         [1 x nModels]   list of models (each is a struct)
%   idxTrain      [1 x M1]        indices of the train set
%   idxTest       [1 x M2]        indices of the test set
% Output:
%   testMeanRes   float           mean residuals on test set
%   trainMeanRes  float           mean residuals on train set
%   testStdRes    float           std residuals on test set
%   trainStdRes   float           std residuals on train set
%   model         [1 x nModels]   list of tuned models (if model has
%                                 parameters to be tuned)
nModels = numel(model);
testMeanRes = zeros(nModels, numel(ts.x));
trainMeanRes = zeros(nModels, numel(ts.x)); 
testStdRes = zeros(nModels, numel(ts.x));
trainStdRes = zeros(nModels, numel(ts.x)); 


for i = 1:nModels
    disp(['Fitting model: ', model(i).name])
    [testRes, trainRes, model(i)] = computeForecastingResiduals(...
                                            ts, model(i), ...
                                            idxTrain, idxTest);
    
     testMeanRes(i,  :) = cellfun(@(x) nanmean(x(:)), testRes); 
     trainMeanRes(i, :) = cellfun(@(x) nanmean(x(:)), trainRes); 
     testStdRes(i,  :) = cellfun(@(x) nanstd(x(:)), testRes); 
     trainStdRes(i, :) = cellfun(@(x) nanstd(x(:)), trainRes); 
end


end