function [MAPE_test, MAPE_train, AIC, model] = calcErrorsByModel(StructTS, model, idxTrain, idxTest)

% This funtions returns forecasting errors on the test and train sets for
% each models and the models with updated fields.

% Input:
%   StructTS                      structure of time-series
%   model         [1 x nModels]   list of models (each is a struct)
%   idxTrain      [1 x M1]        indices of the train set
%   idxTest       [1 x M2]        indices of the test set
% Output:
%   MAPE_test     float           MAPE on test set
%   MAPE_train    float           MAPE on train set
%   AIC           float           AIC on train(test?) set
%   model         [1 x nModels]   list of tuned models (if model has
%       parametest to be tuned
nModels = numel(model);

MAPE_test = zeros(nModels,1);
MAPE_train = zeros(nModels,1); 
AIC = zeros(nModels,1);

for i = 1:nModels
    disp(['Fitting model: ', model(i).name])
    [MAPE_test(i), MAPE_train(i), model(i)] = computeForecastingErrors(...
                                            StructTS, model(i), 0, ...
                                            idxTrain, idxTest);
    AIC(i) = 2*StructTS.deltaTp + size(StructTS.X, 1) * ...
                        log(nan_norm(StructTS.x(StructTS.deltaTp + 1: ...
                        StructTS.deltaTp + numel(StructTS.Y)) - ...
                        model(i).forecasted_y));
end


end