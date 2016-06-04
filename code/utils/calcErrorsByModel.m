function [MAPE_test, MAPE_train, AIC, model] = calcErrorsByModel(StructTS, model, idxTrain, idxTest)

nModels = numel(model);

MAPE_test = zeros(nModels,1);
MAPE_train = zeros(nModels,1); 
AIC = zeros(nModels,1);

for i = 1:nModels
    disp(['Fitting model: ', model(i).name])
    [MAPE_test(i), MAPE_train(i), model(i)] = ComputeForecastingErrors(...
                                            StructTS, 0, model(i), ...
                                            idxTrain, idxTest);
    AIC(i) = 2*StructTS.deltaTp + size(StructTS.matrix, 1) * ...
                        log(nan_norm(StructTS.x(StructTS.deltaTp + 1:end) - ...
                        model(i).forecasted_y'));
end


end