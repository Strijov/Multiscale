function demoForecastAnalysis(StructTS, model)

TRAIN_TEST_RATIO = 0.75;
SUBSAMPLE_SIZE = 50;

StructTS = CreateRegMatrix(StructTS);
% Split design matrix rows into subsamples of size SUBSAMPLE_SIZE
[idxTest, idxTrain] = MultipleSplit(size(StructTS.X, 1), SUBSAMPLE_SIZE, TRAIN_TEST_RATIO);
nSplits = size(idxTrain, 1);
nTrain = size(idxTrain, 2);
nTest = size(idxTest, 2);

testRes = zeros(size(StructTS.Y, 2), numel(idxTest));
trainRes = zeros(size(StructTS.Y, 2), numel(idxTrain));

% Calc frc residuals by split: 
for i = 1:nSplits
    ts = StructTS;
    idxSplit = [idxTest(i, :), idxTrain(i,:)];
    %ts.X = StructTS.X(idxSplit, :);
    %ts.Y = StructTS.Y(idxSplit, :);
    [~, ~, model] = computeForecastingErrors(ts, model, 0, idxTrain(i,:), idxTest(i,:));
    residuals = ts.x(ts.deltaTp + 1:ts.deltaTp + numel(ts.Y)) - model.forecasted_y;
    [testRes(:, (i-1)*nTest + 1:i*nTest), ...
     trainRes(:, (i-1)*nTrain + 1:i*nTrain)] = split_forecast(residuals, ...
                                idxTrain(i, :), idxTest(i, :), ts.deltaTr);
    
end

% Plot evolution of res mean and std by split
plot_residuals_stats(testRes', trainRes');


testRes = testRes(:);
trainRes = trainRes(:);

% Fit residuals to normal distribution:
testPD = fitdist(testRes, 'Normal');
trainPD = fitdist(trainRes, 'Normal');

% Plot normal pdf and QQ-plots for train and test residuals 
plot_residuals_npdf(testRes, trainRes, testPD, trainPD);



end



function [testFrc, trainFrc] = split_forecast(forecasts, idxTrain, idxTest, deltaTr)

idxTrain = repmat(idxTrain, deltaTr, 1) + repmat((0:deltaTr-1)', 1, numel(idxTrain));
%idxTrain = idxTrain(:);
idxTest = repmat(idxTest, deltaTr, 1) + repmat((0:deltaTr-1)', 1, numel(idxTest));
%idxTest = idxTest(:);


testFrc = forecasts(idxTest);
trainFrc = forecasts(idxTrain);

end


