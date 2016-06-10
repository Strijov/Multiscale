function [testFrc, trainFrc] = splitForecastsByTs(forecasts, idxTrain, idxTest, ...
                                               deltaTr)
                                           
testFrc = cell(1, numel(deltaTr)); 
trainFrc = cell(1, numel(deltaTr));
for i = 1:numel(deltaTr)
    [testFrc{i}, trainFrc{i}] = split_forecast(cell2mat(forecasts(:, i)), idxTrain, idxTest, deltaTr(i));
end

end


function [testFrc, trainFrc] = split_forecast(forecasts, idxTrain, idxTest, ...
                                               deltaTr)
                                           
nTrain = size(idxTrain, 2);
nTest = size(idxTest, 2);

testFrc = zeros(deltaTr, numel(idxTest));
trainFrc = zeros(deltaTr, numel(idxTrain));


for i = 1:size(idxTrain, 1)
    idxTrainMat = bsxfun(@plus, idxTrain(i, :), (0:deltaTr - 1)');
    idxTestMat = bsxfun(@plus, idxTest(i, :), (0:deltaTr - 1)');

    testFrc(:, (i-1)*nTest + 1:i*nTest) = forecasts(idxTestMat);
    trainFrc(:, (i-1)*nTrain + 1:i*nTrain) = forecasts(idxTrainMat);
end

end