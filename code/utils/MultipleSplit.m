function [idxTrain, idxTest, idxVal] = MultipleSplit(nRowsTotal, nRows, trainTestRatio)
% Split design matrix rows into subsamples of size nRows

idxSplits = sequentialSplit(nRowsTotal, nRows);
[idxTrain, idxTest, idxVal] = TrainTestSplit(nRows, trainTestRatio);

idxTest = idxSplits(:, idxTest);
idxTrain = idxSplits(:, idxTrain);
idxVal = idxSplits(:, idxVal);

end
