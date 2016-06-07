function [idxTest, idxTrain] = MultipleSplit(nRowsTotal, nRows, trainTestRatio);
% Split design matrix rows into subsamples of size nRows

idxSplits = sequentialSplit(nRowsTotal, nRows);
[idxTrain, idxTest, idxVal] = TrainTestSplit(nRows, 1- trainTestRatio);
idxTest = [idxVal, idxTest];

idxTest = idxSplits(:, idxTest);
idxTrain = idxSplits(:, idxTrain);

end
