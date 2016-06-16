function [idxTrain, idxTest, idxVal] = MultipleSplit(nRowsTotal, nRows, trainTestValRatio);
% Split design matrix rows into subsamples of size nRows

idxSplits = sequentialSplit(nRowsTotal, nRows);
[idxTrain, idxTest, idxVal] = TrainTestSplit(nRows, trainTestValRatio);

idxTest = idxSplits(:, idxTest);
idxTrain = idxSplits(:, idxTrain);
idxVal = idxSplits(:, idxVal);

end
