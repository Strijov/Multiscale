function [idxTrain, idxTest, idxVal] = TrainTestSplit(nSamples, trainTestValRatio)
% Splits nSamples into train, test & validation indices.
%
% Input:
% nSamples            number of rows in regression matrix  (X|Y)
% trainTestValRatio   float [1 x 2]  test M2 to train M1 size ratio
%
% Output:
% idxTrain      [1 x M1] indices of the train set
% idxTest       [1 x M2] indices of the test set
% idxVal        [1 x 1] constant ...

if numel(trainTestValRatio) == 1 && trainTestValRatio <= 1
    trainTestValRatio = [trainTestValRatio, 1-trainTestValRatio];
    trainTestValRatio = floor(trainTestValRatio*(nSamples - 1));
elseif numel(trainTestValRatio) == 2
    trainTestValRatio = trainTestValRatio/sum(trainTestValRatio); 
    trainTestValRatio = floor(trainTestValRatio*(nSamples - 1));
else
    trainTestValRatio = [trainTestValRatio, nSamples - trainTestValRatio - 1];
end


test_size = trainTestValRatio(2);
idxVal = 1;
idxTest = 2:1 + test_size;
idxTrain = 2 + test_size:nSamples;
 
end