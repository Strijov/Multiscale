function [idxTrain, idxTest, idxVal] = TrainTestSplit(nSamples, trainTestValRatio)
% Splits nSamples into train, test & validation indices.
%
% Input:
% nSamples            number of rows in regression matrix  (X|Y)
% trainTestValRatio   float [1 x 3]  test M2 to train M1 size ratio
%
% Output:
% idxTrain      [1 x M1] indices of the train set
% idxTest       [1 x M2] indices of the test set
% idxVal        [1 x M3] indices of validation object, constant ...

trainTestValRatio = trainTestValRatio/sum(trainTestValRatio);
if numel(trainTestValRatio) == 1
    trainTestValRatio = [trainTestValRatio, 1-trainTestValRatio, 0];
elseif numel(trainTestValRatio) == 2
    trainTestValRatio = [trainTestValRatio, 0];
end

trainTestValRatio = floor(trainTestValRatio*nSamples);
val_size = trainTestValRatio(3) + nSamples - sum(trainTestValRatio);
test_size = trainTestValRatio(2);
idxVal = 1:val_size;
idxTest = val_size + 1:val_size + test_size;
idxTrain = val_size + test_size + 1:nSamples;
 
end