function [idxTrain, idxTest, idxVal] = TrainTestSplit(nSamples, trainTestRatio)
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

if numel(trainTestRatio) == 1 && trainTestRatio < 1
    trainTestRatio = [trainTestRatio, 1-trainTestRatio];
    trainTestRatio = floor(trainTestRatio*(nSamples - 1));
elseif numel(trainTestRatio) == 2
    positiveFlag = all(trainTestRatio > 0);
    trainTestRatio = trainTestRatio/sum(trainTestRatio); 
    trainTestRatio = floor(trainTestRatio*(nSamples - 1));
    if positiveFlag && sum(trainTestRatio) == 1
       warning('emptyTrainTest:id', 'idxTrain or idxTest is empty, though trainTestRatio is positive'); 
    elseif positiveFlag && trainTestRatio(1) == 0
        trainTestRatio = trainTestRatio + [1, -1];
    elseif positiveFlag && trainTestRatio(2) == 0
        trainTestRatio = trainTestRatio + [-1, 1];
    end
else
    trainTestRatio = [trainTestRatio, nSamples - trainTestRatio - 1];
end


train_size = trainTestRatio(1);
idxVal = 1;
idxTest = 2:nSamples - train_size;
idxTrain = nSamples - train_size + 1:nSamples;
 
end