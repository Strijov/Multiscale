function [idxTrain, idxTest, idxVal] = TrainTestSplit(nSamples, alpha_coeff)
% Splits matrix to train, test & validation.
%
% Input:
% nSamples	    number of rows in regression matrix  (X|Y)
% alpha_coeff   [float] test M2 to train M1 size ratio
%
% Output:
% idxTrain      [1 x M1] indices of the train set
% idxTest       [1 x M2] indices of the test set
% idxVal        1 - index of validation object, constant ...

test_size = floor(alpha_coeff*(nSamples - 1));
idxTest = 2:test_size + 1;
idxTrain = test_size + 2:nSamples;
idxVal = 1;
 
end