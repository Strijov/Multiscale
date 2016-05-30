function [idxTrain, idxTest, idxVal] = TrainTestSplit(nSamples, alpha_coeff)
%Splits matrix to train, test & validation.
%
% Input:
% mat	        [M x N] regression matrix  (X|Y)
% alpha_coeff   [float] test size M2 divided to train size M1
%
% Output:
% matTrain      [M1 x N]
% matTest       [M2 x N]
% matVal        [1 x N]

train_size = floor((1 - alpha_coeff) * (nSamples - 1));
idxTrain = 2:train_size + 1;
idxTest = train_size + 2:nSamples;
idxVal = 1;
%mat(1,:) = []; % FIXIT Please remove this line and change addresses.
%matTest = mat(1:test_size,:);
%matTrain = mat(test_size+1:end, :);
end