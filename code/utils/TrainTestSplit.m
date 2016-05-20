function [matTrain, matTest, natVal] = TrainTestSplit(mat, alpha_coeff)
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

natVal = mat(1,:);
mat(1,:) = []; % FIXIT Please remove this line and change addresses.
test_size = floor(alpha_coeff * (size(mat, 1) - 1));
matTest = mat(1:test_size,:);
matTrain = mat(test_size+1:end, :);
end