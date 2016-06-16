function test_forecast = MLSSVRPredict(testX, trainX, kernel_type, p1, p2, alpha, b, lambda)
% Prediction of MLSSVR model
%
% Input: 
% testX  	[m x N] feature columns of the test matrix
% trainX	[m x N] feature columns of the design matrix
% kernel_type   [str] can be {'linear', 'poly', 'rbf', 'erbf', 'sigmoid'}
% p1, p2    [1],[1] kernel parameters (see Kerfun.m)
% alpha     [m x deltaTr*n_predictions] model's parameters
% b         [1] bias term
% lambda    [1] trade off coefficient in cost function
%
% Output: 
% predictY  [m x deltaTr*n_predictions] predcitions
% TSE       [1] total squared error
% R2        [1] correlation coefficient

m = size(alpha, 2); 
l = size(trainX, 1); 

testN = size(testX, 1); 
b = b(:); 
    
K = KernelFunction(kernel_type, p1, p2, testX, trainX); 
test_forecast = repmat(sum(K*alpha, 2), 1, m) + K*alpha*(m/lambda) + repmat(b', testN, 1); 

end
