function [alpha, b] = MLSSVRTrain(trainX, trainY, kernel_type, p1, p2, gamma, lambda)
% Train MLSSVR model
%
% Input: 
% trainX	[m x N] feature columns of the design matrix
% trainY	[m x deltaTr*n_predictions] target columns of the design matrix
% kernel_type   [str] can be {'linear', 'poly', 'rbf', 'erbf', 'sigmoid'}
% p1, p2    [1],[1] kernel parameters (see Kerfun.m)
% gamma     [1] trade off coefficient in cost function
% lambda    [1] trade off coefficient in cost function
%
% Output: 
% alpha     [m x deltaTr*n_predictions] model's parameters
% b         [1] bias term

if (size(trainX, 1) ~= size(trainY, 1)) 
    display('The number of rows in trnX and trnY must be equal.'); 
    return; 
end

[l, m] = size(trainY); 

K = Kerfun(kernel_type, trainX, trainX, p1, p2); 
H = repmat(K, m, m) + eye(m * l) / gamma; 

P = zeros(m*l, m); 
for t = 1: m
    idx1 = l * (t - 1) + 1; 
    idx2 = l * t; 
    
    H(idx1: idx2, idx1: idx2) = H(idx1: idx2, idx1: idx2) + K*(m/lambda); 
    
    P(idx1: idx2, t) = ones(l, 1); 
end

eta = H \ P; 
nu = H \ trainY(:); 
S = P'*eta; 
b = inv(S)*eta'*trainY(:); 
alpha = nu - eta*b; 

alpha = reshape(alpha, l, m); 
