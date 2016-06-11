function [add_features, mdl] = ConvGenerator(X, mdl, ~)
% Generates new features based on statistics of the current feature matrix.
%
% Input:
% X       contains the feature matrix X[m x n], where matrices X_i for 
%    various time series ([m x n_i]) are horiontally concatenated
% mdl     feature selection model. Structure with fileds: handle, params,
%         transform, replace. See GenerateFeatures.m
% deltaTp vector [n_1, ..., n_N] of X_i second dimensions, where N is the number of time series composing
%         the design matrix, n_i shoud sum up to n
%
% Output:
% [m x 5] matrix of the new features to add

    add_features = [sum(X, 2), mean(X, 2), min(X, [], 2), max(X, [], 2), std(X, 0, 2)]; 
    mdl.transform = @(X) [sum(X, 2), mean(X, 2), min(X, [], 2), max(X, [], 2), std(X, 0, 2)];
    
    
end