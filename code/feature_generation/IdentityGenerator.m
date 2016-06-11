function [X, mdl] = IdentityGenerator(X, mdl, ~)
% Returns the features matrix as it is. To be used when no feature generation is required.
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
% X [m x deltaTp] matrix of the current features

mdl.transform = @(X) X;
    
end