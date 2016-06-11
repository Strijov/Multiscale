function [add_features, mdl] = CubicGenerator(X, mdl, ~)
% Generates new features based on the coefficients of the polynomial (cubic) 
% model fitted to the current feature matrix.
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
% [m x 4] matrix of the new features to add

n = 3;
add_features = transform(X, n);
mdl.transform = @(X) transform(X, n);
 

end

function res = transform(X, n)

res = zeros(size(X,1), n + 1);
idx = 1:size(X, 2);
for i = 1:size(X,1)
    y = X(i, :);
    p = polyfit(idx, y, n);
    res(i, :) = p;
end

end