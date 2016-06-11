function [add_features, mdl] = SsaGenerator(X, mdl, ~)
% Performs SSA of of each feature row and returns its eigenvalues as new features.
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
% [m x N_COMP] matrix of the new features to add

N_COMP = 3;
add_features = transform(X, N_COMP); 
mdl.transform = @(X) transform(X, N_COMP);

end

function res = transform(X, n)

res = zeros(size(X, 1), n);
caterpillar_length = floor(size(X, 2) / 2);
for i = 1:size(X, 1)
    x = X(i, :);
    eigenvalues = principalComponentAnalysis(x, caterpillar_length, 0, 0);
    res(i, :) = eigenvalues(1:n);
end


end