function [add_features, mdl] = NwGenerator(X, mdl, ~)
% Performs smoothing of the local prehistory (current features).
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
% [m x n] matrix of the new features to add

if isempty(mdl.transform) 
    [add_features, bandwidth] = fit_transform(X);
    mdl.transform = @(X) transform(X, bandwidth);
else
    add_features = feval(mdl.transform, X); 
end



% RN: Sourse TS is being replaced with the smoothed one, that's why dims
%don't change. Smoothed TS overwrites old one in X immediately. 
end

function [X, h] = fit_transform(X)

h = zeros(1, size(X, 1));
for i = 1:size(X,1)
    [x_smoothed, h(i)] = NWSmoothing(X(i, :));
    X(i,:) = x_smoothed;
end

h = median(h);

end

function X = transform(X, h)

for i = 1:size(X,1)
    x_smoothed = NWSmoothing(X(i, :), h);
    X(i,:) = x_smoothed;
end

end