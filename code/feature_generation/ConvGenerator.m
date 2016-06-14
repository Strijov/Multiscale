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

DEPTH = 5;
add_features = [mean(X, 2), min(X, [], 2), max(X, [], 2), std(X, 0, 2), haarTransform(X, DEPTH)]; 
mdl.transform = @(X) [mean(X, 2), min(X, [], 2), max(X, [], 2), std(X, 0, 2), haarTransform(X, DEPTH)];
    
    
end

function res = haarTransform(X, depth)

logSizeX = floor(log2(size(X, 2)));
depth = min([depth, logSizeX]);
zeroPadX = zeros(size(X, 1), 2^(logSizeX + 1));
zeroPadX(:, 1:size(X, 2)) = X;
depth = 2.^(logSizeX-1:-1:logSizeX-depth);
idx = [0,cumsum(depth)];
resAv = zeros(size(X, 1), sum(depth));
resDif = zeros(size(X, 1), sum(depth));

avX = zeroPadX;
difX = zeroPadX;
for i = 1:numel(depth)
    avX = (avX(:, 1:2:2*depth(i) - 1) + avX(:, 2:2:2*depth(i)))/2;
    difX = (difX(:, 1:2:2*depth(i) - 1) - difX(:, 2:2:2*depth(i)))/2;
    resAv(:, idx(i)+1:idx(i+1)) = avX;
    resDif(:, idx(i)+1:idx(i+1)) = difX; 
end

res = [resAv, resDif];
end