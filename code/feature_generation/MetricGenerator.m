function [add_features, mdl] = MetricGenerator(X, mdl)
% Generates new features based on the distances to the centroids
%
% Input:
% workStructTS	see createRegMatrix.m for explanation
% 	workStructTS.matrix = [X Y]  contains the feature matrix X[m x deltaTp] 
%    horiontally concatenated with the target matrix Y[m x deltaTr]
%
% Output:
% [m x n] matrix of the new features to add

n = 3; %number of centroids
    
add_features = transform(X, n);
if ~mdl.replace
    add_features = [X, add_features];
    mdl.transform = @(X) [X, transform(X, n)];
else
    mdl.transform = @(X) transform(X, n);
end
 

end


function res = transform(X, n)
if size(X, 1) < n
    n = size(X, 1);
end

if size(X, 1) > 0
    [~, centroids] = kmeans(X, n);
end
res = zeros(size(X,1), n);
for i = 1:size(X,1)
    y = X(i, :);
    p = euclidDistance(repmat(y, size(centroids, 1), 1), centroids);
    res(i, :) = p;
end

end