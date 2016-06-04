function [add_features, mdl] = CubicGenerator(X, mdl)
% Generates new features based on the coefficients of the polynomial (cubic) 
% model fitted to the current feature matrix.
%
% Input:
% workStructTS	see createRegMatrix.m for explanation
% 	workStructTS.matrix = [X Y]  contains the feature matrix X[m x deltaTp] 
%    horiontally concatenated with the target matrix Y[m x deltaTr]
%
% Output:
% [m x 4] matrix of the new features to add

n = 3;
add_features = transform(X, n);
if ~mdl.replace
    add_features = [X, add_features];
    mdl.transform = @(X) [X, transform(X, n)];
else
    mdl.transform = @(X) transform(X, n);
end
 

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