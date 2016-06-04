function [add_features, mdl] = NwGenerator(X, mdl)
% Performs smoothing of the local prehistory (current features).
%
% Input:
% workStructTS	see createRegMatrix.m for explanation
% 	workStructTS.matrix = [X Y]  contains the feature matrix X[m x n - deltaTr] 
%    horiontally concatenated with the target matrix Y[m x deltaTr]
%
% Output:
% [m x deltaTp] matrix of the new features to add

% smoothing is only applied to the target time series
add_features = transform(X);

%{
if ~mdl.replace
    add_features = [X, add_features];
    mdl.transform = @(X) [X, transform(X, n)];
else
%}
mdl.transform = @(X) transform(X);


% RN: Sourse TS is being replaced with the smoothed one, that's why dims
%don't change. Smoothed TS overwrites old one in X immediately. 
end

function X = transform(X)


for i = 1:size(X,1)
    x_smoothed = NWSmoothing(X(i, :));
    X(i,:) = x_smoothed;
end


end