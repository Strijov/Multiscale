function add_features = NwGenerator(workStructTS)
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
add_features = zeros(size(workStructTS.matrix,1), workStructTS.deltaTp);
for i = 1:size(workStructTS.matrix,1)
    x = workStructTS.matrix(i, 1:workStructTS.deltaTp);
    x_smoothed = NWSmoothing(x);
    add_features(i,:) = x_smoothed;
end


% RN: Sourse TS is being replaced with the smoothed one, that's why dims
%don't change. Smoothed TS overwrites old one in X immediately. 
end