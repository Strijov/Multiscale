function [add_features, mdl] = ConvGenerator(X, mdl)
% Generates new features based on statistics of the current feature matrix.
%
% Input:
% workStructTS	see createRegMatrix.m for explanation
% 	workStructTS.matrix = [X Y]  contains the feature matrix X[m x n - deltaTr] 
%    horiontally concatenated with the target matrix Y[m x deltaTr]
%
% Output:
% [m x 5] matrix of the new features to add

    add_features = [sum(X, 2), mean(X, 2), min(X, [], 2), max(X, [], 2), std(X, 0, 2)]; 
    if ~mdl.replace
        add_features = [X, add_features];
        mdl.transform = @(X) [X, sum(X, 2), mean(X, 2), min(X, [], 2), max(X, [], 2), std(X, 0, 2)];
    else
        mdl.transform = @(X) [sum(X, 2), mean(X, 2), min(X, [], 2), max(X, [], 2), std(X, 0, 2)];
    end
    
    
    %{
    for i = [1:size(workStructTS.matrix,1)]
        x = workStructTS.matrix(i,:); % AM ? Only use first deltaTp of the matrix
        y = [sum(x), mean(x), min(x), max(x), std(x)]; % real(fft(x)), imag(fft(x))];
        add_features(i, :) = y;
    end
    %}
end