function [add_features, mdl] = SsaGenerator(ts, mdl)
% Performs SSA of of each feature row and returns its eigenvaluesas new features.
%
% Input:
% ts - see createRegMatrix.m for explanation
% 	ts.matrix = [X Y]  contains the feature matrix X[m x n - deltaTr] 
%    horiontally concatenated with the target matrix Y[m x deltaTr]
%
% Output:
% [m x N_COMP] matrix of the new features to add

N_COMP = 3;
add_features = transform(ts.matrix(:, 1:end - ts.deltaTr), N_COMP); 
mdl.transform = @(x) transform(x, N_COMP);

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