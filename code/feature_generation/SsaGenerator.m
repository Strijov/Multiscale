function add_features = SsaGenerator(workStructTS)
% Performs SSA of of each feature row and returns its eigenvaluesas new features.
%
% Input:
% workStructTS	see createRegMatrix.m for explanation
% 	workStructTS.matrix = [X Y]  contains the feature matrix X[m x n - deltaTr] 
%    horiontally concatenated with the target matrix Y[m x deltaTr]
%
% Output:
% [m x N_COMP] matrix of the new features to add

N_COMP = 3;
add_features = zeros(size(workStructTS.matrix, 1), N_COMP);
for i = [1:size(workStructTS.matrix,1)]
    x = workStructTS.matrix(i, 1:end - workStructTS.deltaTr);
    caterpillar_length = floor(numel(x) / 2);
    eigenvalues = principalComponentAnalysis(x, caterpillar_length, 0, 0);
    add_features(i, :) = eigenvalues(1:N_COMP);
end
 
end