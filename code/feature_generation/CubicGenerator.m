function add_features = CubicGenerator(workStructTS)
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
add_features = zeros(size(workStructTS.matrix,1), n + 1);
x = [1:size(workStructTS.matrix,2)];
for i = [1:size(workStructTS.matrix,1)]
    y = workStructTS.matrix(i,:);
    p = polyfit(x,y,n);
    add_features(i, :) = p;
end
    
end