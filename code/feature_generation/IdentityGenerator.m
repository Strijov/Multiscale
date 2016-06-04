function [X, mdl] = IdentityGenerator(X, mdl)
% Returns the features matrix as it is. To be used when no feature
% generation is required.
%
% Input:
% workStructTS	see createRegMatrix.m for explanation
% 	workStructTS.matrix = [X Y]  contains the feature matrix X[m x deltaTp] 
%    horiontally concatenated with the target matrix Y[m x deltaTr]
%
% Output:
% X [m x deltaTp] matrix of the current features

mdl.transform = @(X) X;
    
end