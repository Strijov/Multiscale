function [ts, feature_selection_mdl] = FeatureSelection(ts, feature_selection_mdl, ...
                                                        idxTrain, idxTest)
% Felects a subset of features using methods, specified with feature_selection_mdl.
%
% Input:
% ts - main time series structure, see createRegMatrix.m for explanation
% feature_selection_mdl  - a structure, which describes feature selection
% model. Main fields: 
%   feature_selection_mdl.handle - handle to the main method,
%   feature_selection_mdl.params - parameters of the model, 
%   feature_selection_mdl.transform - handle to feature selection on the
%   test set.
%   procedure for the test set
% idxTrain - indices of train set, optional 
% idxTest  - indices of test set, optional. If idxTrain and idxTest are
%   specified, trainable feature slection paramaters are learnt on train set 
%   to be used in feature_selection_mdl.transform.  
%
% Output:
% ts with new feature matrix
if nargin < 3
    idxTrain = 1:size(ts.X, 1);
    idxTest = [];
end

feature_selection_mdl.params.minComps = ts.deltaTr;
[Xtrain, feature_selection_mdl] = feval(feature_selection_mdl.handle, ...
                                   ts.X(idxTrain, :), ...
                                   feature_selection_mdl);
Xtest = ts.X(idxTest, :);
X = zeros(size(ts.X, 1), size(Xtrain, 2));
if ~isempty(idxTest)
    X(idxTest, :) = feature_selection_mdl.transform(Xtest);
end

X(idxTrain, :) = Xtrain;
ts.X = X;

end