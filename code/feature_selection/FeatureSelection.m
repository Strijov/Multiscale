function [ts, feature_selection_mdl] = FeatureSelection(ts, feature_selection_mdl, ...
                                                        idxTrain, idxTest)

if nargin < 3
    idxTrain = 1:size(ts.matrix, 1);
    idxTest = [];
end
feature_selection_mdl.params.minComps = ts.deltaTr;
[Xtrain, feature_selection_mdl] = feval(feature_selection_mdl.handle, ...
                                   ts.X(idxTrain, :), ...
                                   feature_selection_mdl);
Xtest = ts.X(idxTest, :);
X = zeros(size(Y, 1), size(Xtrain, 2));
if ~isempty(idxTest)
    X(idxTest, :) = feature_selection_mdl.transform(Xtest);
end

X(idxTrain, :) = Xtrain;
ts.X = X;

end