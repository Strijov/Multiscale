function [ts, feature_selection_mdl] = FeatureSelection(ts, feature_selection_mdl)

% FIXIT: do not forget to split this into train and test!!!
feature_selection_mdl.params.minComps = ts.deltaTr;
[X, feature_selection_mdl] = feval(feature_selection_mdl.handle, ts.matrix, feature_selection_mdl);
Y = ts.matrix(:, end - ts.deltaTr + 1:end);
ts.matrix = [X, Y];

end