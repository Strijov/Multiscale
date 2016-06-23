function tests = testFeatureSelection

tests  = functiontests(localfunctions);

end


function testSingularInput(testCase)

% checks PCA behaviour when the input matrix is singular

import matlab.unittest.constraints.IsAnything
import matlab.unittest.fixtures.SuppressedWarningsFixture
testCase.applyFixture(SuppressedWarningsFixture('stats:pca:ColRankDefX'));


ts = createRandomDataStruct(3, 500); % make it relatively small
ts = CreateRegMatrix(ts);

% feature selection:
pars = struct('maxComps', 50, 'expVar', 90, 'plot', @plot_pca_results);
feature_selection_mdl = struct('handle', @DimReducePCA, 'params', pars);


% set X to arbitrary degenerate matrix:
% columns are the same:
ts.X = repmat(ts.X(:, 1), 1, 100);
% check that feature selections returns anything: 
verifyThat(testCase, FeatureSelection(ts, feature_selection_mdl), IsAnything);

% rows are the same:
ts.X = repmat(ts.X(1, :), size(ts.X, 1), 1);
verifyThat(testCase, FeatureSelection(ts, feature_selection_mdl), IsAnything);

% all inputs are the same:
ts.X = ones(size(ts.X));
verifyThat(testCase, FeatureSelection(ts, feature_selection_mdl), IsAnything);


end

function testInputNans(testCase)

% checks that Nans in input matrix do not brake anything

import matlab.unittest.constraints.IsAnything
import matlab.unittest.fixtures.SuppressedWarningsFixture
testCase.applyFixture(SuppressedWarningsFixture('stats:pca:ColRankDefX'));

% random data:
ts = createRandomDataStruct(3, 500); % make it relatively small
ts = CreateRegMatrix(ts);

% feature selection:
pars = struct('maxComps', 50, 'expVar', 90, 'plot', @plot_pca_results);
feature_selection_mdl = struct('handle', @DimReducePCA, 'params', pars);


% add some Nans:
ts.X = imputeNans(ts.X, 0.1);

% check that feature selections returns anything: 
verifyThat(testCase, FeatureSelection(ts, feature_selection_mdl), IsAnything);


end