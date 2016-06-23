% this script runs all unit tests
addpath(genpath(cd));

% test matrix creation:
runtests('testRegressionMatrix.m');

% tests for the foresting modules:
runtests('testForecasts.m');

% tests for feature generation block:
runtests('testFeatureGeneration.m');

% tests for feature selection block:
runtests('testFeatureSelection');

% various tests:
runtests('testMiscellaneous.m');