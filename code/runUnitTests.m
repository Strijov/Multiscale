% this scrip runs all unit tests

addpath(genpath(cd));

% test matrix creation:
runtests('testRegressionMatrix.m');

% tests for the foresting modules:
runtests('testForecasts.m');

% test for feature generation block:
runtests('testFeatureGeneration.m');

% various tests:
runtests('testMiscellaneous.m');