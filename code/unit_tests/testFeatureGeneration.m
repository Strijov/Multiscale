function tests = testFeatureGeneration

% Test forecasting methods includes the following testing scenarios:
% - testNWreplace: checks that replace=false in NwGnerator leads to a warning

tests  = functiontests(localfunctions);

end

function testNWreplace(testCase)

% checks that replace=false in NwGnerator leads to a warning
ts = createRandomDataStruct(3, 500); % make it relatively small
ts = CreateRegMatrix(ts);

generators = struct('handle', @NwGenerator, 'name', 'NW', ...
                     'replace', false, 'transform', []);

warningFunc = @() GenerateFeatures(ts, generators);
verifyWarning(testCase, warningFunc, 'genFeatsReplace:id');

end