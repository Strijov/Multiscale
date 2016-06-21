function tests = testMiscellaneous

% Test forecasting methods includes the following testing scenarios:
% - testTrainTestSplits: checks behaviour of trainTestSplit.m 


tests  = functiontests(localfunctions);

end


function testTrainTestSplits(testCase)

% checks that splitters correctly respond to various inputs

for i = 1:50
ts = createRandomDataStruct(); 
ts = CreateRegMatrix(ts);

nRowsTotal = size(ts.X, 1);
nRows = max(round(size(ts.X, 1)/5), 2);

trainTestSizes = floor([0.75, 0.25]*(nRows - 1));
if sum(trainTestSizes) == 1
    % this is a warning sutuation, since train and test are expected
    % nonempty
    warningFun = @() MultipleSplit(nRowsTotal, nRows, [0.75, 0.25]);
    verifyWarning(testCase, warningFun, 'emptyTrainTest:id');
else
    % check that outputs are time-consistent: idxTrain > idxTest > idxVal
    [idxTrain, idxTest, idxVal] = MultipleSplit(nRowsTotal, nRows, [0.75, 0.25]);

    verifyTrue(testCase, all(min(idxTrain, [], 2) > max(idxTest, [], 2)));
    verifyTrue(testCase, all(min(idxTest, [], 2) > max(idxVal, [], 2)));
end

end

% check some special cases:
trainTestRatio = [1, 0];
[idxTrain, idxTest, ~] = MultipleSplit(nRowsTotal, nRows, trainTestRatio);
verifyEqual(testCase, size(idxTrain, 2), nRows - 1);
verifyTrue(testCase, isempty(idxTest));

trainTestRatio = [0, 1];
[idxTrain, idxTest, ~] = MultipleSplit(nRowsTotal, nRows, trainTestRatio);
verifyEqual(testCase, size(idxTest, 2), nRows - 1);
verifyTrue(testCase, isempty(idxTrain));

end


function testTrainTestSplitsInputs(testCase)


% check that fractional and integer ratios produce the same output:
for i = 1:50
ts = createRandomDataStruct(); 
ts = CreateRegMatrix(ts);
nRowsTotal = size(ts.X, 1);
nRows = max(round(size(ts.X, 1)/5), 2);
    
trainSize = randperm(nRows);
% try with integers:
trainTestRatio1 = [trainSize(1), nRows - 1 - trainSize(1)];
[idxTrain1, idxTest1, idxVal1] = MultipleSplit(nRowsTotal, nRows, trainTestRatio1);

% try with univariate output:
trainTestRatio3 = size(idxTrain1, 2); % in univariate format train size is exactly as specified
[idxTrain3, idxTest3, idxVal3] = MultipleSplit(nRowsTotal, nRows, trainTestRatio3);

% try with fractional output:
trainTestRatio2 = trainTestRatio1/sum(trainTestRatio1);
[idxTrain2, idxTest2, idxVal2] = MultipleSplit(nRowsTotal, nRows, trainTestRatio2);

% check that results are equal:
verifyEqual(testCase, idxTrain1, idxTrain2);
verifyEqual(testCase, idxTest1, idxTest2);
verifyEqual(testCase, idxVal1, idxVal2);

verifyEqual(testCase, idxTrain3, idxTrain2);
verifyEqual(testCase, idxTest3, idxTest2);
verifyEqual(testCase, idxVal3, idxVal2);

end



end
