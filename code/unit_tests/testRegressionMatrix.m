function tests = testRegressionMatrix

% Test forecasting methods includes the following testing scenarios:
% - testTimeFlow: checks that matrix inputs are consistent in time
% - testInputNans: checks that a warning is raised when all inputs are Nans
% - testNumTimeSeries: checks that shapes and dimensions are consistent
% - testUnravelForecasts: checks that the Y matrix unravels into the original time series

tests  = functiontests(localfunctions);

end

function testTimeFlow(testCase)

% checks that the time flow is consistent in regression matrix
data = createRandomDataStruct();
% set data equal to time: 
newTs = {data().time};
[data.x] = deal(newTs{:});

ts = CreateRegMatrix(data);
%verifyTrue(testCase, issorted(ts.Y, 'rows'))
verifyLessThan(testCase, max(ts.X, [], 2), min(ts.Y, [], 2));

end

function testInputNans(testCase)

% checks that warning is raised if the inputs of time series are all nans
ts = createRandomDataStruct();
ts(1).x = ts(1).x*NaN;

warningFunc = @() CreateRegMatrix(ts);
verifyWarning(testCase, warningFunc, 'regMatrixAllNans:id');


end


function testNumTimeSeries(testCase)

% checks that shapes and dimensions are consistent
data = createRandomDataStruct();
ts = CreateRegMatrix(data);

verifyEqual(testCase, numel(data), numel(ts.x));
verifyEqual(testCase, numel(data), numel(ts.time));
verifyEqual(testCase, numel(data), numel(ts.deltaTr));
verifyEqual(testCase, numel(data), numel(ts.deltaTp));
verifyEqual(testCase, size(ts.Y, 1), size(ts.X, 1));
verifyEqual(testCase, size(ts.Y, 2), sum([data().deltaTr]));
verifyEqual(testCase, size(ts.X, 2), sum([data().deltaTp]));

end


function testUnravelForecasts(testCase)

% Will need to use approximate equality tests:
import matlab.unittest.constraints.IsEqualTo
import matlab.unittest.constraints.AbsoluteTolerance
TOL = AbsoluteTolerance(10^(-10));

% checks that the Y matrix unravels into the original time series
data = createRandomDataStruct();
ts = CreateRegMatrix(data);

Y = unravel_target_var(ts.Y, ts.deltaTr, ts.norm_div, ts.norm_subt);
verifyThat(testCase, Y, IsEqualTo(ts.x, 'Within', TOL));

end