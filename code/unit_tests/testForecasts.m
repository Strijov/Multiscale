function tests = testForecasts

% Test forecasting methods includes the following testing scenarios:
% - testIdentity: checks that forcasts and residues are assigned correctly 
% - testMdlOutput: checks that results on the test set are equivalent to 
%                  application of .transform 
% - testTrainTestIdx: checks that various train\test inputs are handled
%                     correctly
% - testDenormalization: checks that the time series are normalized and denormalized properly


tests  = functiontests(localfunctions);

end

function testIdentity(testCase)

% Will need to use approximate equality tests:
import matlab.unittest.constraints.IsEqualTo
import matlab.unittest.constraints.AbsoluteTolerance
TOL = AbsoluteTolerance(10^(-10));

data = createRandomDataStruct();
ts = CreateRegMatrix(data);
% replace independent variables with targets:
ts.X = ts.Y;
ts.deltaTp = ts.deltaTr;

model = struct('handle', @IdentityForecast, 'name', 'Identity', 'params', [], 'transform', [],...
    'trainError', [], 'testError', [], 'unopt_flag', false, 'forecasted_y', [],...
    'bias', []);

[idxTrain, idxTest, idxVal] = MultipleSplit(size(ts.Y, 1), size(ts.Y, 1), [0.75, 0.25]); 

% First, ensure that Identity does work:
% pass target variables as X to identity Frc:
[testFrc, trainFrc, model] = IdentityForecast(ts.Y(idxTest, :), model, ts.Y(idxTrain, :));
valFrc = feval(model.transform, ts.Y(idxVal, :));

verifyEqual(testCase, trainFrc, ts.Y(idxTrain, :));
verifyEqual(testCase, testFrc, ts.Y(idxTest, :));
verifyEqual(testCase, valFrc, ts.Y(idxVal, :));


% Now see if it works with frcResiduals:
[testRes, trainRes, model] = computeForecastingResiduals(ts, model, ...
                                idxTrain, [idxVal, idxTest]);
                            
expectedTrainRes = arrayfun(@(x) zeros(x*numel(idxTrain), 1), ts.deltaTr, 'UniformOutput', 0); 
expectedTestRes = arrayfun(@(x) zeros(x*numel([idxVal, idxTest]), 1), ts.deltaTr, 'UniformOutput', 0); 

% check that residues are all zero:
verifyThat(testCase, trainRes, IsEqualTo(expectedTrainRes, 'Within', TOL));
verifyThat(testCase, testRes, IsEqualTo(expectedTestRes, 'Within', TOL));

% check that forecasts equal to the original time  series:
verifyThat(testCase, model.forecasted_y, IsEqualTo(ts.x, 'Within', TOL));

% check that errors are zero:
verifyThat(testCase, model.testError, IsEqualTo(zeros(1, numel(ts.x)), 'Within', TOL));
verifyThat(testCase, model.trainError, IsEqualTo(zeros(1, numel(ts.x)), 'Within', TOL));


end


function testEmptyInput(testCase)

% Init models to test:
nameModel = {'VAR', 'MSVR', 'Random Forest', 'Neural network'};   % Set of models. 
handleModel = {@VarForecast, @MLSSVRMethod, @TreeBaggerForecast, @NnForecast};
pars = cell(1, numel(nameModel));
pars{1} = struct('regCoeff', 2);
pars{2} = struct('kernel_type', 'rbf', 'p1', 2, 'p2', 0, 'gamma', 0.5, 'lambda', 4);
pars{3} = struct('nTrees', 25, 'nVars', 48);
pars{4} = struct('nHiddenLayers', 25);
model = struct('handle', handleModel, 'name', nameModel, 'params', pars, 'transform', [],...
    'trainError', [], 'testError', [], 'unopt_flag', false, 'forecasted_y', [],...
    'bias', []);

% Generate test data:
ts = createRandomDataStruct(3, 500);
ts = CreateRegMatrix(ts);

for i = 1:numel(model)
    [~, ~, model(i)] = computeForecastingResiduals(ts, model(i));
    
    % check that test errors have the same size but contain only nans:
    verifyEqual(testCase, size(model(i).testError), size(model(i).trainError));
    verifyTrue(testCase, all(isnan(model(i).testError)));
    % extract test frc from model.forecasted_y:
    
end

end

function testMdlOutput(testCase)

% checks that testForeacasts returned by the model are the same as
% transforms

% Init models to test:
nameModel = {'VAR', 'MSVR', 'Random Forest', 'Neural network'};   % Set of models. 
handleModel = {@VarForecast, @MLSSVRMethod, @TreeBaggerForecast, @NnForecast};
pars = cell(1, numel(nameModel));
pars{1} = struct('regCoeff', 2);
pars{2} = struct('kernel_type', 'rbf', 'p1', 2, 'p2', 0, 'gamma', 0.5, 'lambda', 4);
pars{3} = struct('nTrees', 25, 'nVars', 48);
pars{4} = struct('nHiddenLayers', 25);
model = struct('handle', handleModel, 'name', nameModel, 'params', pars, 'transform', [],...
    'trainError', [], 'testError', [], 'unopt_flag', false, 'forecasted_y', [],...
    'bias', []);

% Generate test data:
ts = createRandomDataStruct(3, 500);
ts = CreateRegMatrix(ts);

[idxTrain, idxTest, idxVal] = MultipleSplit(size(ts.Y, 1), size(ts.Y, 1), [0.75, 0.25]); 
idxTest = [idxVal, idxTest];

for i = 1:numel(model)
    [~, ~, model(i)] = computeForecastingResiduals(ts, model(i), ...
                                idxTrain, idxTest);

    % extract test frc from model.forecasted_y:
    idxFrc = idxTest;
    idxFrc = arrayfun(@(x) (idxFrc(1) - 1)*x + 1:(idxFrc(1) - 1)*x + numel(idxFrc)*x, ...
                            ts.deltaTr, 'UniformOutput', false);  
    idxFrc = cellfun(@(x, y) fliplr(numel(y) + 1 - x), idxFrc, ts.x, 'UniformOutput', false);
    
    modelTestFrc = cellfun(@(x, y) x(y), model(i).forecasted_y, idxFrc, ...
                                                'UniformOutput', false);
    
    % generate testFrc from transform:
    transformTestFrc = feval(model(i).transform, ts.X(idxTest, :));
    transformTestFrc = unravel_target_var(transformTestFrc, ts.deltaTr, ...
                                          ts.norm_div, ts.norm_subt);
    
    verifyEqual(testCase, modelTestFrc, transformTestFrc);
end

end


function testDenormalization(testCase)

% check that the time series are normalized and denormalized properly

% Will need to use approximate equality tests:
import matlab.unittest.constraints.IsEqualTo
import matlab.unittest.constraints.AbsoluteTolerance
TOL = AbsoluteTolerance(10^(-10));

% Generate test data:
ts = createRandomDataStruct(3, 200);
tsN = CreateRegMatrix(ts, 1, true); % with normalization
  
% apply the same normalization to ts.x and make another design matrix
tsUN = ts; 
ts_x = renormalize({ts.x}, tsN.norm_div, tsN.norm_subt);
[tsUN.x] = deal(ts_x{:});
tsUN = CreateRegMatrix(tsUN, 1, false); % w/o normalization


% Init models to test:
nameModel = {'MLR', 'MSVR', 'RF', 'ANN'};   % Set of models. 
handleModel = {@VarForecast, @MLSSVRMethod, @TreeBaggerForecast, @NnForecast};
pars = cell(1, numel(nameModel));
pars{1} = struct('regCoeff', 2);
pars{2} = struct('kernel_type', 'rbf', 'p1', 2, 'p2', 0, 'gamma', 0.5, 'lambda', 4);
pars{3} = struct('nTrees', 25, 'nVars', 48);
pars{4} = struct('nHiddenLayers', 25);
model = struct('handle', handleModel, 'name', nameModel, 'params', pars, 'transform', [],...
    'trainError', [], 'testError', [], 'unopt_flag', false, 'forecasted_y', [],...
    'intercept', []);

[idxTrain, idxTest, ~] = MultipleSplit(size(tsN.Y, 1), size(tsN.Y, 1), [0.75, 0.25]); 

for i = 1:numel(model)
    [~, ~, modelN] = computeForecastingResiduals(tsN, model(i), idxTrain, idxTest);
    [~, ~, modelUN] = computeForecastingResiduals(tsUN, model(i), idxTrain, idxTest);
    % denormalize forecasts of modelUN:
    verifyEqual(testCase, modelUN.trainError, modelUN.trainError);
    verifyEqual(testCase, modelN.testError, modelN.testError);
    
    verifyThat(testCase, modelN.bias, IsEqualTo(modelUN.bias.*tsN.norm_div, 'Within', TOL));
    renFrc = renormalize(modelN.forecasted_y, tsN.norm_div, tsN.norm_subt);
    denFrc = denormalize(modelUN.forecasted_y, tsN.norm_div, tsN.norm_subt);
    verifyThat(testCase, modelUN.forecasted_y, IsEqualTo(renFrc, 'Within', TOL));
    verifyThat(testCase, modelN.forecasted_y, IsEqualTo(denFrc, 'Within', TOL));
end


end

function testTrainTestIdx(testCase)

% check that the code produces warnings if idxTrain and idxTest intersect or idxTest preceeds idxTrain

% use random data and identity forecast
ts = createRandomDataStruct();
ts = CreateRegMatrix(ts);
ts.X = ts.Y;
ts.deltaTp = ts.deltaTr;

model = struct('handle', @IdentityForecast, 'name', 'Identity', 'params', [], 'transform', [],...
    'trainError', [], 'testError', [], 'unopt_flag', false, 'forecasted_y', [],...
    'intercept', []);
[idxTrain, idxTest, idxVal] = MultipleSplit(size(ts.Y, 1), size(ts.Y, 1), [0.75, 0.25]); 


% pass unsorted tested indicess, this should proceed w/o warnings:
noWarningFunc = @() computeForecastingResiduals(ts, model, idxTrain, [idxTest, idxVal]);
verifyWarningFree(testCase,  noWarningFunc);

% pass idxTest and idxTrain in the wrong order:  
noWarningFunc = @() computeForecastingResiduals(ts, model, [idxVal, idxTest], idxTrain);
verifyWarning(testCase, noWarningFunc, ...
                        'idxTrainTest:id');
                    
                    
% check that forecasts do not depend on the order of indices incide train
% and test:
idxTest = [idxVal, idxTest];
[testRes, trainRes, model] = computeForecastingResiduals(ts, model, ...
                                idxTrain, idxTest);
idxShuffle = randperm(numel(idxTrain));
idxTrain = idxTrain(idxShuffle);
idxShuffle = randperm(numel(idxTest));
idxTest = idxTest(idxShuffle);
[testResShuffled, trainResShuffled, modelShuffled] = ...
            computeForecastingResiduals(ts, model, idxTrain, idxTest);  
verifyEqual(testCase, testRes, testResShuffled);
verifyEqual(testCase, trainRes, trainResShuffled);
verifyEqual(testCase, model.forecasted_y, modelShuffled.forecasted_y);
end