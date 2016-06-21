function tests = testForecasts

tests  = functiontests(localfunctions);

end

function testIdentity(testCase)

data = createRandomDataStruct();
ts = CreateRegMatrix(data);
% replace independent variables with targets:
ts.X = ts.Y;
ts.deltaTp = ts.deltaTr;

model = struct('handle', @IdentityForecast, 'name', 'Identity', 'params', [], 'transform', [],...
    'trainError', [], 'testError', [], 'unopt_flag', false, 'forecasted_y', [],...
    'intercept', []);

[idxTrain, idxTest, idxVal] = MultipleSplit(size(ts.Y, 1), size(ts.Y, 1), [0.75, 0.25]); 

% First, ensure that Identity does work:
% pass taret variables as X to identity Frc:
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
verifyEqual(testCase, trainRes, expectedTrainRes);
verifyEqual(testCase, testRes, expectedTestRes);

% check that forecasts equal the original time  series:
verifyEqual(testCase, model.forecasted_y, ts.x);

% check that errors are zero:
verifyEqual(testCase, model.testError, zeros(1, numel(ts.x)));
verifyEqual(testCase, model.trainError, zeros(1, numel(ts.x)));


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
    'intercept', []);

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