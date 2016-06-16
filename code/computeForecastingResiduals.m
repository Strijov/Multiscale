function [testRes, trainRes, model] = computeForecastingResiduals(ts, ...
                                                model, ...
                                                idxTrain,...
                                                idxTest)


% The function fits the foreasting model to the target time series on the
% train set and applies it to foreast the test set
% Input:
% ts - basic time series structure (for more details see createRegMatrix.m)
%       ts.X [m x n] and ts.Y [m x ts.deltaTr] store the independent and
%        the target variables of the design matrix
% model - basic model structure (see Systemdocs.doc)
%       model.handle: handle to the forecasting model, 
%       model.params: model parameters
%       model.testErrors, model.trainErrors: forecasting errors (symMAPE)
%       model.forecasted_y: [(T- (n - ts.deltaTr) x 1)] forcasted values of the
%       original time series ts.x [T x 1]
% idxTrain - indices of train objects from design matrix
% idxTest   - test (validation) indices of test objects. FIXIT Validation set is used as test
%        set...


if nargin == 2
    idxTrain = 1:size(ts.X, 1);
    idxTest = [];
end
if nargin == 3
    trainTestValRatio = idxTrain;
    [idxTrain, idxTest, ~] = TrainTestSplit(size(ts.X, 1), trainTestValRatio);
end
idxTrain = sort(idxTrain);
idxTest = sort(idxTest);

[forecastY, matTrainForecastY, model] = feval(model.handle, ts.X(idxTest, :), model, ...
                            ts.X(idxTrain, :), ts.Y(idxTrain, :)); 

% Remember that forecasts, ts.X and ts.Y are normalized, while ts.x are not!
% Unravel forecasts from matrices to vecors and denormalize forecasts:
% unravel_target_var returns a cell array of size [1 x nTimeSeries]
if isempty(model.forecasted_y)
    model.forecasted_y = cell(1, numel(ts.x));
    model.forecasted_y = cellfun(@(x) zeros(numel(x), 1), ts.x, 'UniformOutput', false);
end
nPredictions = size(ts.Y, 2)/sum(ts.deltaTr);
[model.forecasted_y, idxTrain] = addFrcToModel(model.forecasted_y, matTrainForecastY, idxTrain, ts, nPredictions);                         
[model.forecasted_y, idxTest] = addFrcToModel(model.forecasted_y, forecastY, idxTest, ts, nPredictions);                         
if ~checkTrainTestIdx(idxTrain, idxTest)
    disp('compFrc: idxTrain and idxTest intersect')
end
% compute frc residuals for each time series (cell array [1 x nTimeSeries])
residuals = calcResidualsByTs(model.forecasted_y, ts.x, ts.deltaTp);
testRes = cellfun(@(x, y) x(y), residuals, idxTest, 'UniformOutput', false);
trainRes = cellfun(@(x, y) x(y), residuals, idxTrain, 'UniformOutput', false);

testFrc = cellfun(@(x, y) x(y), model.forecasted_y, idxTest, 'UniformOutput', false);
trainFrc = cellfun(@(x, y) x(y), model.forecasted_y, idxTrain, 'UniformOutput', false);

testY = cellfun(@(x, y) x(y), ts.x, idxTest, 'UniformOutput', false);
trainY = cellfun(@(x, y) x(y), ts.x, idxTrain, 'UniformOutput', false);

% split them into 2 arrays of residuals, each of size [1 x nTimeSeries]

%{
[testRes, trainRes] = splitForecastsByTs(residuals, idxTrain, idxTest, ...
                                                  ts.deltaTr*nPredictions);
[testFrc, trainFrc] = splitForecastsByTs(model.forecasted_y, idxTrain, idxTest, ...
                                                  ts.deltaTr*nPredictions);                                              


[testY, trainY] = splitForecastsByTs(ts.x, idxTrain, idxTest, ...
                                                  ts.deltaTr*nPredictions);  
%}                                            
model.testError = cellfun(@calcSymMAPE, testY, testFrc);
model.trainError = cellfun(@calcSymMAPE, trainY, trainFrc);


end


function checkRes = checkTrainTestIdx(idxTrain, idxTest)

checkRes = ~any(cell2mat(cellfun(@(x, y) ismember(x, y), idxTrain, idxTest,...
                            'UniformOutput', false)));

end

function [modelFrc, idxFrc] = addFrcToModel(modelFrc, newFrc, idxFrc, ts, nPred)

forecasted_y = unravel_target_var(newFrc, ...
                                    ts.deltaTr, ts.norm_div, ts.norm_subt);
%idxFrc = cellfun(@(x) idxFrc(1):idxFrc(1) + x, mat2cell(idxFrc, 1, numel(idxFrc)), ...
%                                'UniformOutput', false);
idxFrc = arrayfun(@(x) (idxFrc(1) - 1)*x + 1:(idxFrc(1) - 1)*x + numel(idxFrc)*x, ...
    ts.deltaTr*nPred, 'UniformOutput', false);                             
modelFrc = cellfun(@(x, y, z) addVecByIdx(x, y, z), ...
                                modelFrc, forecasted_y, idxFrc, ...
                                'UniformOutput', false);


end


function oldVec = addVecByIdx(oldVec, addVec, idxAdd)

oldVec(idxAdd) = addVec;

end

% DISCUSS: Compute quality of forecast in extra function? Additional quality/error
% functions?



         
        
        
    