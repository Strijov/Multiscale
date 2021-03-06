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

ERROR_HANDLE = @calcMAPE;
%ERROR_HANDLE = @calcSymMAPE; @calcMASE;

if nargin == 2
    idxTrain = 1:size(ts.X, 1);
    idxTest = [];
end
if nargin == 3
    trainTestRatio = idxTrain;
    [idxTrain, idxTest, ~] = TrainTestSplit(size(ts.X, 1), trainTestRatio);
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

% addFrc unravels forecasts and denormalizes them:
nPredictions = size(ts.Y, 2)/sum(ts.deltaTr);
[model.forecasted_y, idxTrain] = addFrcToModel(model.forecasted_y, matTrainForecastY, idxTrain, ts, nPredictions);                         
[model.forecasted_y, idxTest] = addFrcToModel(model.forecasted_y, forecastY, idxTest, ts, nPredictions);                         

if ~checkTrainTestIdx(idxTrain, idxTest)
    warning('idxTrainTest:id', 'idxTrain and idxTest intersect');
    %disp('compFrc: idxTrain and idxTest intersect')
end

% compute frc residuals for each time series (cell array [1 x nTimeSeries])
residuals = calcResidualsByTs(model.forecasted_y, ts.x, ts.deltaTp);


if ~isfield(model, 'bias') || isempty(model.bias)
    trainRes = cellfun(@(x, y) x(y), residuals, idxTrain, 'UniformOutput', false);
    model.bias = cell2mat(cellfun(@(x) mean(x), trainRes, 'UniformOutput', false));
end
%model.forecasted_y = cellfun(@(x, y) x + y, model.forecasted_y, model.bias, 'UniformOutput', false);
%residuals = cellfun(@(x, y) x - y, residuals, model.bias, 'UniformOutput', false);



testRes = cellfun(@(x, y) x(y), residuals, idxTest, 'UniformOutput', false);
trainRes = cellfun(@(x, y) x(y), residuals, idxTrain, 'UniformOutput', false);

testFrc = cellfun(@(x, y) x(y), model.forecasted_y, idxTest, 'UniformOutput', false);
trainFrc = cellfun(@(x, y) x(y), model.forecasted_y, idxTrain, 'UniformOutput', false);

testY = cellfun(@(x, y) x(y), ts.x, idxTest, 'UniformOutput', false);
trainY = cellfun(@(x, y) x(y), ts.x, idxTrain, 'UniformOutput', false);
                                            
model.testError = cellfun(ERROR_HANDLE, testY, testFrc);
model.trainError = cellfun(ERROR_HANDLE, trainY, trainFrc);


end


function checkRes = checkTrainTestIdx(idxTrain, idxTest)

if isempty(idxTrain) || isempty(idxTest)
    checkRes = true;
    return
end
checkRes = all(cell2mat(cellfun(@(x, y) max(x) < min(y), idxTrain, idxTest,...
                            'UniformOutput', false)));
checkRes = checkRes & ~any(cell2mat(cellfun(@(x, y) ismember(x, y), idxTrain, idxTest,...
                            'UniformOutput', false)));

end

function [modelFrc, idxFrc] = addFrcToModel(modelFrc, newFrc, idxFrc, ts, nPred)

if isempty(idxFrc)
    idxFrc = cell(size(modelFrc));
    return
end
forecasted_y = unravel_target_var(newFrc, ...
                                    ts.deltaTr, ts.norm_div, ts.norm_subt);
idxFrc = arrayfun(@(x) (idxFrc(1) - 1)*x + 1:(idxFrc(1) - 1)*x + numel(idxFrc)*x, ...
    ts.deltaTr*nPred, 'UniformOutput', false);  
idxFrc = cellfun(@(x, y) fliplr(numel(y) + 1 - x), idxFrc, ts.x, 'UniformOutput', false);  
modelFrc = cellfun(@(x, y, z) addVecByIdx(x, y, z), ...
                                modelFrc, forecasted_y, idxFrc, ...
                                'UniformOutput', false);


end


function oldVec = addVecByIdx(oldVec, addVec, idxAdd)

oldVec(idxAdd) = addVec;

end

% DISCUSS: Compute quality of forecast in extra function? Additional quality/error
% functions?



         
        
        
    