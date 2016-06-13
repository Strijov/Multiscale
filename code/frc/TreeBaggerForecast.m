function [test_forecast, train_forecast, model] = TreeBaggerForecast(validationX, model, trainX, trainY)
% Compute forecasts using 'Random Forest' model with fixed parameters.
%
% Input:
% validationX [mtest x nx] feature row for the forecasted points
% model [struct] containing model handle and parameters:
%   model.obj stores the trained model
% trainX, trainY store training data:
%   trainX [mtrain x nx] stores features
%   trainY [mtrain x ny] stores target variables
% 
%
% Output:
% test_forecast  [mtest x ny] forecasted values of test y (regression of x)
% train_forecast  [mtrain x ny] forecasted values of train y (regression of x)
% model - ipdated model structure

N_PREDICTORS = 20;
N_TREES = 50;
N_RAND_SEARCH_TRIALS = 100;

if model.unopt_flag
    model = tb_random_search(trainX, trainY, model, N_RAND_SEARCH_TRIALS);    
    model.unopt_flag = false;
    
end    

model = train_tree_bagger(trainX, trainY, model);

train_forecast = feval(model.transform, trainX);
test_forecast = feval(model.transform, validationX);


end

function model = train_tree_bagger(trainX, trainY, model)

tb = cell(1, size(trainY, 2));
for j = 1:size(trainY, 2)
tb{j} = TreeBagger(model.params.nTrees, trainX, trainY(:, j),...
                              'Method', 'regression', ...
                              'NVarToSample', model.params.nVars);
                          
end

model.transform = @(x) transform(x, tb);

end


function frc = transform(X, tb)

frc = zeros(size(X, 1), numel(tb));
for j = 1:numel(tb)
    frc(:, j) = tb{j}.predict(X);
end
    

end


function model = tb_random_search(X, Y, model, nTrials)

nTreesRange = [5, 50]; % min and max values
nVarsRange = [10, 50]; % min and max values

lstTr = round(rand(1, nTrials)*nTreesRange(2) + nTreesRange(1));
lstVars = round(rand(1, nTrials)*nVarsRange(2) + nVarsRange(1));
%idxSplits = randomSplit(size(X, 1), size(X, 1) - nTrials + 1);

error = zeros(1, nTrials);
for i = 1:nTrials
    [Train, Test] = crossvalind('HoldOut', size(X, 1), 0.25); 
    model.params.nTrees = lstTr(i);
    model.params.nVars = lstVars(i);
    model = train_tree_bagger(X(Train, :), Y(Train, :), model);
    res = feval(model.transform, X(Test, :)) - Y(Test, :);
    error(i) = mean(sqrt(mean(res.*res, 2)));
end
[~, optIdx] = min(error);
model.params.nTrees = lstTr(optIdx);  
model.params.nVars = lstVars(optIdx); 


plotInterpolated(lstTr', lstVars', error', ...
                                {'nTrees', 'nVars'});

end

function plotInterpolated(X, Y, Z, names)
F = scatteredInterpolant([X, Y], Z);

fig = figure;
[Xq,Yq] = meshgrid(linspace(min(X), max(X), 100), linspace(min(Y), max(Y), 100));
Zq = F(Xq,Yq);
surf(Xq,Yq,Zq);
xlabel(names{1},'fontweight','b'), ylabel(names{2},'fontweight','b');
zlabel('MSRE','fontweight','b');
saveas(fig, ['tb_rand_search_',regexprep(names{1}, '$', ''), '_',...
                                regexprep(names{2}, '$', '')], 'fig');
close(fig);



end