function [testMAPE, trainMAPE] = demoForecastAnalysis(tsStructArray, model, generators, feature_selection_mdl)

TRAIN_TEST_VAL_RATIO = [0.5, 0.4, 0.1];
SUBSAMPLE_SIZE = 50;
N_PREDICTIONS = 1;

% Check arguments:
if nargin < 2 || isempty(generators)
generators = struct('handle', @IdentityGenerator, 'name', 'Identity', ...
                                          'replace', false, 'transform', []);
end
if nargin < 3 || isempty(feature_selection_mdl)
feature_selection_mdl = struct('handle', @IdentityGenerator, 'params', []);    
end

% Create dir for saving figures:
FOLDER = fullfile('fig/frc_analysis');
if ~exist(FOLDER, 'dir')
    mkdir(FOLDER);
end
if ~exist(fullfile(FOLDER, tsStructArray{1}(1).dataset), 'dir')
    mkdir(fullfile(FOLDER, tsStructArray{1}(1).dataset));
end
%--------------------------------------------------------------------------
% put all time series from the dataset into a huge desigm matrix:
ts = MergeDataset(tsStructArray, N_PREDICTIONS);
%ts = CreateRegMatrix(structTsArray, N_PREDICTIONS);

% Split design matrix rows into train and test subsamples
[idxTrain, idxPreTrain, idxTest] = MultipleSplit(size(ts.X, 1), size(ts.X, 1), TRAIN_TEST_VAL_RATIO);

%--------------------------------------------------------------------------
tsGen = GenerateFeatures(ts, generators, idxPreTrain, [idxTrain, idxTest]);
[tsPT, feature_selection_mdl] = FeatureSelection(tsGen, feature_selection_mdl, ...
                                             idxPreTrain, [idxTrain, idxTest]);
[~, preTrainRes, model] = computeForecastingResiduals(tsPT, model, idxPreTrain, [idxTrain, idxTest]);


% Split design matrix rows into subsamples of size SUBSAMPLE_SIZE
[idxTrain, ~, idxTest] = MultipleSplit(size(ts.X, 1) - numel(idxPreTrain), ...
                            SUBSAMPLE_SIZE, TRAIN_TEST_VAL_RATIO);
nSplits = size(idxTrain, 1);
trainMAPE = zeros(nSplits, 1);    
testMAPE = zeros(nSplits, 1);    
testStats = zeros(nSplits, 2*numel(ts.x));
trainStats = zeros(nSplits, 2*numel(ts.x));
%--------------------------------------------------------------------------
% Calc frc residuals by split: 
for i = 1:nSplits
    [ts, feature_selection_mdl] = FeatureSelection(tsGen, feature_selection_mdl, ...
                                             idxTrain(i, :), idxTest(i, :));

    [testRes, trainRes, model] = computeForecastingResiduals(ts, model, idxTrain(i,:), idxTest(i,:));
    
    trainMAPE(i) = mean(model.trainError);
    testMAPE(i) = mean(model.testError);
    testStats(i, :) = cell2mat(cellfun(@(x) stats(x), testRes, 'UniformOutput', false));
    trainStats(i, :) = cell2mat(cellfun(@(x) stats(x), trainRes, 'UniformOutput', false));
end
save(['res_',model.name, '_', ts.legend{1}, '.mat'], 'testMAPE', 'trainMAPE', 'testStats', 'trainStats');
trainMAPE = mean(trainMAPE);
testMAPE = mean(testMAPE);
disp(model.name)
disp([trainMAPE, testMAPE])
%--------------------------------------------------------------------------
% Plot evolution of res mean and std by for each model 
plot_results(testRes, trainRes, ts, model, N_PREDICTIONS, FOLDER);
                            
                            

%--------------------------------------------------------------------------


end

function res = stats(x)

res = [mean(x(:)), std(x(:))];
end

function plot_results(testRes, trainRes, StructTS, model, ...
                                           nPredictions, FOLDER)
                                       
                                       
if numel(StructTS.x) > 1
    str = '_mult_';
else
    str = '_marg_';
end

plot_forecasting_results(StructTS, model, nPredictions, 10, FOLDER, str);                                       
                  
for i = 1:numel(testRes)
    if nPredictions*ts.deltaTr(i) > 1
    plot_residuals_stats(testRes{i}', trainRes{i}',...
                         StructTS, model, FOLDER, ...
                         [regexprep(StructTS.legend{i}, ' ', '_'), ...
                         str ,regexprep(model.name, ' ', '_')]);


    testRes{i} = testRes{i}(:);
    trainRes{i} = trainRes{i}(:);

    % Fit residuals to normal distribution:
    testPD = fitdist(testRes{i}, 'Normal');
    trainPD = fitdist(trainRes{i}, 'Normal');


    % Plot normal pdf and QQ-plots for train and test residuals 
    plot_residuals_npdf(testRes{i}, trainRes{i}, testPD, trainPD, ...
                        StructTS, model, FOLDER, ...
                        [regexprep(StructTS.legend{i}, ' ', '_'), ...
                        regexprep(model.name, ' ', '_')]);

    end
end

end



