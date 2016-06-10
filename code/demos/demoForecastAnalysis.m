function demoForecastAnalysis(StructTS, model, generators, feature_selection_mdl)

TRAIN_TEST_RATIO = 0.75;
SUBSAMPLE_SIZE = 50;
N_PREDICTIONS = 10;
% Create dir for saving figures:
FOLDER = fullfile('fig/frc_analysis');
if ~exist(FOLDER, 'dir')
    mkdir(FOLDER);
end
if ~exist(fullfile(FOLDER, StructTS(1).dataset), 'dir')
    mkdir(fullfile(FOLDER, StructTS(1).dataset));
end
%--------------------------------------------------------------------------
StructTS = CreateRegMatrix(StructTS, N_PREDICTIONS);
% Split design matrix rows into subsamples of size SUBSAMPLE_SIZE
[idxTest, idxTrain] = MultipleSplit(size(StructTS.X, 1), SUBSAMPLE_SIZE, TRAIN_TEST_RATIO);
nSplits = size(idxTrain, 1);

StructTS = GenerateFeatures(StructTS, generators);
residuals = cell(nSplits, numel(StructTS.x));
    
%--------------------------------------------------------------------------
% Calc frc residuals by split: 
for i = 1:nSplits
    ts = StructTS;
    [ts, feature_selection_mdl] = FeatureSelection(ts, feature_selection_mdl, ...
                                             idxTrain(i, :), idxTest(i, :));

    [testRes, trainRes, model] = computeForecastingResiduals(ts, model, 0, idxTrain(i,:), idxTest(i,:));
    
    
end


%--------------------------------------------------------------------------
% Plot evolution of res mean and std by for each model 
plot_results(testRes, trainRes, StructTS, model, ...
                                ts.deltaTr*N_PREDICTIONS, FOLDER);
                            
                            

%--------------------------------------------------------------------------


end

function [testFrc, trainFrc] = split_forecast_by_ts(forecasts, idxTrain, idxTest, ...
                                               deltaTr)
                                           
testFrc = cell(1, numel(deltaTr)); 
trainFrc = cell(1, numel(deltaTr));
for i = 1:numel(deltaTr)
    [testFrc{i}, trainFrc{i}] = split_forecast(cell2mat(forecasts(:, i)), idxTrain, idxTest, deltaTr(i));
end

end


function [testFrc, trainFrc] = split_forecast(forecasts, idxTrain, idxTest, ...
                                               deltaTr)
                                           
nTrain = size(idxTrain, 2);
nTest = size(idxTest, 2);

testFrc = zeros(deltaTr, numel(idxTest));
trainFrc = zeros(deltaTr, numel(idxTrain));


for i = 1:size(idxTrain, 1)
    idxTrainMat = bsxfun(@plus, idxTrain(i, :), (0:deltaTr - 1)');
    idxTestMat = bsxfun(@plus, idxTest(i, :), (0:deltaTr - 1)');

    testFrc(:, (i-1)*nTest + 1:i*nTest) = forecasts(idxTestMat);
    trainFrc(:, (i-1)*nTrain + 1:i*nTrain) = forecasts(idxTrainMat);
end

end

function plot_results(testRes, trainRes, StructTS, model, ...
                                           nPredictions, FOLDER)

for i = 1:numel(testRes)
    if nPredictions(i) > 1
    plot_residuals_stats(testRes{i}', trainRes{i}',...
                         StructTS, model, FOLDER, ...
                         [regexprep(StructTS.legend{i}, ' ', '_'), ...
                         '_fs_',regexprep(model.name, ' ', '_')]);


    testRes{i} = testRes{i}(:);
    trainRes{i} = trainRes{i}(:);

    % Fit residuals to normal distribution:
    testPD = fitdist(testRes{i}, 'Normal');
    trainPD = fitdist(trainRes{i}, 'Normal');


    % Plot normal pdf and QQ-plots for train and test residuals 
    plot_residuals_npdf(testRes{i}, trainRes{i}, testPD, trainPD, ...
                        StructTS, model, FOLDER, ...
                        [regexprep(StructTS.legend{i}, ' ', '_'), ...
                        '_fs_',regexprep(model.name, ' ', '_')]);

    end
end

end

