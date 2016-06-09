function demoForecastAnalysis(StructTS, model, generators, feature_selection_mdl)

TRAIN_TEST_RATIO = 0.75;
SUBSAMPLE_SIZE = 50;
% Create dir for saving figures:
FOLDER = fullfile('fig/frc_analysis');
if ~exist(FOLDER, 'dir')
    mkdir(FOLDER);
end
if ~exist(fullfile(FOLDER, StructTS(1).dataset), 'dir')
    mkdir(fullfile(FOLDER, StructTS(1).dataset));
end

% Init structure to generate report:
report_struct = struct('handles', [], 'algos', [], 'headers', [],...
                 'res',  []); 
report_struct.handles = {@include_subfigs, @vertical_res_table};   
report_struct.algos = model.name;
report_struct.headers = {'Mean $\varepsilon$, test', 'Std $\varepsilon$, test',...
                        'Mean $\varepsilon$, train', 'Std $\varepsilon$, train'};


%--------------------------------------------------------------------------
StructTS = CreateRegMatrix(StructTS);
% Split design matrix rows into subsamples of size SUBSAMPLE_SIZE
[idxTest, idxTrain] = MultipleSplit(size(StructTS.X, 1), SUBSAMPLE_SIZE, TRAIN_TEST_RATIO);
nSplits = size(idxTrain, 1);
nTrain = size(idxTrain, 2);
nTest = size(idxTest, 2);

StructTS = GenerateFeatures(StructTS, generators);
residuals = cell(nSplits, numel(StructTS.x));
    
%--------------------------------------------------------------------------
% Calc frc residuals by split: 
for i = 1:nSplits
    ts = StructTS;
    [ts, feature_selection_mdl] = FeatureSelection(ts, feature_selection_mdl, ...
                                             idxTrain(i, :), idxTest(i, :));

    %idxSplit = [idxTest(i, :), idxTrain(i,:)];
    %ts.X = StructTS.X(idxSplit, :);
    %ts.Y = StructTS.Y(idxSplit, :);
    [~, ~, model] = computeForecastingErrors(ts, model, 0, idxTrain(i,:), idxTest(i,:));
    %residuals = [residuals, calcResidualsByTs(model.forecasted_y, ts.x, ts.deltaTp)];
    residuals(i, :) = calcResidualsByTs(model.forecasted_y, ts.x, ts.deltaTp);
    %[testRes(:, (i-1)*nTest + 1:i*nTest), ...
    % trainRes(:, (i-1)*nTrain + 1:i*nTrain)] = split_forecast(residuals, ...
    %                            idxTrain(i, :), idxTest(i, :), ts.deltaTr);
    
end
[testRes, trainRes] = split_forecast_by_ts(residuals, idxTrain, idxTest, ...
                                           ts.deltaTr*size(StructTS.Y, 2)/sum(ts.deltaTr));

%--------------------------------------------------------------------------
% Plot evolution of res mean and std by for each model 

for i = 1:numel(testRes)
[stats_fname, stats_caption] = plot_residuals_stats(testRes{i}', trainRes{i}',...
                                        StructTS, model, FOLDER, ...
                                        ['_fs_',regexprep(model.name, ' ', '_')]);


testRes{i} = testRes{i}(:);
trainRes{i} = trainRes{i}(:);

% Fit residuals to normal distribution:
testPD = fitdist(testRes{i}, 'Normal');
trainPD = fitdist(trainRes{i}, 'Normal');

disp('Residuals mean and standard deviation');
table([testPD.mu, testPD.sigma; trainPD.mu, trainPD.sigma]);


% Plot normal pdf and QQ-plots for train and test residuals 
[fname, caption] = plot_residuals_npdf(testRes{i}, trainRes{i}, testPD, trainPD, ...
                                          StructTS, model, FOLDER, ...
                                          ['_fs_',regexprep(model.name, ' ', '_')]);

end
figs = struct('names', cell(1), 'captions', cell(1));
figs.names = [stats_fname, fname];
figs.captions = [stats_caption, caption];
report_struct.res = struct('data', ts.name, 'errors', [testPD.mu, testPD.sigma, ...
                                                trainPD.mu, trainPD.sigma]);
report_struct.res.figs = figs;
                                      
%--------------------------------------------------------------------------
% save results and generate report:
%save(['report_struct_fa_', StructTS.name ,'.mat'], 'report_struct');
%generate_tex_report(report_struct, 'FrcAnalysis.tex');


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

