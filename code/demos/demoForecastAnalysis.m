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

testRes = zeros(size(StructTS.Y, 2), numel(idxTest));
trainRes = zeros(size(StructTS.Y, 2), numel(idxTrain));
StructTS = GenerateFeatures(StructTS, generators);
    
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
    residuals = ts.x(ts.deltaTp + 1:ts.deltaTp + numel(ts.Y)) - model.forecasted_y;
    [testRes(:, (i-1)*nTest + 1:i*nTest), ...
     trainRes(:, (i-1)*nTrain + 1:i*nTrain)] = split_forecast(residuals, ...
                                idxTrain(i, :), idxTest(i, :), ts.deltaTr);
    
end

%--------------------------------------------------------------------------
% Plot evolution of res mean and std by for each model 
[stats_fname, stats_caption] = plot_residuals_stats(testRes', trainRes',...
                                        StructTS, model, FOLDER, ...
                                        ['_fs_',regexprep(model.name, ' ', '_')]);


testRes = testRes(:);
trainRes = trainRes(:);

% Fit residuals to normal distribution:
testPD = fitdist(testRes, 'Normal');
trainPD = fitdist(trainRes, 'Normal');

disp('Residuals mean and standard deviation');
table([testPD.mu, testPD.sigma; trainPD.mu, trainPD.sigma]);


% Plot normal pdf and QQ-plots for train and test residuals 
[fname, caption] = plot_residuals_npdf(testRes, trainRes, testPD, trainPD, ...
                                          StructTS, model, FOLDER, ...
                                          ['_fs_',regexprep(model.name, ' ', '_')]);

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

function [testFrc, trainFrc] = split_forecast(forecasts, idxTrain, idxTest, deltaTr)

idxTrain = bsxfun(@plus, idxTrain, (0:deltaTr - 1)');
idxTest = bsxfun(@plus, idxTest, (0:deltaTr - 1)');

testFrc = forecasts(idxTest);
trainFrc = forecasts(idxTrain);

end


