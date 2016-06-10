function demoForecastHorizon(cellStructTs, model)


%Generating extra features:
generator_names = {'Identity'};
generator_handles = {@IdentityGenerator};
generators = struct('handle', generator_handles, 'name', generator_names, ...
                                            'replace', true, 'transform', []);
                               
% Feature selection:
pars = struct('minComps', 1);
feature_selection_mdl = struct('handle', @IdentityGenerator, 'params',  pars);

% A range of forcast horizon values:
MAX_FRC_POINTS = 10;
MIN_ROWS = 5;
max_frc = min(MAX_FRC_POINTS, floor((size(cellStructTs(1).x, 1) - cellStructTs(1).deltaTp)/...
                                            cellStructTs(1).deltaTr/MIN_ROWS));
frc_horizons = 1:max_frc;

% Init structure to generate report:
report_struct = struct('handles', [], 'algos', [], 'headers', [],...
                 'res',  []); 
report_struct.handles = {@vertical_res_table};   
report_struct.algos = {cellStructTs().legend};
report_struct.headers = {'MAPE test', 'MAPE train', 'Mean Res', 'Std Res'};
report_struct.res = cell(1);%, length(frc_horizons)); 


% If necessary, create dir where the figures will be stored
FOLDER = fullfile('fig/frc_horizon/');
if ~exist(FOLDER, 'dir')
    mkdir(FOLDER);
end
if ~exist(fullfile(FOLDER, cellStructTs(1).dataset), 'dir')
    mkdir(fullfile(FOLDER, cellStructTs(1).dataset));
end
FOLDER = fullfile(FOLDER, cellStructTs(1).dataset);

TRAIN_TEST_RATIO = 0.75;

nTimeSeries = numel(cellStructTs);
nHorizons = length(frc_horizons);
MAPE_train = zeros(nHorizons, nTimeSeries);
MAPE_test = zeros(nHorizons, nTimeSeries);
testMeanRes = zeros(nHorizons, nTimeSeries);
trainMeanRes = zeros(nHorizons, nTimeSeries); 
testStdRes = zeros(nHorizons, nTimeSeries);
trainStdRes = zeros(nHorizons, nTimeSeries);

%--------------------------------------------------------------------------
% Here comes the main part: calc errors and plot forecasts by horizon
% length
for i = 1:nHorizons
    % Add regression matrix to the main structure:
    ts = CreateRegMatrix(cellStructTs, frc_horizons(i));
    
    [idxTrain, idxTest, idxVal] = TrainTestSplit(size(ts.X, 1), 1 - TRAIN_TEST_RATIO);
    idxTest = [idxVal, idxTest];
    if isempty(idxTrain)
       max_frc = i - 1;
       break; 
    end
    % Generate more eatures:
    ts = GenerateFeatures(ts, generators, idxTrain, idxTest);
    % Select best features:
    ts = FeatureSelection(ts, feature_selection_mdl, idxTrain, idxTest);

    % (Re-)init model:
    model.unopt_flag = true;
    model.transform = [];
    % Train models, obtain forecasts and calc errors:
    [testRes, trainRes , model] = ...
              computeForecastingResiduals(ts, model, 0, idxTrain, idxTest);
    MAPE_train(i, :) = model.trainError;
    MAPE_test(i, :) = model.testError;                   
    testMeanRes(i,  :) = cellfun(@(x) nanmean(x(:)), testRes); 
    trainMeanRes(i, :) = cellfun(@(x) nanmean(x(:)), trainRes); 
    testStdRes(i,  :) = cellfun(@(x) nanstd(x(:)), testRes); 
    trainStdRes(i, :) = cellfun(@(x) nanstd(x(:)), trainRes);

    %Plot results:
    %[fname, caption, fname_m, caption_m] = plot_forecasting_results(ts, model, 1:ts.deltaTr, ...
    %                        10, FOLDER, ['_hor_', num2str(frc_horizons(i))]);
    
      
end
% Put results into report_struct
report_struct.res{i} = struct('data', 'All', 'errors', [mean(MAPE_test);...
                           mean(MAPE_train); mean(testMeanRes); mean(testStdRes)]');

save(['report_struct_horizon_', regexprep(model.name, ' ', '_') ,'.mat'], 'report_struct');
plot_mean_std(MAPE_train, MAPE_test, testMeanRes, trainMeanRes, ...
       testStdRes, trainStdRes, ts.legend, FOLDER, [ts.name,'_', regexprep(model.name, ' ', '_')]); 

%--------------------------------------------------------------------------
% save results and generate report:
%generate_tex_report(report_struct, ['FrcHorizon_', regexprep(model.name, ' ', '_') ,'.tex']);




end


function plot_mean_std(MAPE_train, MAPE_test, testMeanRes, trainMeanRes, ...
       testStdRes, trainStdRes, legends, folder, name)
   
for i = 1:numel(legends)
fig = figure;
%title(legend{i})
errorbar(1:size(trainMeanRes, 1), trainMeanRes(:, i), trainStdRes(:, i));
hold on;
errorbar(1:size(testMeanRes, 1), testMeanRes(:, i), testStdRes(:, i));
legend({'Train', 'Test'}, 'Location', 'NorthWest');
xlabel('Horizon length', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Residuals mean', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;
figname = fullfile(folder, strcat('mean_res_', name, '_', legends{i}, '.eps'));
saveas(fig, figname, 'epsc');
close(fig) 

fig = figure;
%title(legend{i})
plot(MAPE_train(:, i));
hold on;
plot(MAPE_test(:, i));
legend({'Train', 'Test'}, 'Location', 'NorthWest');
xlabel('Horizon length', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('MAPE', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;
figname = fullfile(folder, strcat('MAPE_', name, '_', legends{i}, '.eps'));
saveas(fig, figname, 'epsc');
close(fig) 


end

fig = figure;
h = plot(MAPE_train, 'linewidth', 2);
hold on;
plot(MAPE_test, '--', 'linewidth', 2);
legend(h, legends, 'Location', 'NorthWest');
xlabel('Horizon length', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('MAPE', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;
figname = fullfile(folder, strcat('MAPE_', name, '_all_ts.eps'));
saveas(fig, figname, 'epsc');
close(fig)
   
end

