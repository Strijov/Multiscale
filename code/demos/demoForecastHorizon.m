function demoForecastHorizon(ts)

% Models
nameModel = {'VAR', 'SVR', 'Random Forest', 'Neural network'};   % Set of models. 
handleModel = {@VarForecast, @SVRMethod, @TreeBaggerForecast, @NnForecast};

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
max_frc = min(MAX_FRC_POINTS, floor((size(ts.x, 1) - ts.deltaTp)/...
                                            ts.deltaTr/MIN_ROWS));
frc_horizons = [1:max_frc];

% Init structure to generate report:
report_struct = struct('handles', [], 'algos', [], 'headers', [],...
                 'res',  []); 
report_struct.handles = {@include_subfigs, @vertical_res_table};   
report_struct.algos = nameModel;
report_struct.headers = {'MAPE test', 'MAPE train', 'AIC'};
report_struct.res = cell(1, length(frc_horizons)); 


% If necessary, create dir where the figures will be stored
FOLDER = fullfile('fig/frc_horizon/');
if ~exist(FOLDER, 'dir')
    mkdir(FOLDER);
end
if ~exist(fullfile(FOLDER, ts(1).dataset), 'dir')
    mkdir(fullfile(FOLDER, ts(1).dataset));
end

TRAIN_TEST_RATIO = 0.75;

MAPE_train = zeros(numel(nameModel), length(frc_horizons));
MAPE_test = zeros(numel(nameModel), length(frc_horizons));
AIC = zeros(numel(nameModel), length(frc_horizons));

%--------------------------------------------------------------------------
% Here comes the main part: calc errors and plot forecasts by horizon
% length
for i = 1:length(frc_horizons)
    % Add regression matrix to the main structure:
    ts = CreateRegMatrix(ts, frc_horizons(i));
    
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

    % Init models:
    model = struct('handle', handleModel, 'name', nameModel, 'params', [], 'transform', [],...
        'trainError', [], 'testError', [], 'unopt_flag', true, 'forecasted_y', []);

    % Train models, obtain forecasts and calc errors:
    [MAPE_test(:,i), MAPE_train(:,i), AIC(:, i), model] = calcErrorsByModel(ts, model,...
                                                          idxTrain, idxTest);
    disp(['Results for horizon length = ', num2str(frc_horizons(i))])
    table(MAPE_test(:,i), MAPE_train(:,i), AIC(:, i), 'RowNames', nameModel)

    %Plot results:
    [fname, caption, fname_m, caption_m] = plot_forecasting_results(ts, model, 1:ts.deltaTr, ...
                            10, FOLDER, ['_hor_', num2str(frc_horizons(i))]);
    % Put results into report_struct
    figs = struct('names', cell(1), 'captions', cell(1));
    figs(1).names = fname;
    figs(1).captions = caption;
    figs(1).names = fname_m;
    figs(1).captions = caption_m;
    report_struct.res{i} = struct('data', 'All', 'errors', [MAPE_test(:,i),...
                                                MAPE_train(:,i), AIC(:, i)]);
    report_struct.res{i}.figs = figs;    
end
report_struct.res = report_struct.res(1:max_frc);

%--------------------------------------------------------------------------
% save results and generate report:
save('report_struct_hor.mat', 'report_struct');
generate_tex_report(report_struct, 'FrcHorizon.tex');




end