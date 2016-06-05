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
feature_selection_mdl = struct('handle', @IdentityGenerator, 'params', {});

% A range of forcast horizon values:
frc_horizons = [1:10];

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


MAPE_train = zeros(1, length(frc_horizons));
MAPE_test = zeros(1, length(frc_horizons));
AIC = zeros(1, length(frc_horizons));

%--------------------------------------------------------------------------
% Here comes the main part: calc errors and plot forecasts by horizon
% length
for i = 1:length(frc_horizons)
    % Add regression matrix to the main structure:
    ts = CreateRegMatrix(ts, frc_horions(i));
    % Generate more eatures:
    ts = GenerateFeatures(ts, generators);
    % Select best features:
    ts = FeatureSelection(ts, feature_selection_mdl);

    % Init models:
    model = struct('handle', handleModel, 'name', nameModel, 'params', [], 'obj', [],...
        'trainError', [], 'testError', [], 'unopt_flag', true, 'forecasted_y', []);

    % Train models, obtain forecasts and calc errors:
    [MAPE_test(i), MAPE_train(i), AIC(i), model] = calcErrorsByModel(ts, model,...
                                                          idxTrain, idxTest);
    disp('Results with all generators:')
    table(MAPE_test, MAPE_train, AIC, 'RowNames', nameModel)

    %Plot results:
    [fname, caption, ~, ~] = plot_forecasting_results(ts, model, 1:ts.deltaTr, ...
                                                       1e3, FOLDER, '_fs_all');
    % Put results into report_struct
    figs = struct('names', cell(1), 'captions', cell(1));
    figs(1).names = {gen_fname, fname};
    figs(1).captions = {gen_caption, caption};
    report_struct.res{i} = struct('data', 'All', 'errors', [MAPE_test, MAPE_train, AIC]);
    report_struct.res{i}.figs = figs;    
end


%--------------------------------------------------------------------------
% save results and generate report:
save('report_struct_fs.mat', 'report_struct');
generate_tex_report(report_struct, 'FeatureSelection.tex');




end