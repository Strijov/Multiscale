% Script demoCompareForecasts runs one forecasting experiment.
% It applies several competitive models to single dataet. 

% FIXIT Plese type DONE here after your changes.
addpath(genpath(cd));

% Data and models.
nameModel = {'VAR', 'SVR', 'Neural network'};   % Set of models. 
handleModel = {@VarForecast, @SVRMethod, @NnForecast};
nModels = numel(nameModel);

model = struct('handle', handleModel, 'name', nameModel, 'params', [], ...
    'error', [], 'unopt_flag', true, 'forecasted_y', []);

% Experiment settings. 
alpha_coeff = 0; % FIXIT Please explain. 
K = 1; % FIXIT Please explain. 

%Generating extra features:
generator_names = {'SSA', 'NW', 'Cubic', 'Conv'};
generator_handles = {@SsaGenerator, @NwGenerator, @CubicGenerator, @ConvGenerator};

% Load and prepare dataset.
ts_struct_array  = LoadTimeSeries();
numDataSets = numel(ts_struct_array);

report_struct = struct('handles', [], 'algos', [], 'headers', [],...
                 'res',  []); 
report_struct.handles = {@include_subfigs, @vertical_res_table};   
report_struct.algos = nameModel;
report_struct.headers = {'MAPE target', 'MAPE full', 'AIC'};
report_struct.res = cell(1, numDataSets);
figs = struct('names', cell(1,2), 'captions', cell(1,2));

for nDataSet = 1:numDataSets
inputStructTS = ts_struct_array{nDataSet};
[fname, caption] = feval(inputStructTS(1).plot_handle, inputStructTS(1));
figs(1).names = fname;
figs(1).captions = caption;

workStructTS = CreateRegMatrix(inputStructTS);    % Construct regression matrix.

workStructTS = GenerateFeatures(workStructTS, generator_handles);
disp(['Generation finished. Total number of features: ', num2str(workStructTS.deltaTp)]);
[gen_fname, gen_caption] = plot_generated_feature_matrix(workStructTS.matrix, ...
                                generator_names, ...
                                workStructTS.name, inputStructTS(1).dataset);

for i = 1:nModels
    disp(['Fitting model: ', nameModel{i}])
    [~, model(i), real_y] = ComputeForecastingErrors(workStructTS, K, alpha_coeff, model(i));
end


N_PREDICTIONS = 5;
idx_target = 1:min(workStructTS.deltaTr*N_PREDICTIONS, numel(real_y));
% plot idx_target forecasts of real_y if the error does not exceed 1e3
try
[fname, caption] = plot_forecasting_results(real_y, model, 1:workStructTS.deltaTr, 1e3, ...
                                         workStructTS.name, inputStructTS(1).dataset);
catch
    disp('')
end
figs(2).names = {gen_fname, fname};
figs(2).captions = {gen_caption, caption};

% VAR results are not plotted because it's unstable on samples [MxN] where
% M < N, just like our case. Feature selection is vital for it.

MAPE_full = zeros(nModels,1);
MAPE_target = zeros(nModels,1);
AIC = zeros(nModels,1);

for i = 1:nModels % FIXIT, please.
    epsilon_full = (model(i).forecasted_y - real_y);
    epsilon_target = epsilon_full(idx_target);
    MAPE_target(i) = mean(abs(epsilon_target./real_y(idx_target)));
    MAPE_full(i) = mean(abs(epsilon_full./real_y)); % is this MAPE??
    AIC(i) = 2*workStructTS.deltaTp + size(workStructTS.matrix, 1) * log(norm(epsilon_full));
end



report_struct.res{nDataSet} = struct('data', workStructTS.name, 'errors', [MAPE_target, MAPE_full, AIC]);
report_struct.res{nDataSet}.figs = figs;

table(MAPE_target, MAPE_full, AIC, 'RowNames', nameModel)


end
save('report_struct_NN.mat', 'report_struct');
generate_tex_report(report_struct, 'CompareModels_NNdata.tex');

