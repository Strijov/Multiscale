% Script demoCompareForecasts runs one forecasting experiments.
% It applies several competitive models to single dataet. 

% FIXIT Plese type DONE here after your changes.
addpath(genpath(cd));

% Data and models.
nameTsSheaf = 'SL2';                            % The only dataset to test.
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
report_struct.handles = {@include_subfigs, @horizontal_res_table};   
report_struct.algos = nameModel;
report_struct.headers = {'MAPE target', 'MAPE full', 'AIC'};
report_struct.res = cell(1, numDataSets);


for nDataSet = 1:numDataSets
inputStructTS = ts_struct_array{nDataSet};
workStructTS = CreateRegMatrix(inputStructTS);    % Construct regression matrix.

workStructTS = GenerateFeatures(workStructTS, generator_handles);
disp(['Generation finished. Total number of features: ', num2str(workStructTS.deltaTp)]);
[gen_fname, gen_caption] = plot_generated_feature_matrix(workStructTS.matrix, ...
                                           generator_names, workStructTS.name);


for i = 1:nModels
    disp(['Fitting model: ', nameModel{i}])
    [~, model(i), real_y] = ComputeForecastingErrors(workStructTS, K, alpha_coeff, model(i));
end

% plot 1:24 forecasts of real_y if the error does not exceed 1e3
[fname, caption] = plot_forecasting_results(real_y, model, 1:24, 1e3, ...
                                            workStructTS.name);


% VAR results are not plotted because it's unstable on samples [MxN] where
% M < N, just like our case. Feature selection is vital for it.

MAPE_full = zeros(nModels,1);
MAPE_target = zeros(nModels,1);
AIC = zeros(nModels,1);

for i = 1:nModels % FIXIT, please.
    epsilon_target = (model(i).forecasted_y(1:24) - real_y(1:24));
    MAPE_target(i) = sqrt((1/24)*norm(epsilon_target));
    epsilon_full = (model(i).forecasted_y - real_y);
    MAPE_full(i) = sqrt(1/workStructTS.deltaTr)*norm(epsilon_full);
    AIC(i) = 2*workStructTS.deltaTp + size(workStructTS.matrix, 1) * log(norm(epsilon_full));
end


report_struct.res{nDataSet} = struct('data', workStructTS.name, 'names', [],...
                             'captions', [], 'errors', [MAPE_target, MAPE_full, AIC]);
report_struct.res{nDataSet}.names = {gen_fname, fname};
report_struct.res{nDataSet}.captions = {gen_caption, caption};
%report_struct.res{nDataSet}.errors = [MAPE_target, MAPE_full, AIC];

table(MAPE_target, MAPE_full, AIC, 'RowNames', nameModel)


end

generate_tex_report(report_struct, 'CompareModels.tex');

