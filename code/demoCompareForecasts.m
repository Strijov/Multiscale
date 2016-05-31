% Script demoCompareForecasts runs one forecasting experiment.
% It applies several competitive models to single dataet. 

% FIXIT Plese type DONE here after your changes.
addpath(genpath(cd));

% Data and models.
nameModel = {'VAR', 'SVR', 'Random Forest', 'Neural network'};   % Set of models. 
handleModel = {@VarForecast, @SVRMethod, @TreeBaggerForecast, @NnForecast};
nModels = numel(nameModel);

model = struct('handle', handleModel, 'name', nameModel, 'params', [], 'obj', [],...
    'trainError', [], 'testError', [], 'unopt_flag', true, 'forecasted_y', []);

% Experiment settings. 
alpha_coeff = 0; % FIXIT Please explain. 
K = 1; % FIXIT Please explain. 

%Generating extra features:
generator_names = {'Identity'};%{'SSA', 'NW', 'Cubic', 'Conv'};
generator_handles = {@IdentityGenerator}; %{@SsaGenerator, @NwGenerator, @CubicGenerator, @ConvGenerator};

% Load and prepare dataset.
%LoadAndSave('EnergyWeatherTS/orig');
ts_struct_array  = LoadTimeSeries('EnergyWeather');
numDataSets = numel(ts_struct_array);

report_struct = struct('handles', [], 'algos', [], 'headers', [],...
                 'res',  []); 
report_struct.handles = {@include_subfigs, @vertical_res_table};   
report_struct.algos = nameModel;
report_struct.headers = {'MAPE test', 'MAPE train', 'AIC'};
report_struct.res = cell(1, numDataSets);
figs = struct('names', cell(1,2), 'captions', cell(1,2));

for nDataSet = 1:numDataSets    
StructTS = ts_struct_array{nDataSet};

StructTS = CreateRegMatrix(StructTS);    % Construct regression matrix.
[fname, caption] = plot_ts(StructTS);
figs(1).names = fname;
figs(1).captions = caption;


StructTS = GenerateFeatures(StructTS, generator_handles);
disp(['Generation finished. Total number of features: ', num2str(StructTS.deltaTp)]);
[gen_fname, gen_caption] = plot_generated_feature_matrix(StructTS.matrix, ...
                                generator_names, ...
                                StructTS.name, StructTS.dataset);

MAPE_test = zeros(nModels,1);
MAPE_train = zeros(nModels,1); 
model = struct('handle', handleModel, 'name', nameModel, 'params', [], 'obj', [],...
    'trainError', [], 'testError', [], 'unopt_flag', true, 'forecasted_y', []);


for i = 1:nModels
    disp(['Fitting model: ', nameModel{i}])
    [MAPE_test(i), MAPE_train(i), model(i)] = ComputeForecastingErrors(StructTS, K, alpha_coeff, model(i));
end


N_PREDICTIONS = 5;
idx_target = 1:min(StructTS.deltaTr*N_PREDICTIONS, numel(StructTS.x));
% plot idx_target forecasts of real_y if the error does not exceed 1e3

[fname, caption] = plot_forecasting_results(StructTS.x, model, 1:StructTS.deltaTr, 1e3, ...
                                         StructTS.name, StructTS.dataset);

figs(2).names = {gen_fname, fname};
figs(2).captions = {gen_caption, caption};

% VAR results are not plotted because it's unstable on samples [MxN] where
% M < N, just like our case. Feature selection is vital for it.

AIC = zeros(nModels,1);

for i = 1:nModels % FIXIT, please.
    AIC(i) = 2*StructTS.deltaTp + size(StructTS.matrix, 1) * log(norm(epsilon_full));
end



report_struct.res{nDataSet} = struct('data', StructTS.name, 'errors', [MAPE_target, MAPE_full, AIC]);
report_struct.res{nDataSet}.figs = figs;

table(MAPE_test, MAPE_train, AIC, 'RowNames', nameModel)


end
save('report_struct_EW.mat', 'report_struct');
generate_tex_report(report_struct, 'CompareModels_EW.tex');

