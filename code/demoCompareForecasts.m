% Script demoCompareForecasts runs one forecasting experiments.
% It applies several competitive models to single dataet. 

% FIXIT Plese type DONE here after your changes.
% Data preparation. Time series length are fixed and precomputed.
% addpath(genpath(cd));
% filename = 'data/orig/SL2.xls';
% sheet = 'Arkusz1';
% xlRange = 'D3:AA1098';
% ts0 = xlsread(filename,sheet,xlRange);
% ts0 = reshape(ts0', numel(ts0), 1);
% 
% tmp = xlsread('data/orig/weatherdata.xls', 'weatherdata', 'E2:J1093');
% ts{1} = ts0;
% for i = [1:size(tmp, 2)]
%     ts{i+1} = tmp(:, i);
% end
% ts_length = 155;
% ts_legend = {'Consumption', 'Max Temperature','Min Temperature','Precipitation','Wind','Relative Humidity','Solar'};
% time_step = {1, 24,24,24,24,24,24};
% self_deltaTp = {6*24,6,6,6,6,6,6};
% self_deltaTr = {24,1,1,1,1,1,1};
% tmp1 = linspace(numel(ts{2}), numel(ts{2}) - 7 * ts_length, ts_length+1);
% tmp2 = linspace(numel(ts{1}), numel(ts{1}) - 24 * 7 * ts_length, ts_length+1);
% time_points = {tmp2,tmp1,tmp1,tmp1,tmp1,tmp1, tmp1};
% inputStructTS = struct('x', ts, 'time_step', time_step, 'legend', ts_legend, 'deltaTp', self_deltaTp, 'deltaTr', self_deltaTr, 'time_points', time_points);

% Data and models.
nameTsSheaf = 'SL2';                            % The only dataset to test.
nameModel = {'VAR', 'SVR', 'Neural network'};   % Set of models. 
handleModel = {@VarForecast, @SVRMethod, @NnForecast};
nModels = numel(nameModel);

% Experiment settings. 
alpha_coeff = 0; % FIXIT Please explain. 
K = 1; % FIXIT Please explain. 

% Load and prepare dataset.
inputStructTS = LoadTimeSeriesSheaf(nameTsSheaf);
workStructTS = CreateRegMatrix(inputStructTS);    % Construct regression matrix.

%Generating extra features:
generator_names = {'SSA', 'NW', 'Cubic', 'Conv'};
generator_handles = {@SsaGenerator, @NwGenerator, @CubicGenerator, @ConvGenerator};

workStructTS = GenerateFeatures(workStructTS, generator_handles);
plot_generated_feature_matrix(workStructTS.matrix, generator_names);
%

model = struct('handle', handleModel, 'name', nameModel, 'params', [], ...
    'error', [], 'unopt_flag', true, 'forecasted_y', []);

for i = 1:nModels
    disp(['Fitting model: ', nameModel{i}])
    [~, model(i), real_y] = ComputeForecastingErrors(workStructTS, K, alpha_coeff, model(i));
end

% plot 1:24 forecasts of real_y if the error does not exceed 1e3
plot_forecasting_results(real_y, model, 1:24, 1e3);

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
table(MAPE_target, MAPE_full, AIC, 'RowNames', nameModel)


