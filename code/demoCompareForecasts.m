% Script demoCompareForecasts runs one forecasting experiment.
% It applies several competitive models to single dataet. 

% FIXIT Plese type DONE here after your changes.
addpath(genpath(cd));

% Feature selection:
pars = struct('maxComps', 50, 'expVar', 90, 'plot', @plot_pca_results);
feature_selection_mdl = struct('handle', @DimReducePCA, 'params', pars);


% Models
nameModel = {'VAR', 'SVR', 'Random Forest', 'Neural network'};   % Set of models. 
handleModel = {@VarForecast, @SVRMethod, @TreeBaggerForecast, @NnForecast};

% Experiment settings. 
alpha_coeff = 0; % FIXIT Please explain. 

%Generating extra features:
generator_names = {'SSA', 'Cubic', 'Conv', 'NW'}; %{'Identity'};
generator_handles = {@SsaGenerator, @NwGenerator, @CubicGenerator, @ConvGenerator}; %{@IdentityGenerator};
generators = struct('handle', generator_handles, 'name', generator_names, ...
                                         'replace', true, 'transform', []);


% Load and prepare dataset.
LoadAndSave('EnergyWeatherTS/orig');
ts_struct_array  = LoadTimeSeries('EnergyWeather');
numDataSets = numel(ts_struct_array);

report_struct = struct('handles', [], 'algos', [], 'headers', [],...
                 'res',  []); 
report_struct.handles = {@include_subfigs, @vertical_res_table};   
report_struct.algos = nameModel;
report_struct.headers = {'MAPE test', 'MAPE train', 'AIC'};
report_struct.res = cell(1, numDataSets);
figs = struct('names', cell(1,4), 'captions', cell(1,4));

FOLDER = fullfile('fig', ts_struct_array{1}(1).dataset);
% If necessary, create dir
if ~exist(FOLDER, 'dir')
    mkdir(FOLDER);
end


for nDataSet = 1:numDataSets    
StructTS = ts_struct_array{nDataSet};

% Add regression matrix to the main structure:
StructTS = CreateRegMatrix(StructTS);    

% Plot time series and a range of segments to forecast
[fname, caption] = plot_ts(StructTS);
figs(1).names = fname;
figs(1).captions = caption;


[idxTrain, ~, idxTest] = TrainTestSplit(size(StructTS.Y, 1), 0);

%
StructTS = GenerateFeatures(StructTS, generators, idxTrain, idxTest);
disp(['Generation finished. Total number of features: ', num2str(StructTS.deltaTp)]);
[gen_fname, gen_caption] = plot_generated_feature_matrix(StructTS, ...
                                                         generator_names);

[StructTS, feature_selection_mdl] = FeatureSelection(StructTS, feature_selection_mdl,...
                                                        idxTrain, idxTest);
[fs_fname, fs_caption] = feval(feature_selection_mdl.params.plot, feature_selection_mdl.res, ...
                                                            generator_names,...
                                                            StructTS); 
figs(2).names = [gen_fname, fs_fname];
figs(2).captions = [gen_caption, fs_caption];
                                                        
                                                        

model = struct('handle', handleModel, 'name', nameModel, 'params', [], 'transform', [],...
    'trainError', [], 'testError', [], 'unopt_flag', true, 'forecasted_y', []);

[MAPE_test, MAPE_train, AIC, model] = calcErrorsByModel(StructTS, model, ...
                                                        idxTrain, idxTest);


N_PREDICTIONS = 10;
idx_target = 1:min(StructTS.deltaTr*N_PREDICTIONS, numel(StructTS.x));

% plot idx_target forecasts of real_y if the error does not exceed 1e3
[fname, caption, fname_by_models, caption_by_models] = plot_forecasting_results(StructTS, model, 1:StructTS.deltaTr, 1e3);
figs(3).names = fname;
figs(3).captions = caption;
figs(4).names = fname_by_models;
figs(4).captions = caption_by_models;

report_struct.res{nDataSet} = struct('data', StructTS.name, 'errors', [MAPE_test, MAPE_train, AIC]);
report_struct.res{nDataSet}.figs = figs;

table(MAPE_test, MAPE_train, AIC, 'RowNames', nameModel)

end
save('report_struct_EW.mat', 'report_struct');
generate_tex_report(report_struct, 'CompareModels_EW.tex');

