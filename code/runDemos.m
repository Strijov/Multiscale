% This script will be used to run various experiments stored in 'demos/'

addpath(genpath(cd));

% Dataset options: 'HascData', 'EnergyWeather', 'NNcompetition'
DATASET = 'EnergyWeather';
NAME_PATTERN = 'orig*'; % set to \w* to get all names

% All .mat data is stored in data/ProcessedData/ directory;
DATADIR = fullfile('data', 'ProcessedData');

% Check if data dir exists and is not empty 
if ~exist(DATADIR, 'dir') || isempty(dir(fullfile(DATADIR, '*.mat')))
    % Otherwise, load all data from 'data/' and save it 'data/ProcessedData/'
    LoadAndSave();
end

% LoadTimeSeries returns a cell array of ts structure arrays
tsStructArray  = LoadTimeSeries(DATASET, NAME_PATTERN);
ts = tsStructArray{1}; % FIXIT 

% Models
nameModel = {'VAR', 'SVR', 'Random Forest', 'Neural network'};   % Set of models. 
handleModel = {@VarForecast, @SVRMethod, @TreeBaggerForecast, @NnForecast};
model = struct('handle', handleModel, 'name', nameModel, 'params', [], 'transform', [],...
    'trainError', [], 'testError', [], 'unopt_flag', true, 'forecasted_y', []);


%Generating extra features:
generator_names = {'SSA', 'Cubic', 'Conv', 'NW'}; %{'Identity'};
generator_handles = {@SsaGenerator, @CubicGenerator, @ConvGenerator, @NwGenerator}; %{@IdentityGenerator};
generators = struct('handle', generator_handles, 'name', generator_names, ...
                                             'replace', false, 'transform', []);
generators(4).replace = true; % NW applies smoothing to the original data

% Feature selection:
pars = struct('maxComps', 50, 'expVar', 90, 'plot', @plot_pca_results);
feature_selection_mdl = struct('handle', @DimReducePCA, 'params', pars);



for i = 1:numel(model)
demoForecastAnalysis(tsStructArray, model(i), generators, feature_selection_mdl);
end


demoCompareForecasts(tsStructArray, model, generators, feature_selection_mdl);
demoFeatureSelection(ts);

for i = 1:numel(model)
demoForecastHorizon(ts, model(i));
end



%