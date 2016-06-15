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
pars = cell(1, numel(nameModel));
pars{1} = struct('regCoeff', 2);
pars{2} = struct('C', 10000, 'lambda', 0.00001, 'epsilon', 0.01);
pars{3} = struct('nTrees', 25, 'nVars', 48);
pars{4} = struct('nHiddenLayers', 25);
model = struct('handle', handleModel, 'name', nameModel, 'params', pars, 'transform', [],...
    'trainError', [], 'testError', [], 'unopt_flag', false, 'forecasted_y', []);


%Generating extra features:
generator_names = {'SSA', 'Cubic', 'Conv', 'NW'}; %{'Identity'};
generator_handles = {@SsaGenerator, @CubicGenerator, @ConvGenerator, @NwGenerator}; %{@IdentityGenerator};
generators = struct('handle', generator_handles, 'name', generator_names, ...
                                             'replace', false, 'transform', []);
generators(4).replace = true; % NW applies smoothing to the original data

% Feature selection:
pars = struct('maxComps', 50, 'expVar', 90, 'plot', @plot_pca_results);
feature_selection_mdl = struct('handle', @DimReducePCA, 'params', pars);

% Separately:
trainMAPE = zeros(numel(model) + 1, 1);
testMAPE = zeros(numel(model) + 1, 1);

for nTs = 1:numel(tsStructArray{1}) 
% Define baseline model:    
ts = {tsStructArray{1}(i), tsStructArray{2}(i)};    
pars = struct('deltaTr', ts{1}.deltaTr, 'deltaTp', ts{1}.deltaTp);
baselineModel = struct('handle', @MartingalForecast, 'name', 'Martingal', 'params', pars, 'transform', [],...
    'trainError', [], 'testError', [], 'unopt_flag', false, 'forecasted_y', []);
[testMAPE(1), trainMAPE(1)] = demoForecastAnalysis(ts, baselineModel, [], []);    

for i = 1:numel(model)
    [testMAPE(i + 1), trainMAPE(i + 1)] = demoForecastAnalysis(ts, model(i), generators, feature_selection_mdl);
end
end
disp([testMAPE, trainMAPE])

% Proposed framework:
for i = 1:numel(model)
[testMAPE(i), trainMAPE(i)] = demoForecastAnalysis(tsStructArray, model(i), generators, feature_selection_mdl);
end
disp([testMAPE, trainMAPE])

for i = 1:numel(tsStructArray)
demoFeatureSelection(tsStructArray{i}, model, generators);
end





demoCompareForecasts(tsStructArray, model, generators, feature_selection_mdl);

for i = 1:numel(model)
demoForecastHorizon(ts, model(i));
end



%