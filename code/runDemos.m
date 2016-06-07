% This script will be used to run various experiments stored in 'demos/'

addpath(genpath(cd));

% Dataset options: 'HascData', 'EnergyWeather', 'NNcompetition'
DATASET = 'HascData';

% All .mat data is stored in data/ProcessedData/ directory;
DATADIR = fullfile('data', 'ProcessedData');

% Check if data dir exists and is not empty 
if ~exist(DATADIR, 'dir') || isempty(dir(fullfile(DATADIR, '*.mat')))
    % Otherwise, load all data from 'data/' and save it 'data/ProcessedData/'
    LoadAndSave();
end

% LoadTimeSeries returns a cell array of ts structure arrays
tsStructArray  = LoadTimeSeries(DATASET);
ts = tsStructArray{1}; % FIXIT 

% Models
nameModel = {'VAR', 'SVR', 'Random Forest', 'Neural network'};   % Set of models. 
handleModel = {@VarForecast, @SVRMethod, @TreeBaggerForecast, @NnForecast};
model = struct('handle', handleModel, 'name', nameModel, 'params', [], 'transform', [],...
    'trainError', [], 'testError', [], 'unopt_flag', true, 'forecasted_y', []);



demoForecastAnalysis(ts, model(1));
demoCompareForeasts(tsStructArray);
demoFeatureSelection(ts);
%demoForecastHorizon(ts);