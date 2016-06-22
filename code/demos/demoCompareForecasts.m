function [testError, trainError, bias, model] = demoCompareForecasts(tsStructArray, model, generators, feature_selection_mdl, verbose)
% Script demoCompareForecasts runs one forecasting experiment.
% It applies several competitive models to single dataset. 


% Check arguments:
if nargin < 3 || isempty(generators)
generators = struct('handle', @IdentityGenerator, 'name', 'Identity', ...
                                          'replace', true, 'transform', []);
end
if nargin < 4 || isempty(feature_selection_mdl)
feature_selection_mdl = struct('handle', @IdentityGenerator, 'params', []);    
end
if nargin < 5
    verbose = true;
end

% Experiment settings. 
trainTestRatio = [0.75, 0.25];
N_PREDICTIONS = 5; % plotting pars
MAX_ERROR = 5; % plotting pars

% Shotcuts
nameModel = {model().name};
generator_names = {generators().name};
nModels = numel(model);
reset_transform = cell(1, nModels);

% Load and prepare dataset.
numDataSets = numel(tsStructArray);

report_struct = struct('handles', [], 'algos', [], 'headers', [],...
                 'res',  []); 
report_struct.handles = {@include_subfigs, @vertical_res_table};   
report_struct.algos = nameModel;
report_struct.headers = {'MAPE test', 'MAPE train'};
report_struct.res = cell(1, numDataSets);
figs = struct('names', cell(1,4), 'captions', cell(1,4));

FOLDER = fullfile('fig', tsStructArray{1}(1).dataset);
% If necessary, create dir
if ~exist(FOLDER, 'dir')
    mkdir(FOLDER);
end

testError = zeros(numDataSets, nModels);
trainError = zeros(numDataSets, nModels); 
bias = zeros(numDataSets, nModels); 


for nDataSet = 1:numDataSets    
ts = tsStructArray{nDataSet};

% Add regression matrix to the main structure:
ts = CreateRegMatrix(ts);    

% Plot time series and a range of segments to forecast
if verbose
[fname, caption] = plot_ts(ts);
figs(1).names = fname;
figs(1).captions = caption;
end

[idxTrain, idxTest, idxVal] = TrainTestSplit(size(ts.Y, 1), trainTestRatio);
idxTest = [idxVal, idxTest];

% Generate additional features:
ts = GenerateFeatures(ts, generators, idxTrain, idxTest);
disp(['Generation finished. Total number of features: ', num2str(size(ts.X, 2))]);
[gen_fname, gen_caption] = plot_generated_feature_matrix(ts, ...
                                                         generator_names);

% Select or transform features:
[ts, feature_selection_mdl] = FeatureSelection(ts, feature_selection_mdl,...
                                                        idxTrain, idxTest);
if verbose  
    if isfield(feature_selection_mdl.params, 'plot')
    [fs_fname, fs_caption] = feval(feature_selection_mdl.params.plot, ...
                                    feature_selection_mdl.res, ...
                                    generator_names, ts); 
    else
        fs_fname = '';
        fs_caption = '';
    end
figs(2).names = [gen_fname, fs_fname];
figs(2).captions = [gen_caption, fs_caption];
end                                                        
                                                        
% Reinit models:
[model.transform] = deal(reset_transform{:});
[~, ~, ~, ~, model] = calcErrorsByModel(ts, model, idxTrain, idxTest);
testError(nDataSet, :) = mean(reshape([model().testError], [], nModels), 1);
trainError(nDataSet, :) = mean(reshape([model().trainError], [], nModels), 1);
bias(nDataSet, :) = mean(reshape([model().bias], [], nModels), 1);

% plot N_PREDICTIONS forecasts of real_y if the error does not exceed
% MAX_ERROR
if verbose
[fname, caption, fname_by_models, caption_by_models] = plot_forecasting_results(...
                                            ts, model, N_PREDICTIONS, MAX_ERROR);
figs(3).names = fname;
figs(3).captions = caption;
figs(4).names = fname_by_models;
figs(4).captions = caption_by_models;

report_struct.res{nDataSet} = struct('data', ts.name, 'errors', [testError', trainError']);
report_struct.res{nDataSet}.figs = figs;
end

end

% Generating report:
if verbose
save(['Errors_', ts.dataset,'.mat'], 'testError', 'trainError');   
save(['report_struct_', ts.dataset,'.mat'], 'report_struct');
generate_tex_report(report_struct, ['CompareModels_', ts.dataset,'.tex']);
end


end

