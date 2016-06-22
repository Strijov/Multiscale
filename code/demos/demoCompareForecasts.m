function model = demoCompareForecasts(tsStructArray, model, generators, feature_selection_mdl)
% Script demoCompareForecasts runs one forecasting experiment.
% It applies several competitive models to single dataset. 


% Check arguments:
if nargin < 2 || isempty(generators)
generators = struct('handle', @IdentityGenerator, 'name', 'Identity', ...
                                          'replace', true, 'transform', []);
end
if nargin < 3 || isempty(feature_selection_mdl)
feature_selection_mdl = struct('handle', @IdentityGenerator, 'params', []);    
end

% Experiment settings. 
trainTestRatio = [0.75, 0.25];

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

testMAPE = zeros(numDataSets, numel(model));
trainMAPE = zeros(numDataSets, numel(model)); 

for nDataSet = 1:numDataSets    
ts = tsStructArray{nDataSet};

% Add regression matrix to the main structure:
ts = CreateRegMatrix(ts);    

% Plot time series and a range of segments to forecast
[fname, caption] = plot_ts(ts);
figs(1).names = fname;
figs(1).captions = caption;


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
                                                        
                                                        
% Reinit models:
[model.transform] = deal(reset_transform{:});
[~, ~, ~, ~, model] = calcErrorsByModel(ts, model, idxTrain, idxTest);
testMAPE(nDataSet, :) = mean(reshape([model().testError], [], numel(model)), 1);
trainMAPE(nDataSet, :) = mean(reshape([model().trainError], [], numel(model)), 1);

% plot idx_target forecasts of real_y if the error does not exceed 1e3
[fname, caption, fname_by_models, caption_by_models] = plot_forecasting_results(ts, model, 5, 10);
figs(3).names = fname;
figs(3).captions = caption;
figs(4).names = fname_by_models;
figs(4).captions = caption_by_models;

report_struct.res{nDataSet} = struct('data', ts.name, 'errors', [testMAPE', trainMAPE']);
report_struct.res{nDataSet}.figs = figs;

end
%save('MAPE_EW.mat', 'testMAPE', 'trainMAPE');
%save('report_struct_EW.mat', 'report_struct');
%generate_tex_report(report_struct, 'CompareModels_EW.tex');

end

