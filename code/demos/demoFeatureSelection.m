function demoFeatureSelection(StructTS, model, generators)

N_PREDICTIONS = 5;
TRAIN_TEST_VAL_RATIO = [0, 0.75, 0.25];

nGenerators = numel(generators);


% Feature selection models are defined later

% Init structure to generate report:
report_struct = struct('handles', [], 'algos', [], 'headers', [],...
                 'res',  []); 
report_struct.handles = {@include_subfigs, @vertical_res_table};   
report_struct.algos = {model().name};
report_struct.headers = {'MAPE test', 'MAPE train'};
report_struct.res = cell(1, nGenerators + 4); % + no features, all features and PCA
figs = struct('names', cell(1, 2), 'captions', cell(1, 2));

FOLDER = fullfile('fig/feature_selection/');
% If necessary, create dir
if ~exist(FOLDER, 'dir')
    mkdir(FOLDER);
end
if ~exist(fullfile(FOLDER, StructTS(1).dataset), 'dir')
    mkdir(fullfile(FOLDER, StructTS(1).dataset));
end


%--------------------------------------------------------------------------
% Add regression matrix to the main structure:
StructTS = CreateRegMatrix(StructTS);

% Plot time series and a range of segments to forecast
[fname, caption] = plot_ts(StructTS, FOLDER);
figs(1).names = fname;
figs(1).captions = caption;

% Split data into train and test:
[idxTrain, ~, idxTest] = MultipleSplit(size(StructTS.X, 1), size(StructTS.X, 1), ...
                                        TRAIN_TEST_VAL_RATIO);
                                            
%
%--------------------------------------------------------------------------
% First, try with original features
[testMeanRes, trainMeanRes, testStdRes, trainStdRes, model] = calcErrorsByModel(StructTS, model, idxTrain, idxTest);
testMAPE = mean(reshape([model().testError], [], numel(model)), 1);
trainMAPE = mean(reshape([model().trainError], [], numel(model)), 1);

disp('Results with original features:')
disp([testMAPE', trainMAPE'])  
    
% plot frc results:
[fname, caption, ~, ~] = plot_forecasting_results(StructTS, model, N_PREDICTIONS, 10, ...
                                                   FOLDER, '');
figs(2).names = fname;
figs(2).captions = caption;
report_struct.res{1} = struct('data', 'History', 'errors', [testMAPE', trainMAPE']);
report_struct.res{1}.figs = figs;
                                          
%}

%--------------------------------------------------------------------------
% Next, try transformations by one:
for n_gen = 1:nGenerators
    generators(n_gen).replace = false;
    
    newStructTS = GenerateFeatures(StructTS, generators(n_gen), idxTrain, idxTest);
    [gen_fname, gen_caption] = plot_generated_feature_matrix(newStructTS, ...
                                {generators(n_gen).name}, ...
                                FOLDER, ['_fs_',generators(n_gen).name]);
    
    generators(n_gen).replace = true;
    model.transform = [];
    [~,~,~,~, model] = calcErrorsByModel(newStructTS, model, idxTrain, idxTest);
    testMAPE = mean(reshape([model().testError], [], numel(model)), 1);
    trainMAPE = mean(reshape([model().trainError], [], numel(model)), 1);
    disp(['Results with ', generators(n_gen).name])
    disp([testMAPE', trainMAPE'])  
    
    [fname, caption, ~, ~] = plot_forecasting_results(newStructTS, model, N_PREDICTIONS, 10,...
                                                       FOLDER, ['_fs_',generators(n_gen).name]);
    figs = struct('names', cell(1), 'captions', cell(1));
    figs(1).names = {gen_fname, fname};
    figs(1).captions = {gen_caption, caption};
    report_struct.res{1 + n_gen} = struct('data', generators(n_gen).name, 'errors', [testMAPE', trainMAPE']);
    report_struct.res{1 + n_gen}.figs = figs;                               
    %}
end
%}
%--------------------------------------------------------------------------
% Then for all features ...
    
StructTS = GenerateFeatures(StructTS, generators, idxTrain, idxTest);
[gen_fname, gen_caption] = plot_generated_feature_matrix(StructTS, ...
                            {generators().name}, ...
                            FOLDER, '_fs_all');

model.transform = [];
[~,~,~,~, model] = calcErrorsByModel(StructTS, model, idxTrain, idxTest);
testMAPE = mean(reshape([model().testError], [], numel(model)), 1);
trainMAPE = mean(reshape([model().trainError], [], numel(model)), 1);

disp('Results with all generators:')
disp([testMAPE', trainMAPE'])  

[fname, caption, ~, ~] = plot_forecasting_results(StructTS, model, N_PREDICTIONS, ...
                                                   10, FOLDER, '_fs_all');
figs = struct('names', cell(1), 'captions', cell(1));
figs(1).names = {gen_fname, fname};
figs(1).captions = {gen_caption, caption};
report_struct.res{2 + nGenerators} = struct('data', 'All', 'errors', [testMAPE', trainMAPE']);
report_struct.res{2 + nGenerators}.figs = figs;


%--------------------------------------------------------------------------
% Finally, apply feature selection:
% PCA:
pars = struct('maxComps', 50, 'expVar', 90, 'plot', @plot_pca_results);
feature_selection_mdl = struct('handle', @DimReducePCA, 'params', pars);

  
[pcaStructTS, feature_selection_mdl] = FeatureSelection(StructTS, feature_selection_mdl, ...
                                                            idxTrain, idxTest);
[fs_fname, fs_caption] = feval(feature_selection_mdl.params.plot, feature_selection_mdl.res, ...
                                                            {generators().name},...
                                                            pcaStructTS, ...
                                                            FOLDER, '_fs_pca');
figs = struct('names', cell(1, 2), 'captions', cell(1, 2));
figs(1).names = fs_fname;
figs(1).captions = fs_caption;                                                        

model.transform = [];
[~,~,~,~, model] = calcErrorsByModel(StructTS, model, idxTrain, idxTest);
testMAPE = mean(reshape([model().testError], [], numel(model)), 1);
trainMAPE = mean(reshape([model().trainError], [], numel(model)), 1);

disp('Results with PCA applied to all generators:')
disp([testMAPE', trainMAPE'])  
% plot frc results:
[~, ~, fname, caption] = plot_forecasting_results(StructTS, model, N_PREDICTIONS, 10, ...
                                                        FOLDER, '_fs_pca');
figs(2).names = fname;
figs(2).captions = caption;
report_struct.res{3 + nGenerators} = struct('data', 'PCA', 'errors', ...
                                            [testMAPE', trainMAPE']);
report_struct.res{3 + nGenerators}.figs = figs;
%--------------------------------------------------------------------------
% nonlinear PCA:
pars = struct('maxComps', 50, 'expVar', 90, 'plot', @plot_pca_results);
feature_selection_mdl = struct('handle', @DimReducePCA, 'params', pars);


[StructTS, feature_selection_mdl] = FeatureSelection(StructTS, feature_selection_mdl, ...
                                                            idxTrain, idxTest);
[fs_fname, fs_caption] = feval(feature_selection_mdl.params.plot, feature_selection_mdl.res, ...
                                                            {generators().name},...
                                                            StructTS, ...
                                                            FOLDER, '_fs_pca');
figs = struct('names', cell(1, 2), 'captions', cell(1, 2));
figs(1).names = fs_fname;
figs(1).captions = fs_caption;                                                        

model.transform = [];
[~,~,~,~, model] = calcErrorsByModel(StructTS, model, idxTrain, idxTest);
testMAPE = mean(reshape([model().testError], [], numel(model)), 1);
trainMAPE = mean(reshape([model().trainError], [], numel(model)), 1);

disp('Results with NPCA applied to all generators:')
disp([testMAPE', trainMAPE'])  
% plot frc results:
[~, ~, fname, caption] = plot_forecasting_results(StructTS, model, N_PREDICTIONS, 10, ...
                                                        FOLDER, '_fs_npca');
figs(2).names = fname;
figs(2).captions = caption;
report_struct.res{4 + nGenerators} = struct('data', 'NPCA', 'errors', ...
                                            [testMAPE', trainMAPE']);
report_struct.res{4 + nGenerators}.figs = figs;

%--------------------------------------------------------------------------
% save results and generate report:
save(['report_struct_fs_', StructTS.name ,'.mat'], 'report_struct');
generate_tex_report(report_struct, 'FeatureSelection.tex');


end
