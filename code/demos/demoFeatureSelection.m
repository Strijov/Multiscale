function demoFeatureSelection(StructTS)
% Models
nameModel = {'VAR', 'SVR', 'Random Forest', 'Neural network'};   % Set of models. 
handleModel = {@VarForecast, @SVRMethod, @TreeBaggerForecast, @NnForecast};

%Generating extra features:
generator_names = {'SSA', 'Cubic', 'Conv', 'NW'}; %{'Identity'};
generator_handles = {@SsaGenerator, @CubicGenerator, @ConvGenerator, @NwGenerator}; %{@IdentityGenerator};
generators = struct('handle', generator_handles, 'name', generator_names, ...
                                             'replace', true, 'transform', []);
nGenerators = numel(generators);


% Feature selection:
pars = struct('maxComps', 50, 'expVar', 90, 'plot', @plot_pca_results);
feature_selection_mdl = struct('handle', @DimReducePCA, 'params', pars);


% Init structure to generate report:
report_struct = struct('handles', [], 'algos', [], 'headers', [],...
                 'res',  []); 
report_struct.handles = {@include_subfigs, @vertical_res_table};   
report_struct.algos = nameModel;
report_struct.headers = {'MAPE test', 'MAPE train', 'AIC'};
report_struct.res = cell(1, nGenerators + 3); % + no features, all features and PCA
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
[idxTrain, ~, idxTest] = TrainTestSplit(size(StructTS.X, 1), 0);
                                            
%
%--------------------------------------------------------------------------
% First, try with original features
model = struct('handle', handleModel, 'name', nameModel, 'params', [], 'obj', [],...
    'trainError', [], 'testError', [], 'unopt_flag', true, 'forecasted_y', []);

[MAPE_test, MAPE_train, AIC, model] = calcErrorsByModel(StructTS, model, idxTrain, idxTest);
disp('Results with original features:')
table(MAPE_test, MAPE_train, AIC, 'RowNames', nameModel)

% plot frc results:
[fname, caption, ~, ~] = plot_forecasting_results(StructTS, model, 1:StructTS.deltaTr, 1e3, ...
                                                   FOLDER, '');
figs(2).names = fname;
figs(2).captions = caption;
report_struct.res{1} = struct('data', 'History', 'errors', [MAPE_test, MAPE_train, AIC]);
report_struct.res{1}.figs = figs;
                                          
%}

%--------------------------------------------------------------------------
% Next, try transformations by one:
for n_gen = 1:nGenerators
    model = struct('handle', handleModel, 'name', nameModel, 'params', [], 'obj', [],...
        'trainError', [], 'testError', [], 'unopt_flag', true, 'forecasted_y', []);
    
    newStructTS = GenerateFeatures(StructTS, generators(n_gen), idxTrain, idxTest);
    [gen_fname, gen_caption] = plot_generated_feature_matrix(newStructTS, ...
                                generator_names(n_gen), ...
                                FOLDER, ['_fs_',generators(n_gen).name]);
    
    
    
    [MAPE_test, MAPE_train, AIC, model] = calcErrorsByModel(newStructTS, model, ...
                                                            idxTrain, idxTest);
    disp(['Results with ', generator_names{n_gen}])
    table(MAPE_test, MAPE_train, AIC, 'RowNames', nameModel)  
    
    [fname, caption, ~, ~] = plot_forecasting_results(newStructTS, model, 1:newStructTS.deltaTr, 1e3,...
                                                       FOLDER, ['_fs_',generator_names{n_gen}]);
    figs = struct('names', cell(1), 'captions', cell(1));
    figs(1).names = {gen_fname, fname};
    figs(1).captions = {gen_caption, caption};
    report_struct.res{1 + n_gen} = struct('data', generator_names{n_gen}, 'errors', [MAPE_test, MAPE_train, AIC]);
    report_struct.res{1 + n_gen}.figs = figs;                               
    %}
end
%}
%--------------------------------------------------------------------------
% Then for all features ...
model = struct('handle', handleModel, 'name', nameModel, 'params', [], 'obj', [],...
        'trainError', [], 'testError', [], 'unopt_flag', true, 'forecasted_y', []);
    
StructTS = GenerateFeatures(StructTS, generators, idxTrain, idxTest);
[gen_fname, gen_caption] = plot_generated_feature_matrix(StructTS, ...
                            generator_names, ...
                            FOLDER, '_fs_all');
[MAPE_test, MAPE_train, AIC, model] = calcErrorsByModel(StructTS, model, ...
                                                        idxTrain, idxTest);
disp('Results with all generators:')
table(MAPE_test, MAPE_train, AIC, 'RowNames', nameModel)

[fname, caption, ~, ~] = plot_forecasting_results(StructTS, model, 1:StructTS.deltaTr, ...
                                                   1e3, FOLDER, '_fs_all');
figs = struct('names', cell(1), 'captions', cell(1));
figs(1).names = {gen_fname, fname};
figs(1).captions = {gen_caption, caption};
report_struct.res{2 + nGenerators} = struct('data', 'All', 'errors', [MAPE_test, MAPE_train, AIC]);
report_struct.res{2 + nGenerators}.figs = figs;


%--------------------------------------------------------------------------
% Finally, apply feature selection:
model = struct('handle', handleModel, 'name', nameModel, 'params', [], 'obj', [],...
        'trainError', [], 'testError', [], 'unopt_flag', true, 'forecasted_y', []);

    
[StructTS, feature_selection_mdl] = FeatureSelection(StructTS, feature_selection_mdl, ...
                                                            idxTrain, idxTest);
[fs_fname, fs_caption] = feval(feature_selection_mdl.params.plot, feature_selection_mdl.res, ...
                                                            generator_names,...
                                                            StructTS, ...
                                                            FOLDER, '_fs_pca');
figs = struct('names', cell(1, 2), 'captions', cell(1, 2));
figs(1).names = fs_fname;
figs(1).captions = fs_caption;                                                        
[MAPE_test, MAPE_train, AIC, model] = calcErrorsByModel(StructTS, model, ...
                                                        idxTrain, idxTest);
disp('Results with PCA applied to all generators:')
table(MAPE_test, MAPE_train, AIC, 'RowNames', nameModel)
% plot frc results:
[~, ~, fname, caption] = plot_forecasting_results(StructTS, model, 1:StructTS.deltaTr, 1e3, ...
                                                        FOLDER, '_fs_pca');
figs(2).names = fname;
figs(2).captions = caption;
report_struct.res{3 + nGenerators} = struct('data', 'PCA', 'errors', [MAPE_test, MAPE_train, AIC]);
report_struct.res{3 + nGenerators}.figs = figs;


%--------------------------------------------------------------------------
% save results and generate report:
save(['report_struct_fs_', StructTS.name ,'.mat'], 'report_struct');
generate_tex_report(report_struct, 'FeatureSelection.tex');


end
