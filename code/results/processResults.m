function processResults

% Aggregates results of feature selection demo
% 'results' are 6 x 10 cell arrays with columns correspondent to feature
% generation strategies:
% 1: baseline (tomorrow is the same as yesterday)
% 2: historical features
% 3--6: SSA, Conv, Cubic, Cenroids(separately) + historical
% 7: NW (smoothing of the original data) 
% 8-10: All, All+PCA, All+NPCA  
%
% The six rows correspond to:
% 1, 2: test/train mean residusls; 3-4: test/train std of residuals, 
% 5-6: mean test and train SMAPE
%
% Each cell is [nModels x nTs] matrix.


fnames = {'results_fs_orig_train.mat', 'results_fs_missing_value_0_01.mat', 'results_fs_missing_value_0_03.mat', ...
    'results_fs_missing_value_0_05.mat', ...
    'results_fs_missing_value_train_0_1.mat', ...
    'results_fs_varying_rates.mat'};
    

refnames = {'orig', '0.01', '0.03', '0.05', '0.1', 'varying'};
model_names = {'MLR', 'MSVR', 'RF', 'ANN'};
ts_names = {'Energy', 'Max T.', 'Min T.',... 
            'Precipitation', 'Wind', 'Humidity', 'Solar'};
features_names = {'History', 'SSA', 'Cubic', 'Conv', 'Centroids', 'NW', ...
                'All', 'PCA', 'NPCA'};
err_names = {'testRes', 'trainRes', 'testResStd', 'trainResStd', 'testSMAPE', 'trainSMAPE'};

IDX_ERROR = 1; % results for testSMAPE

    
rep_struct = struct('handles', [], 'headers', [], 'res', [], 'rows', []);
rep_struct.headers = ts_names;
rep_struct.handles = {@fs_vertical_res_table};%, @make_table};
nTs = 7;
best_res = zeros(6, nTs); % errors by ts
rel = zeros(size(best_res));
idx_improve = zeros(size(best_res));
rep_struct.res = cell(numel(fnames), 1);
names = cell(numel(fnames), nTs);
count_best_models = zeros(numel(model_names), numel(features_names));
for nFile = 1:numel(fnames)
   disp(fnames{nFile});
   load(fnames{nFile});    
   for i = 1:nTs
       [best_res(:, i), rel(:, i), best_names] = get_best_by_ts(results, i, ...
                                              model_names, features_names);                                  
        %disp([ts_names{i}, ' best results: '])
        %table(best_res(:, i), rel(:, i), names)
        names{nFile, i} = [best_names{IDX_ERROR, :}];
        count_best_models = add_counts(count_best_models, best_names, model_names, features_names);
   end
   idx_improve = idx_improve + 1*(rel < 1);
   rep_struct.res{nFile} = struct('name', refnames{nFile}, 'errors',  best_res(IDX_ERROR, :), 'row_names', []); %{'test MAPE'});
end


count_best_freqs(names, ts_names, err_names{IDX_ERROR});
rep_struct.table = struct('rows', [], 'columns', [], 'table', []);
rep_struct.table.table = names;
rep_struct.table.rows = refnames;
rep_struct.table.columns = ts_names;

plot_counts(count_best_models, nFile*nTs*numel(err_names), model_names, features_names, '_all');
plot_counts(idx_improve/nFile, 1, err_names, ts_names, '_vs_baseline');
generate_tex_report(rep_struct, 'fs_table.tex');

end


function [best, rel_best, names] = get_best_by_ts(results, nTs, mdl_names, feat_names)

    ts_results = cellfun(@(x) x(:, nTs), results, 'UniformOutput', 0);
    idx_best_mdl = cell2mat(cellfun(@(x) idx_min(abs(x)), ts_results(:, 2:end), 'UniformOutput', 0));% cell2mat();
    ts_best = cellfun(@(x) min(abs(x)), ts_results(:, 2:end), 'UniformOutput', 1);
    
    [best, idx_best_feat] = min(ts_best, [], 2);
    rel_best = best./abs(cell2mat(ts_results(:, 1)));
    idx_best_mdl = arrayfun(@(i) idx_best_mdl(i, idx_best_feat(i)), ...
                               1:numel(idx_best_feat), ...
                                'UniformOutput', 1);
    names = [mdl_names(idx_best_mdl); feat_names(idx_best_mdl)]';
    
end

function count_best_mdl = add_counts(count_best_mdl, best_names, model_names, features_names)

for i = 1:size(best_names, 1)
    idx_row = ismember(model_names, best_names{i, 1});
    idx_col = ismember(features_names, best_names{i, 2});
    count_best_mdl(idx_row, idx_col) = count_best_mdl(idx_row, idx_col) + 1;
end


end

function plot_counts(matrix, checksum, model_names, features_names, err_name, name)

if ~(sum(matrix(:)) == checksum)
    disp('Number of nonzero elements is inconsistent')
    
end

rgb = [ ...
    94    79   162
    50   136   189
   102   194   165
   171   221   164
   230   245   152
   255   255   191
   254   224   139
   253   174    97
   244   109    67
   213    62    79
   158     1    66  ] / 255;

h = figure;
%colormap(rgb);
imagesc(matrix/checksum);
colorbar;
set(gca, 'FontSize', 16, 'FontName', 'Times')
ax = gca;
ax.XTickLabelRotation = 45;
set(gca, 'YTick', 1:numel(model_names), 'YtickLabel', model_names)
set(gca, 'xtick', 1:size(features_names, 2), 'xticklabel', features_names)
saveas(h, 'best_models',name,'_colormatrix.eps', 'epsc');
close(h);

end


function freqs = count_best_freqs(names, ts_names, err_name)

uniq_names = unique(names);
freqs_by_ts = zeros(numel(uniq_names), size(names, 2));
for i = 1:numel(uniq_names)
    freqs_by_ts(i, :) = mean(ismember(names, uniq_names{i}));
end


rgb = [ ...
    94    79   162
    50   136   189
   102   194   165
   171   221   164
   230   245   152
   255   255   191
   254   224   139
   253   174    97
   244   109    67
   213    62    79
   158     1    66  ] / 255;

h = figure;
colormap(rgb);
imagesc(freqs_by_ts);
colorbar;
set(gca, 'FontSize', 16, 'FontName', 'Times')
ax = gca;
ax.XTickLabelRotation = 45;
set(gca, 'YTick', 1:numel(uniq_names), 'YtickLabel', uniq_names)
set(gca, 'xtick', 1:size(names, 2), 'xticklabel', ts_names)
saveas(h, ['best_models_', err_name, '_colormatrix.eps'], 'epsc');
close(h);

freqs = mean(freqs_by_ts, 2);
[~, idx] = sort(freqs);
h = figure;
bar(freqs(idx));
set(gca, 'FontSize', 16, 'FontName', 'Times')
set(gca, 'xtick', 1:numel(uniq_names), 'xticklabel', uniq_names(idx))
ax = gca;
ax.XTickLabelRotation = 45;
saveas(h, ['best_models_', err_name, '_bar.eps'], 'epsc');
close(h);


end

function i  = idx_min(x)
[~, i] = min(x);
end