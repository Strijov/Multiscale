function ts_struct = LoadNNcompetition(dirname)

fname = 'NN3_final_data.csv';
meta_fname = 'NN3_final_meta.csv';

readme = 'Empirical business monthly time series. Data from the 2006/07 Forecasting Competition for Neural Networks';

% read data as a Table: 
data_table = readtable(fname);   
headers = data_table.Properties.VariableNames;
ts_struct = cell(1, numel(headers));
for i =1:numel(headers)
    name = headers{i};
    ts = data_table.(name);
    ts = ts(~isnan(ts)); % some time series are shorter than others;
    % arbitrarily set deltaTp = 12 (a year) and deltaTr = 6
    deltaTp = 12;
    deltaTr = 6;
    time_points = numel(ts):-1:deltaTr+deltaTp;
    ts_struct{i} = struct('x', ts, 'time_step', 1, ...
                        'deltaTp', deltaTp, 'deltaTr', deltaTr,...
                        'time_points', time_points, 'normalization', [], ...
                        'name', name, 'readme', readme, ...
                        'plot_handle', @plot_ts, 'dataset', 'NNcompetition');
    
    
end

end