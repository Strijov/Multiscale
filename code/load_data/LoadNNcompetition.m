function ts_struct = LoadNNcompetition(~)

fname = 'NN3_final_data.csv';
meta_fname = 'NN3_final_meta.csv';

readme = 'Empirical business monthly time series. Data from the 2006/07 Forecasting Competition for Neural Networks';

% read data as a Table: 
data_table = readtable(fname); 
meta = readtable(meta_fname);
headers = data_table.Properties.VariableNames;
ts_struct = cell(1, numel(headers));

for i =1:numel(headers)
    name = headers{i};
    year = str2double(meta.(name){2});
    ts = data_table.(name);
    
    % some time series are shorter than others:
    ts = ts(~isnan(ts)); 
    
    % arbitrarily set deltaTp = 12 (a year)
    deltaTp = 12;
    deltaTr = 6;
    time = monthly_time_from_year(year, numel(ts));
    ts_struct{i} = struct('x', ts, 'time', time, ...
                        'legend', meta.(name){1}, ...
                        'deltaTp', deltaTp, 'deltaTr', deltaTr,...
                        'name', name, 'readme', readme, ...
                        'dataset', 'NNcompetition');
                    
    
    
end

end

function time = monthly_time_from_year(year, npoints)

% this function creates a list of "npoints" monthly serial date numbers, 
% starting from the "year"

date_vec = repmat([year, zeros(1, 5)], npoints, 1);
add_points = repmat(1:12, 1, ceil(npoints/12))';
date_vec(:, 2) = add_points(1:npoints);

time = datenum(date_vec);

end