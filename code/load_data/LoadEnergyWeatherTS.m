function tsSheaf = LoadEnergyWeatherTS(dirname)


folders = strsplit(fullfile(dirname), filesep);
folder_name = folders(end-1);

%addpath(genpath(cd));


[filename_train, filename_test, filename_weather] = read_missing_value_dir(dirname);    


tsSheaf = num2cell(ones(1, numel(filename_train)));
for i = 1:numel(filename_train)
    [target_ts, weather_data] = load_train_test_weather(filename_train{i},...
                                             filename_test{i},...
                                             filename_weather{i});
                                         
                                    
    ts = [target_ts, num2cell(weather_data, 1)];
    time_step = [ {1} , num2cell(ones(1, size(weather_data, 2))*24) ];
    self_deltaTp = num2cell([24, ones(1, size(weather_data, 2))]*6); % ???
    self_deltaTr = num2cell([24, ones(1, size(weather_data, 2))]);

    ts_length = 155; %size(target_ts, 1);
    tmp1 = linspace(size(weather_data, 1), numel(ts{2}) - 7 * ts_length, ts_length+1);
    tmp2 = linspace(numel(target_ts), numel(target_ts) - 24 * 7 * ts_length, ts_length+1);


    [~, fname, ~] = fileparts(filename_train{i});
    fname = strcat(folder_name, '_', fname);
    fname = regexprep(fname, '\.', '_');
    
    time_points = {tmp2,tmp1,tmp1,tmp1,tmp1,tmp1, tmp1};
    tsSheaf{i} = struct('x', ts, 'time_step', time_step, 'legend', legend, ...
                        'deltaTp', self_deltaTp, 'deltaTr', self_deltaTr,...
                        'time_points', time_points, 'normalization', [], ...
                        'name', fname);
end


end


function [target_ts, weather_data] = load_train_test_weather(train, test, weather)

%filename_train = fullfile('orig', 'SL2.xls');
%filename_test = fullfile('orig', 'PL.xls');
[~, ~, extension] = fileparts(train);
if strcmp(extension, '.csv')
    target_ts1 = ProcessCSVOutput(test);
    target_ts2 = ProcessCSVOutput(train);
    weather_data = csvread(weather, 1, 1);
else
    target_ts1 = ProcessXLSOutput(test);
    target_ts2 = ProcessXLSOutput(train);
    weather_data = xlsread(weather, 'weatherdata', 'E2:J1093','basic');
    weather_data(:, 1:4) = [];
end

target_ts = [target_ts1; target_ts2];

end

function [ts, time_stamps, other] = ProcessCSVOutput(filename)

ts = csvread(filename, 1, 0);
time_stamps = ts(:, 1);
other = ts(:, 2:3);
ts = ts(:, 4:end);
ts = reshape(ts, numel(ts), 1);

end

function [ts, time_stamps, other] = ProcessXLSOutput(filename)

[num, txt] = xlsread(filename,'', '','basic');

txt = txt(end-1095:end, :);
idx_floats = ~cellfun('isempty', txt);
ts = num;

try
ts(idx_floats) = cellfun(@str2num, txt(idx_floats));
catch
   new_cell_mat = cellfun(@str2num, txt(end-1095:end, 4:end), 'un', 0);
   idx_fail = cellfun('isempty', new_cell_mat);
   new_cell_mat(idx_fail) = {NaN};
   ts = cell2mat(new_cell_mat);
end
time_stamps = ts(:, 1);
other = ts(:, 2:3);
ts = ts(:, 4:end);
ts = reshape(ts, numel(ts), 1);

end


function [train_fns, test_fns, weather_fns] = read_missing_value_dir(dirname)

train_fns = extractfield(dir(fullfile(dirname, 'train*')), 'name');
test_fns = extractfield(dir(fullfile(dirname,'test*')), 'name');
weather_fns = extractfield(dir(fullfile(dirname, 'weatherdata*')), 'name');


end

