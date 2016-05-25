function tsSheaf = LoadEnergyWeatherTS(dirname)
% Loads a set of time series and returns the cell array of ts structres.
%
% The structure ts has the fields: % FIXIT please move this help where this
% structure is defined and put the reference in.
%
% Describes time series 
% t [T,1] Time in milliseconds since 1/1/1970 (UNIX format)
% x [T, N] Columns of the matrix are time series; missing values are NaNs
% legend {1, N} Time series descriptions ts.x, e.g. ts.legend={?Consumption, ?Price?, ?Temperature?};
% readme [string] Data information (source, formation time etc.)
% type [1,N] (optional) Time series types ts.x, 1-real-valued, 2-binary, k ? k-valued
% timegen [T,1]=func_timegen(timetick) (optional) Time ticks generator, may
% contain the start (or end) time in UNIX format and a function to generate the vector t of the size [T,1]
%
% Iput:
% dirname [string] named of folder with the loaded data. 
% Output:
% tsSheaf [struct]

folders = strsplit(fullfile(dirname), filesep);
folder_name = folders{end-1};

readme = struct('orig', 'Original time energy-weather series',...
                'missing_value', 'Energy-weather time series with artificially inserted missing values',...
                'varying_rate', 'Energy-weather time series with varying sampling rate');
readme = readme.(folder_name);

[filename_train, filename_test, filename_weather] = read_missing_value_dir(dirname);    


tsSheaf = cell(1, 2*numel(filename_train));
for i = 1:numel(filename_train)
    [train_ts, test_ts, train_weather, test_weather] = load_train_test_weather(filename_train{i},...
                                             filename_test{i},...
                                             filename_weather{i});
                                         
                                    
    tsSheaf{2*i-1} = make_ts_struct(train_ts, train_weather, filename_train{i}, folder_name, readme);
    tsSheaf{2*i} = make_ts_struct(test_ts, test_weather, filename_test{i}, folder_name, readme);
    
end


end

function tsSheaf = make_ts_struct(target_ts, weather_data, filename, folder_name, readme)


ts = [target_ts, num2cell(weather_data, 1)];
    time_step = [ {1} , num2cell(ones(1, size(weather_data, 2))*24) ];
    self_deltaTp = num2cell([24, ones(1, size(weather_data, 2))]*6); % ???
    self_deltaTr = num2cell([24, ones(1, size(weather_data, 2))]);

    ts_length = 155; %size(target_ts, 1);
    tmp1 = linspace(size(weather_data, 1), numel(ts{2}) - 7 * ts_length, ts_length+1);
    tmp2 = linspace(numel(target_ts), numel(target_ts) - 24 * 7 * ts_length, ts_length+1);


    [~, fname, ~] = fileparts(filename);
    fname = strcat(folder_name, '_', fname);
    fname = regexprep(fname, '\.', '_');
    
    time_points = {tmp2,tmp1,tmp1,tmp1,tmp1,tmp1, tmp1};
    tsSheaf = struct('x', ts, 'time_step', time_step, ...
                        'deltaTp', self_deltaTp, 'deltaTr', self_deltaTr,...
                        'time_points', time_points, 'normalization', [], ...
                        'name', fname, 'readme', readme);

end


function [train_ts, test_ts, train_weather, test_weather] = load_train_test_weather(train, test, weather)
[~, ~, extension] = fileparts(train);
if strcmp(extension, '.csv')
    test_ts = ProcessCSVOutput(test);
    train_ts = ProcessCSVOutput(train);
    weather_data = csvread(weather, 1, 1);
    if size(weather_data, 2) > 6
        weather_data(:, 7:end) = [];
    end
else
    test_ts = ProcessXLSOutput(test);
    train_ts = ProcessXLSOutput(train);
    weather_data = xlsread(weather, 'weatherdata', '','basic');
    weather_data(:, 1:4) = [];
end


if size(weather_data, 1) ~= 2192
    disp([weather, ': Data size: ', num2str(size(weather_data, 1)),...
        ' does not match the expected size 2192 = 2*1096']);    
end

train_weather = weather_data(1:1096,:);
test_weather = weather_data(1097:2192,:);

end

function [ts, time_stamps, other] = ProcessCSVOutput(filename)

ts = csvread(filename, 1, 0);
time_stamps = ts(:, 1);
other = ts(:, 2:3);
ts = ts(:, 4:end);
ts = reshape(ts, numel(ts), 1);

if numel(ts) ~= 24*1096
    disp([filename, ': Data size: ', num2str(numel(ts)),...
        ' does not match the expected size 26304 = 24x1096']);
    
end

end

function [ts, time_stamps, other] = ProcessXLSOutput(filename)

[num, txt] = xlsread(filename,'', '','basic');

txt = txt(end-1095:end, :);
idx_floats = ~cellfun('isempty', txt);
ts = num;

try
ts(idx_floats) = cellfun(@str2num, txt(idx_floats));
catch
   new_cell_mat = cellfun(@str2num, txt(end-1095:end, :), 'un', 0);
   idx_fail = cellfun('isempty', new_cell_mat);
   new_cell_mat(idx_fail) = {NaN};
   ts = cell2mat(new_cell_mat);
end
time_stamps = ts(:, 1);
other = ts(:, 2:3);
ts = ts(:, 4:end);
ts = reshape(ts, numel(ts), 1);

if numel(ts) ~= 24*1096
    disp([filename, ': Data size: ', num2str(numel(ts)),...
        ' does not match the expected size 26304 = 24x1096']);
    
end

end


function [train_fns, test_fns, weather_fns] = read_missing_value_dir(dirname)

train_fns = extractfield(dir(fullfile(dirname, 'train*')), 'name');
test_fns = extractfield(dir(fullfile(dirname,'test*')), 'name');
weather_fns = extractfield(dir(fullfile(dirname, 'weatherdata*')), 'name');


end

