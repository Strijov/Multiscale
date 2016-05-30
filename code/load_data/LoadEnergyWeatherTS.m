function tsSheaf = LoadEnergyWeatherTS(dirname)
% Loads a set of time series and returns the cell array of ts structres.
%
% The structure ts has the fields: % FIXIT please move this help where this
% structure is defined and put the reference in.
%
% Iput:
% dirname [string] named of folder with the loaded data. 
% Output:
% tsSheaf [struct]

folders = strsplit(fullfile(dirname), filesep);
folder_name = folders{end};

readme = struct('orig', 'Original time energy-weather series',...
                'missing_value', 'Energy-weather time series with artificially inserted missing values',...
                'varying_rates', 'Energy-weather time series with varying sampling rate');
readme = readme.(folder_name);

[filename_train, filename_test, filename_weather] = read_missing_value_dir(dirname);    

tsSheaf = cell(1, 2*numel(filename_train));
for i = 1:numel(filename_train)
    [train_ts, test_ts, time_train, time_test, train_weather, test_weather] = load_train_test_weather(filename_train{i},...
                                             filename_test{i},...
                                             filename_weather{i});
                                         
                                    
    tsSheaf{2*i-1} = make_struct(train_ts, train_weather, time_train, ...
                                    filename_train{i}, folder_name, readme);
    tsSheaf{2*i} = make_struct(test_ts, test_weather, time_test, ...
                                    filename_test{i}, folder_name, readme);
    
end


end

function ts_struct = make_struct(target_ts, weather_data, time, filename, folder_name, readme)


ts = [target_ts, num2cell(weather_data, 1)];
    self_deltaTp = num2cell([24, ones(1, size(weather_data, 2))]*6); % ???
    self_deltaTr = num2cell([24, ones(1, size(weather_data, 2))]);

    [~, fname, ~] = fileparts(filename);
    fname = strcat(folder_name, '_', fname);
    fname = regexprep(fname, '\.', '_');
    
    weather_time = time(1:24:numel(target_ts));
    time = {time, weather_time, weather_time, weather_time, weather_time, weather_time, weather_time};
    
    ts_struct = struct('x', ts, 'time', time, ...
                        'nsamples', self_deltaTr,...
                        'legend', {'target', 'Max Temperature', 'Min Temperature', 'Precipitation', 'Wind', 'Relative Humidity', 'Solar'},...
                        'deltaTp', self_deltaTp, 'deltaTr', self_deltaTr,...
                        'name', fname, 'readme', readme,...
                        'dataset', 'EnergyWeather');
    %save(fullfile('data','ProcessedData', [ts_struct(1).name, '.mat']), 'ts_struct')
end



function [train_ts, test_ts, time_train, time_test, train_weather, test_weather] = load_train_test_weather(train, test, weather)
[~, ~, extension] = fileparts(train);

time_train = []; 
time_test = [];
if strcmp(extension, '.csv')
    [test_ts, time_test] = ProcessCSVOutput(test);
    [train_ts, time_train] = ProcessCSVOutput(train);
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
% add hourly time stamps:
time_stamps = add_hours_to_dates(ts(:, 1), 1:24);
other = ts(:, 2:3);
ts = ts(:, 4:end);
ts = reshape(ts, numel(ts), 1);

if numel(ts) ~= 24*1096
    disp([filename, ': Data size: ', num2str(numel(ts)),...
        ' does not match the expected size 26304 = 24x1096']);
    
end

end

function date_vec = add_hours_to_dates(yyyymmdd_vec, hours)

n_dates = numel(yyyymmdd_vec);

% duplicate time stamps to insert hourly recods:
yyyymmdd_vec = repmat(yyyymmdd_vec', numel(hours), 1);
yyyymmdd_vec = yyyymmdd_vec(:);

% convert to 'yyyymmdd' to serial date numbers:
date_serial = datenum(floor(yyyymmdd_vec/10000), ...
                      floor(mod(yyyymmdd_vec,10000)/100),...
                      floor(mod(yyyymmdd_vec,100))); 
                  
% Convert serials to numeric rows [year, month, day, hour, min, sec];                  
date_vec = datevec(date_serial); 

% add hours
date_vec(:, 4) = date_vec(:, 4) + repmat(hours(:), n_dates,1);

% back to serials:
date_vec = datenum(date_vec);

end


function [train_fns, test_fns, weather_fns] = read_missing_value_dir(dirname)

train_fns = extractfield(dir(fullfile(dirname, 'train*')), 'name');
test_fns = extractfield(dir(fullfile(dirname,'test*')), 'name');
weather_fns = extractfield(dir(fullfile(dirname, 'weatherdata*')), 'name');


end

