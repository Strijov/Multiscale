function ts = LoadTimeSeries(dirnames)
% Data loader
%
% Input: dirnames - cell array of directory names
% Output: a cell array of ts structures

if nargin < 1
    dirnames = {'EnergyWeatherTS/orig/', 'EnergyWeatherTS/missing_value/', ...
        'EnergyWeatherTS/missing_value/'};
end

ts = {};
for dirname = dirnames
    dirname = fullfile('data', dirname{1});
    folders = strsplit(dirname, filesep);
    parent_dir = folders{2};
    loader_func_handle = str2func(['Load', parent_dir]);
    ts = [ts, feval(loader_func_handle, dirname)];
end


end