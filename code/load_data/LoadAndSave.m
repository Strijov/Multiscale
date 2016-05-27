function LoadAndSave(dirnames)
% Data loader
%
% Input: dirnames - cell array, contains names of folders inside data/
% directory, where the desired datasets are stored. Each folder 'DataName' inside
% data/ directory has a corresponding loader 'LoadDataName'.
%
% Output: a cell array of ts structures
% Description of the time series structure: 
%   t [T,1]  - Time in milliseconds since 1/1/1970 (UNIX format)
%   x [T, N] - Columns of the matrix are time series; missing values are NaNs
%   legend {1, N}  - Time series descriptions ts.x, e.g. ts.legend={?Consumption, ?Price?, ?Temperature?};
%   nsamples [1] - Number of samples per time series, relative to the target time series 
%   deltaTp [1] - Number of local historical points
%   deltaTr [1] - Number of points to forecast
%   readme [string] -  Data information (source, formation time etc.)
%   dataset [string] - Reference name for the dataset
%   name [string] - Reference name for the time series
%   Optional:
%   type [1,N] (optional) Time series types ts.x, 1-real-valued, 2-binary, k ? k-valued
%   timegen [T,1]=func_timegen(timetick) (optional) Time ticks generator, may
%   contain the start (or end) time in UNIX format and a function to generate the vector t of the size [T,1]


if nargin < 1
    dirnames = {'HascData/sequence', 'NNcompetition/', 'EnergyWeatherTS/orig/', 'EnergyWeatherTS/missing_value/', ...
        'EnergyWeatherTS/missing_value/'};
end

if ~iscell(dirnames)
   dirnames = {dirnames}; 
end

save_dirname = fullfile('data','ProcessedData');
if ~exist(save_dirname, 'dir')
    mkdir(save_dirname);
end

for dirname = dirnames
    dirname = fullfile('data', dirname{1});
    folders = strsplit(dirname, filesep);
    parent_dir = folders{2};
    loader_func_handle = str2func(['Load', parent_dir]);
    ts_struct = feval(loader_func_handle, dirname);
    save_them_all(ts_struct, save_dirname);
end

end

function save_them_all(ts, save_dirname)

    for i = 1:numel(ts)
        ts_struct = ts{i};     
        save(fullfile(save_dirname, [ts_struct(1).name, '.mat']), 'ts_struct');
    end
end