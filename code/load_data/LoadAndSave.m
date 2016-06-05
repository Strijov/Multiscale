function LoadAndSave(dirnames)
% Data loader
%
% Input: dirnames - cell array, contains names of folders inside data/
% directory, where the desired datasets are stored. Each folder 'DataName' inside
% data/ directory has a corresponding loader 'LoadDataName'.
%
% Output: a cell array of ts structures
% Description of the time series structure: 
% s         [1xn_ts] array of structures with fields:
% x         [nx1] vector, time-series
% time	    [nx1] time stamps, date serials in Unix format
% legend	[string] contains lengend of each TS
% deltaTp	[int] number of local history points to consider
% deltaTr	[int] number of time points to forecast
% name  	[string] reference name of the particular time series
% readme  	[string] (optional) data description, needed for report
% dataset  	[string] reference name of the dataset

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