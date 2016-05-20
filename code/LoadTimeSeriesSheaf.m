function tsSheaf = LoadTimeSeriesSheaf(txtAlias)
% Load a set (Sheaf) of time series and returns the structre, where each ts is an item.
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
% txtAlias [string] an alias of time series colliction
%
% Output:
% tsSheaf [struct]
 
if nargin < 1, txtAlias = 'SL2'; end % Assign the default collection.

% List the predefined collections.
switch txtAlias
   case 'SL2'
      tsSheaf = load_SL2;
   case 'SL3'
      tsSheaf = load_SL2; % TODO Put the other collections here.
   otherwise
      error(['There is no loader for collection ', txtAlias,'.']);
end

end

% Collection SLC2
% FIXIT Put its description here.
function tsSheaf = load_SL2()
% This temporary script loads unique dataset.

if strcmp(computer,'MACI64'), WIN_XLS = false;
else WIN_XLS = true; end % Fix for xlsread.

addpath(genpath(cd));
filename = 'data/orig/SL2.xls';
sheet = 'Arkusz1';
xlRange = 'D3:AA1098';
ts0 = xlsread(filename,sheet,xlRange,'basic');
if ~WIN_XLS, ts0(:, 1:3) = []; end % TODO If there is no Windows the xlsread does not work.
ts0 = reshape(ts0', numel(ts0), 1);

tmp = xlsread('data/orig/weatherdata.xls', 'weatherdata', 'E2:J1093','basic');
if ~WIN_XLS, tmp(:, 1:4) = []; end % TODO If there is no Windows the xlsread does not work.
if WIN_XLS, tmp(:, 1:4) = []; end % AM: This leaves columns that correspond to ts_legend below
% ts = {ts0};
% for i = 1:size(tmp, 2)
%     ts{i+1} = tmp(:, i);
% end

ts  = [ts0, num2cell(tmp,1)];
ts_length = 155;
ts_legend = {'Consumption', 'Max Temperature','Min Temperature',...
                'Precipitation','Wind','Relative Humidity','Solar'};
time_step = {1, 24,24,24,24,24,24};
self_deltaTp = {6*24,6,6,6,6,6,6};
self_deltaTr = {24,1,1,1,1,1,1};
tmp1 = linspace(numel(ts{2}), numel(ts{2}) - 7 * ts_length, ts_length+1);
tmp2 = linspace(numel(ts{1}), numel(ts{1}) - 24 * 7 * ts_length, ts_length+1);
time_points = {tmp2,tmp1,tmp1,tmp1,tmp1,tmp1, tmp1};
tsSheaf = struct('x', ts, 'time_step', time_step, 'legend', legend, ...
                    'deltaTp', self_deltaTp, 'deltaTr', self_deltaTr,...
                    'time_points', time_points, 'normalization', []);
end