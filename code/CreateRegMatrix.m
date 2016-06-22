function s = CreateRegMatrix(s, nPredictions, normalFlag) 
% This function creates a design matrix from the input time series structure
% and and returns a single updated the structure 
%
%
% Input:
% s is a cell array of ts structures (see load_data/LoadAndSave.m for details)
% with main fields:
%   x       [nx1] vector, time-series
%   time	[nx1] time stamps, date serials in Unix format
%   deltaTp	[int] number of local history points to consider
%   deltaTr	[int] number of time points to forecast
% nPredictions  [int] (optional) number of predictions to make. Specifies
%                       horizon length as n_predictions*deltaTr
%
% Output:
% this function adds the following fields:
% X	[m x N] feature columns of the design matrix
% Y	[m x deltaTr*n_predictions] target columns of the design matrix
% norm_div	[1xn] cell with normalization multiplier.
% norm_subt	[1xn] cell with minimum for each TS. 
%                 To get back to real values should multiply with
%                 the first value and sum up with second.



if nargin < 2
    nPredictions = 1;
end
if nargin < 3
    normalFlag = true;
end

nTs = numel(s);

% compute the number of rows in design matrix
nRows = floor((numel(s(1).x) - s(1).deltaTp)/s(1).deltaTr/nPredictions);


% init normalization fields:
norm_div = zeros(1, nTs);
norm_subt = zeros(1, nTs);

% define boundaries of var blocks in design matrix 
xBlocks = [0, cumsum([s().deltaTp])];
yBlocks = [0, cumsum([s().deltaTr])*nPredictions];
X = zeros(nRows, xBlocks(end));
Y = zeros(nRows, yBlocks(end));
timeY = zeros(nRows, nTs); % for testing purposes, delete later

% normalize time series before adding them to design matrix 
for i = 1:nTs
    s(i).deltaTr = s(i).deltaTr*nPredictions;
    if normalFlag
        [normalizedTs, norm_div(i), norm_subt(i)] = NormalizeTS(s(i).x);
    % s(i).x = normalizedTs;
    else
        normalizedTs = s(i).x;
        norm_div(i) = 1;
        norm_subt(i) = 0;
    end
    
    [Y(:, yBlocks(i) +  1:yBlocks(i+1)), ...
     X(:, xBlocks(i) + 1:xBlocks(i+1)), ...
     timeY(:, i)] = create_matrix_from_target(s(i), normalizedTs);
end

if ~ckeckTimeIsRight(timeY)
    warning('regMatrixTimes:id', 'CreateRegMatrix: Time entries might be inconsistent');
end


% write the results to one stuct:
s(1).deltaTr = [s().deltaTr];
s(1).deltaTp = [s().deltaTp];
s(1).x = {s().x};
s(1).time = {s().time};
s(1).legend = {s().legend};
s = s(1);
s.X = X;
s.Y = Y;
s.norm_subt = norm_subt;
s.norm_div = norm_div;

s = trimTimeSeries(s);
   
if normalFlag && ~checkTsTrimming(s)
    warning('regMatrixTsTrimming:id', 'CreateRegMatrix: Time series trimming went wrong');
end

end

function [Y, X, timeY] = create_matrix_from_target(s, normalized_ts)

% reverse time series, so that the top row is always to be forecasted
ts = flipud(normalized_ts);
time = flipud(s.time);

idx_rows = 1:s.deltaTr:numel(ts) - s.deltaTr - s.deltaTp + 1;
idx = bsxfun(@plus, idx_rows', 0:s.deltaTr + s.deltaTp - 1);

Y = fliplr(ts(idx(:, 1:s.deltaTr)));
X = fliplr(ts(idx(:, s.deltaTr + 1:s.deltaTr + s.deltaTp)));
timeY = time(idx_rows + s.deltaTr - 1);

if numel(idx_rows) == 1
    Y = Y';
    X = X';
end
% test: uravel Y and plot it against original timeseries
% this should be an identity plot
% plot(reshape(flipud(Y)', 1, numel(Y)), normalized_ts(s.deltaTp+1:end))
end

function ts = trimTimeSeries(ts)

% leaves only the parts of original time series that will be foreasted
nRows = size(ts.Y, 1);
ts.x = arrayfun(@(i) ts.x{i}(ts.deltaTp(i) + 1:ts.deltaTp(i) ...
                            + nRows*ts.deltaTr(i)), 1:numel(ts.x), ...
                            'UniformOutput', false);
%ts.time = arrayfun(@(i) ts.time{i}(ts.deltaTp(i) + 1:ts.deltaTp(i) ...
%                            + nRows*ts.deltaTr(i)), 1:numel(ts.x), ...
%                            'UniformOutput', false);
end

function checkRes = ckeckTimeIsRight(timeY)

checkRes =  ~any(max(timeY(2:end, :), [], 2) > min(timeY(1:end-1, :), [], 2));

end

function checkRes = checkTsTrimming(ts)
TOL = 10^(-10);
y2ts = unravel_target_var(ts.Y, ts.deltaTr, ts.norm_div, ts.norm_subt);
%y2ts = denormalize(y2ts, ts.norm_div, ts.norm_subt);
checkRes = all(cell2mat(cellfun(@(x, y) all(abs(x - y) < TOL), y2ts, ts.x,...
                            'UniformOutput', false)));

end