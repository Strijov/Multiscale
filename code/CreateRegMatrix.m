function s = CreateRegMatrix(s, nPredictions) 
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
    [normalizedTs, norm_div(i), norm_subt(i)] = NormalizeTS(s(i));
    [Y(:, yBlocks(i) +  1:yBlocks(i+1)), ...
     X(:, xBlocks(i) + 1:xBlocks(i+1)), ...
     timeY(:, i)] = create_matrix_from_target(s(i), normalizedTs);
end

if ~ckeckTimeIsRight(timeY)
    disp('CreateRegMatrix: Time entries might be inconsistent');
end

% write the results to one stuct:
s(1).deltaTr = [s().deltaTr];
s(1).deltaTp = [s().deltaTp];
s = s(1);
s.X = X;
s.Y = Y;
s.norm_subt = norm_subt;
s.norm_div = norm_div;

end

function [Y, X, timeY] = create_matrix_from_target(s, normalized_ts)

% reverse time series, so that the top row is always to be forecasted
ts = flip(normalized_ts);
time = flip(s.time);

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
% plot(reshape(flip(Y)', 1, numel(Y)), normalized_ts(s.deltaTp+1:end))
end

function checkRes = ckeckTimeIsRight(timeY)

checkRes =  ~any(max(timeY(2:end, :), [], 2) > min(timeY(1:end-1, :), [], 2));

end


function X = add_timeseries(s, nozmalized_ts, timeY)

% reverse time series, so that the top row is allways to be forecasted
ts = flip(nozmalized_ts);
time = flip(s.time);

X = zeros(numel(timeY), s.deltaTp);
for i = 1:numel(timeY)
   idx_rows = find(time < timeY(i));
   idx_rows = idx_rows(s.deltaTp:-1:1);  
   X(i, :) = ts(idx_rows);
end

end