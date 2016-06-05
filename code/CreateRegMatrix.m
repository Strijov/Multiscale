function s = CreateRegMatrix(s, n_predictions) 
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
% n_predictions  [int] (optional) number of predictions to make. Specifies
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
    n_predictions = 1;
end


ts_num = numel(s);
% remember deltaTr value:
deltaTr = s(1).deltaTr;
% temporarily replace it deltaTr value:
s(1).deltaTr = deltaTr*n_predictions;

%s().norm_div = [];
%s().norm_subt = [];

[normalized_ts, s(1).norm_div, s(1).norm_subt] = NormalizeTS(s(1));
[Y, X, timeY] = create_matrix_from_target(s(1), normalized_ts);

if ts_num == 1
   s = s(1);
   s.X = X;
   s.Y = Y; 
   return
end

for i = 2:ts_num
    [normalized_ts, ~, ~] = NormalizeTS(s(i));
    X = [X, add_timeseries(s(i), normalized_ts, timeY)];
end

s = s(1);
%s.matrix = [X, Y];
s.X = X;
s.Y = Y;

% finally, return deltaTr value to its place:
s.deltaTr = deltaTr;

end

function [Y, X, timeY] = create_matrix_from_target(s, normalized_ts)

% reverse time series, so that the top row is always to be forecasted
ts = flip(normalized_ts);
time = flip(s.time);

idx_rows = 1:s.deltaTr:numel(ts) - s.deltaTr - s.deltaTp + 1;
idx = repmat(idx_rows', 1, s.deltaTr + s.deltaTp)...
    + repmat(0:s.deltaTr + s.deltaTp - 1, numel(idx_rows), 1);

Y = fliplr(ts(idx(:, 1:s.deltaTr)));
X = fliplr(ts(idx(:, s.deltaTr + 1:s.deltaTr + s.deltaTp)));
timeY = time(idx_rows + s.deltaTr - 1);

% test: uravel Y and plot it against original timeseries
% this should be an identity plot
% plot(reshape(flip(Y)', 1, numel(Y)), normalized_ts(s.deltaTp+1:end))
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