function s = CreateRegMatrix(s) 
% This function creates a design matrix from the input time series structure
% and and returns a single updated the structure 
%
% FIXIT and TODO make one simple structure and declare it in one place.
% Put link to this place here&
%
% Input:
% s         [1xn_ts] array of structures with fields:
% x         [nx1] vector, time-series
% time	    [nx1] time stamps, date serials in Unix format
% legend	[string] contains lengend of each TS
% deltaTp	[int] number of local history points to consider
% deltaTr	[int] number of time points to forecast
% name  	[string] reference name of the particular time series
% readme  	[string] (optional) data description, needed for report
% dataser  	[string] reference name of the dataset
%
% Output:
% this function adds the following fields:
% matrix	[NxM] object-features matrix
% norm_div	[1xn] cell with normalization multiplier.
% norm_subt	[1xn] cell with minimum for each TS. 
%                           To get back to real values should multiply with
%                           the first value and sum up with second.

ts_num = numel(s);

%s().norm_div = [];
%s().norm_subt = [];

[normalized_ts, s(1).norm_div, s(1).norm_subt] = NormalizeTS(s(1));
[Y, X, timeY] = create_matrix_from_target(s(1), normalized_ts);

if ts_num == 1
   s = s(1);
   s.matrix = [X, Y]; 
   return
end

for i = 2:ts_num
    [normalized_ts, ~, ~] = NormalizeTS(s(i));
    X = [X, add_timeseries(s(i), normalized_ts, timeY)];
end

s = s(1);
s.matrix = [X, Y];
%s.deltaTp = size(X, 2);

end

function [Y, X, timeY] = create_matrix_from_target(s, nozmalized_ts)

% reverse time series, so that the top row is allways to be forecasted
ts = flip(nozmalized_ts);
time = flip(s.time);

idx_rows = 1:s.deltaTr:numel(ts) - s.deltaTr - s.deltaTp + 1;
idx = repmat(idx_rows', 1, s.deltaTr + s.deltaTp)...
    + repmat(0:s.deltaTr + s.deltaTp - 1, numel(idx_rows), 1);

Y = ts(idx(:, 1:s.deltaTr));
X = ts(idx(:, s.deltaTr + 1:s.deltaTr + s.deltaTp));
timeY = time(idx_rows + s.deltaTr - 1);

end

function X = add_timeseries(s, nozmalized_ts, timeY)

% reverse time series, so that the top row is allways to be forecasted
ts = flip(nozmalized_ts);
time = flip(s.time);

X = zeros(numel(timeY), s.deltaTp);
for i = 1:numel(timeY)
   idx_rows = find(time < timeY(i));
   idx_rows = idx_rows(1:s.deltaTp);  
   X(i, :) = ts(idx_rows);
end

end