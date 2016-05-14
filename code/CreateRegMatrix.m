function workStructTS = CreateRegMatrix(s, legend) % FIXIT why this variable is here?
% It does something. FIXIT
%
% FIXIT and TODO make one simple structure and declare it in one place.
% Put link to this place here&
%
% Input:
% s         [struct]with fields:
% x         [1xn] cell with time-series
% time_step	[1xn] cell with ints - time step for each TS
% legend	[1xn] cell with strings, containing lengend of each TS
% deltaTp	[1xn] cell with ints, corresponds to number of feature columns
% deltaTr	[1xn] cell with ints,corresponds to number of target columns (Y)
% time_points  [1xn] cell with vectors of TS time-ticks
%
% Output:
% workStructTS struct with fields:
% matrix	[NxM] object-features matrix
% legend    [1xn] cell with strings, containing lengend of each TS
% deltaTp	[int] corresponds to number of feature columns (X)
% deltaTr	[int] corresponds to number of target columns (Y)
% self_deltaTp	[1xn] cell with ints, corresponds to number of feature columns
% self_deltaTr	[1xn] cell with ints,corresponds to number of target columns (Y)
% norm_div	[1xn] cell with normalization multiplier.
% norm_subt	[1xn] cell with minimum for each TS. 
%                           To get back to real values shold multiply with
%                           the first value and sum up with second.

ts_num = numel(s);
counterTp = 0;
counterTr = 0;
for i = 1:ts_num
	counterTp = counterTp + s(i).deltaTp;
    counterTr = counterTr + s(i).deltaTr;
end
matrix = zeros(numel(s(1).time_points), counterTp+counterTr);
shiftTp = 0;
shiftTr = counterTp;
norm_div = zeros(1,ts_num);
norm_subt = zeros(1, ts_num);
for i = 1:ts_num
    [nozmalized_ts, norm_div(i), norm_subt(i)] = NormalizeTS(s(i));
    small_matrix = CreateRegMatrix_small(nozmalized_ts, s(i).time_points, s(i).deltaTp, s(i).deltaTr);
    [tmpX,tmpY] = SplitIntoXY(small_matrix, s(i).deltaTp, s(i).deltaTr);
    matrix(:, shiftTp+1:shiftTp+s(i).deltaTp) = tmpX;
    matrix(:, shiftTr+1:shiftTr+s(i).deltaTr) = tmpY;
    shiftTp = shiftTp + s(i).deltaTp;
    shiftTr = shiftTr + s(i).deltaTr;
end
deltaTp = sum([s.deltaTp]);
self_deltaTp = [s.deltaTp];
deltaTr = sum([s.deltaTr]);
self_deltaTr = [s.deltaTr];

% FIXIT To discuss: does this struct bring extra complexity to this function?
workStructTS = struct('matrix', matrix, 'deltaTp', deltaTp, 'deltaTr', deltaTr, 'self_deltaTp', self_deltaTp, 'self_deltaTr', self_deltaTr, 'norm_div', norm_div, 'norm_subt', norm_subt);
end

% FIXIT Please simplify this part, get rid of loops.
function [matrix] = CreateRegMatrix_small(ts, time_points, deltaTp, deltaTr)
    obj_len = deltaTp + deltaTr;
    matrix = zeros(numel(time_points), obj_len);
    %FIXIT: first create zero-filled matrix, then add there real values -
    %increases computing speed.
    for i = 1:numel(time_points)
        tp = time_points(i);
        object = ts(tp - obj_len + 1 : tp);
        if (size(object, 1) > size(object, 2))
            object = object';
        end
        matrix(i, :) = matrix(i,:) + object;
    end  
end
