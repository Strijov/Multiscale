function [matrix, deltaTp, deltaTr] = CreateRegMatrix(s)
%     input:
%         struct s with fields:
%             x           - [1xn] cell with time-series
%             time_step   - [1xn] cell with ints - time step for each TS
%             legend      - [1xn] cell with strings, containing lengend of each TS
%             deltaTp     - [1xn] cell with ints, corresponds to number of feature columns
%             deltaTr     - [1xn] cell with ints,corresponds to number of target columns (Y)
%             time_points - [1xn] cell with vectors of TS time-ticks
%             normalization - [1x2] vector. First value is normalization
%             multiplier, second is TS minimum. To get back to real values 
%               shold multiply with the first value and sum up with second.
%      output:
%         matrix          - [NxM] object-features matrix
%         deltaTp         - int, corresponds to number of feature columns (X)
%         deltaTr         - int, corresponds to number of target columns (Y)

    
\ No newline at end of file
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
    for i = 1:ts_num
        nozmalized_ts = NormalizeTS(s(i));
        small_matrix = CreateRegMatrix_small(nozmalized_ts, s(i).time_points, s(i).deltaTp, s(i).deltaTr);
        [tmpX,tmpY] = SplitIntoXY(small_matrix, s(i).deltaTp, s(i).deltaTr);
        matrix(:, shiftTp+1:shiftTp+s(i).deltaTp) = tmpX;
        matrix(:, shiftTr+1:shiftTr+s(i).deltaTr) = tmpY;
        shiftTp = shiftTp + s(i).deltaTp;
        shiftTr = shiftTr + s(i).deltaTr;
    end
    deltaTp = sum([s.deltaTp]);
    deltaTr = sum([s.deltaTr]);
end