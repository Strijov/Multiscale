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


