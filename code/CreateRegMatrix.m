function [matrix] = CreateRegMatrix(ts, time_points, deltaTp, deltaTr)
    obj_len = deltaTp + deltaTr;
    matrix = [];
    for tp = time_points
        object = ts(tp - obj_len + 1 : tp);
        if (size(object, 1) > size(object, 2))
            object = object';
        end
        matrix = [matrix; object];
    end  
end

    