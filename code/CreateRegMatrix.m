function [X, Y, x ,y] = CreateRegMatrix(ts, time_points, deltaTp, deltaTr)
    obj_len = deltaTp + deltaTr;
    matrix = []
    
    for tp = time_points
        object = ts(tp - obj_len + 1 : tp);
        if (size(object, 1) > size(object, 2))
            object = object';
        end
        matrix = [matrix; object];
    end
    
    X = matrix(:, 1:deltaTp);
    Y = matrix(:, deltaTp + 1:obj_len);
    x = X(1, :);
    X(1, :) = [];
    y = Y(1, :);
    Y(1, :) = [];
end

    