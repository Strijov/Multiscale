function [X, Y] = SplitIntoXY(matrix, deltaTp, deltaTr)
    X = matrix(:, 1:deltaTp);
    Y = matrix(:, deltaTp + 1:deltaTp + deltaTr);
end