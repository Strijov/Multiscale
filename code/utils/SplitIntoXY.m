function [X, Y] = SplitIntoXY(matrix, deltaTp, deltaTr)
    X = matrix(:, 1:end - deltaTr);
    Y = matrix(:, end - deltaTr + 1:deltaTr);
end