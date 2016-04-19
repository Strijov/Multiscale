function [trainMatrix, testMatrix, validation] = TrainTestSplit(matrix, alpha_coeff)
    validation = matrix(1,:);
    matrix(1,:) = [];
    test_size = floor(alpha_coeff * (size(matrix, 1) - 1));
    testMatrix = matrix(1:test_size,:);
    trainMatrix = matrix(test_size+1:end, :);
end