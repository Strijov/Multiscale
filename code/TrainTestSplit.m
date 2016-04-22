function [trainMatrix, testMatrix, validation] = TrainTestSplit(matrix, alpha_coeff)
    %Input:
    %   matrix         - regression matrix [MxN] (X|Y)
    %   alpha_coeff    - float, test size M2 divided to train size M1
    %Output:
    %   trainMatrix    - matrix[M1xN]
    %   testMatrix     - matrix[M2xN]
    %validation     - vector[1xN]
    %Just splits matrix to train, test & validation
    validation = matrix(1,:);
    matrix(1,:) = [];
    test_size = floor(alpha_coeff * (size(matrix, 1) - 1));
    testMatrix = matrix(1:test_size,:);
    trainMatrix = matrix(test_size+1:end, :);
end