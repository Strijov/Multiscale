function [trainX, trainY, testX, testY, val_x, val_y] = FullSplit(matrix, alpha_coeff, deltaTp, deltaTr)
    %Input:
    %   matrix         - regression matrix [MxN] (X|Y)
    %   alpha_coeff    - float, test size M2 divided to train size M1
    %   deltaTp        - X strings size, prehistory length
    %   deltaTr        - Y strings size, forecast length
    %Output:
    %   trainX, trainY - matrices [M1xdeltaTp] and [M1xdeltaTr] 
    %   testX, testY   - matrices [M2xdeltaTp] and [M2xdeltaTr] 
    %   val_x, val_y   - vectors[1xdeltaTp] and [1xdeltaTr]
    %Splitting matrix to X-Y train, test and validation sets.
    [trainMatrix, testMatrix, validation] = TrainTestSplit(matrix, alpha_coeff);
    [trainX, trainY] = SplitIntoXY(trainMatrix, deltaTp, deltaTr);
    [testX, testY] = SplitIntoXY(testMatrix, deltaTp, deltaTr);
    [val_x, val_y] = SplitIntoXY(validation, deltaTp, deltaTr);
end