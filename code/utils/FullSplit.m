function [idxTrain, idxTest, idxVal, idxX, idxY] = FullSplit(nSamples, nVars, alpha_coeff, deltaTr)
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
    [idxTrain, idxTest, idxVal] = TrainTestSplit(nSamples, alpha_coeff);
    idxX = 1:nVars - deltaTr;
    idxY = nVars - deltaTr + 1: nVars;
end
