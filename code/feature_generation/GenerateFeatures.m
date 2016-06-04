function [StructTS] = GenerateFeatures(StructTS, generators, ...
                                            idxTrain, idxTest)
% Generates new feature matrix using methods, specified with generators.
%
% Input:
% workStructTS	see createRegMatrix.m for explanation
% generators    cell array of feature generator handles. Options are 
% @ConvGenerator, @CubicGenerator, @NwGenerator, @SsaGenerator. When no
% feature generation is required, use {@IdentityGenerator}
%
% Output:
% workStructTS with new feature matrix

if nargin == 2
   idxTrain = (1:size(StructTS.X, 1))';
   idxTest = zeros(0, 1);
elseif nargin == 3
    idxTest = (1:size(StructTS.X, 1))';
    idxTest = idxTest(~ismember(idxTest, idxTrain));
end


% only use the original feature matrix to generate new features: 
Xold = StructTS.X;

trainXnew = [];
testXnew = [];    
for i  = 1:numel(generators)
    [trainX, generators(i)] = feval(generators(i).handle, Xold(idxTrain, :), generators(i));
    trainXnew = [trainXnew, trainX];
    testXnew = [testXnew, feval(generators(i).transform, Xold(idxTest, :))];
end

Xnew = zeros(size(Y, 1), size(trainXnew, 2));
Xnew(idxTrain, :) = trainXnew;
Xnew(idxTest, :) = testXnew;
StructTS.X = Xnew;
%workStructTS.deltaTp = size(Xnew, 2);
    
end    
   
    

    

