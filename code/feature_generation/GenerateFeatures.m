function [ts, generators] = GenerateFeatures(ts, generators, idxTrain, idxTest)
% Generates new feature matrix using methods, specified with generators.
%
% Input:
% ts	- see createRegMatrix.m for explanation
% generators - cell array of feature generator models structures. 
%   generator.handle - stores handles to possible generator functions. Options 
%   are @ConvGenerator, @CubicGenerator, @NwGenerator, @SsaGenerator. When no
%   feature generation is required, use {@IdentityGenerator}
%   generator.transform - handle to the feature generation on the test set.
%   generator.replace - this parameter specifies weather the new features
%   replace old matrix or add up on the left.
% idxTrain - indices of train set, optional 
% idxTest  - indices of test set, optional. If idxTrain and idxTest are
%   specified, generators learn parameters on train set and return handle to 
%   transform features of the test set.  
%
% Output:
% ts with new feature matrix

if nargin == 2
   idxTrain = (1:size(ts.X, 1))';
   idxTest = zeros(0, 1);
elseif nargin == 3
    idxTest = (1:size(ts.X, 1))';
    idxTest = idxTest(~ismember(idxTest, idxTrain));
end


% Only use the original feature matrix to generate new features: 
Xold = ts.X(:, 1:sum(ts.deltaTp)); % FIXIT this way only historical points are used
xBlocks = [0, cumsum(ts.deltaTp)];
X = [];

idxNW = strcmp({generators().name}, {'NW'});
if ~generators(idxNW).replace
    warning('genFeatsReplace:id', 'Smoothing generator "NW" is used with replace set to false.');
end


% generate new features separately for all submatrices of different time
% series:
[Xnew, generators] = genSubMatrix(generators, Xold, idxTrain, idxTest); 
X = [X, Xnew];

if ~any([generators().replace])
   Xnew = [Xold, X]; 
end

% no matter what, add a constant:
Xnew = [Xnew, ones(size(X, 1), 1)];
ts.X = Xnew;
    
end    


function [Xnew, generators] = genSubMatrix(generators, Xold, idxTrain, idxTest)
   
trainXnew = [];
testXnew = [];    
for i  = 1:numel(generators)
    [trainX, generators(i)] = feval(generators(i).handle, Xold(idxTrain, :), generators(i));
    trainXnew = [trainXnew, trainX];
    testXnew = [testXnew, feval(generators(i).transform, Xold(idxTest, :))];
end

Xnew = zeros(size(Xold, 1), size(trainXnew, 2));
Xnew(idxTrain, :) = trainXnew;
Xnew(idxTest, :) = testXnew;

    

end

