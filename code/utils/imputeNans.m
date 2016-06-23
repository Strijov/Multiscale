function X = imputeNans(X, ratio)

if nargin < 2
    ratio = 0.01;
end
    
nNans = max(1, round(numel(X)*ratio));

linearIdx = randperm(numel(X));
idxNans = linearIdx(1:nNans);

X(idxNans) = NaN;

end