function idxSplits = randomSplit(nRowsTotal, nRowsSplit)
% Randomly splits the design matrix with number of rows equal to nRowsTotal 
% into subsamples of size nRowsSplit


% Define the number of subsamples:
nSplits = nRowsTotal - nRowsSplit; 

% Each row of idxSplit is a list of subsmple indices:  
idxSplits = repmat(1:nRowsSplit, nSplit) + ...
           repmat((0:nSplits-1)',nRowsSplit, 1 );

% Randomply shuffle sample indices
idxShuffle = randperm(nRowsTotal);
idxSplits = idxShuffle(idxSplits);

% Arrange subsaple indices in ascending order, so that test does not preeed
% train objects:
idxSplits = sort(idxSplits, 2);

end