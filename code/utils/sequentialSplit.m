function idxSplits = sequentialSplit(nRowsTotal, nRowsSplit)
% Performs sequential split of design matrix with number of rows equal to
% nRowsTotal into subsamples of size nRowsSplit

% Define the number of subsamples:
nSplits = nRowsTotal - nRowsSplit; 

% Create matrix idxSplits, where each row is a list of subsmple indices:  
idxSplits = repmat(1:nRowsSplit, nSplits, 1) + ...
           repmat((0:nSplits-1)', 1, nRowsSplit);

end