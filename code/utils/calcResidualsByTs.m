function res = calcResidualsByTs(cellY, cellTs, deltaTp)

%{
if ~exist('idxTrain', 'var')
    idxTrain = 1:numel(Y);
end
if ~exist('idxTest', 'var')
    idxTest = [];
end
%}

res = cell(1, numel(cellY));
for i = 1:numel(cellY)
    res{i} = cellTs{i}(1:numel(cellY{i})) -  cellY{i};
    % This part applies if ts are not trimmed:
    %res{i} = cellTs{i}(deltaTp(i) + 1:deltaTp(i) + numel(cellY{i})) ...
    %                  -  cellY{i};
end

end