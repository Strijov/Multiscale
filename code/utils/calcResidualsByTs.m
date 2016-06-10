function res = calcResidualsByTs(cellY, cellTs, deltaTp)

res = cell(1, numel(cellY));
for i = 1:numel(cellY)
    res{i} = cellTs{i}(deltaTp(i) + 1:deltaTp(i) + numel(cellY{i})) ...
                      -  cellY{i};
end

end