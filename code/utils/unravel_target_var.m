function tsArray = unravel_target_var(Y, yBlocks, norm_div, norm_subt)

yBlocks = [0, cumsum(yBlocks)];
tsArray = cell(1, numel(yBlocks) - 1);

for i = 1:numel(yBlocks) - 1
    ts = reshape(flip(Y(:, yBlocks(i) + 1:yBlocks(i+1)))', ...
                    1, numel(Y(:, yBlocks(i) + 1:yBlocks(i+1))));
    ts = ts*norm_div + norm_subt;
    tsArray{i} = ts';
end


end