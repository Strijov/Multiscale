function ts = MergeDataset(tsStructArray, nPredictions)

X = [];
Y = [];
rawTs = cell(1, numel(tsStructArray{1}));
for i = 1:numel(tsStructArray)
    ts = CreateRegMatrix(tsStructArray{i}, nPredictions);
    ts = trimTimeSeries(ts);
    rawTs = MergeTimeSeries(rawTs, ts.x);
    X = [X; ts.X];
    Y = [Y; ts.Y];    
end

ts.X = X;
ts.Y = Y;
ts.x = rawTs;
ts.name = 'All';

end

function ts = trimTimeSeries(ts)

% leaves only the parts of original time series that will be foreasted
nRows = size(ts.Y, 1);
ts.x = arrayfun(@(i) ts.x{i}(ts.deltaTp(i) + 1:ts.deltaTp(i) ...
                            + nRows*ts.deltaTr(i)), 1:numel(ts.x), ...
                            'UniformOutput', false);

end

function ts = MergeTimeSeries(old_ts, new_ts)

% concat new time series with old
ts = cellfun(@(x, y) vertcat(x, y), old_ts, new_ts, 'UniformOutput', false);

end