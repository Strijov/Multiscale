function ts = MergeDataset(tsStructArray, nPredictions)

X = [];
Y = [];
rawTs = cell(1, numel(tsStructArray{1}));
for i = 1:numel(tsStructArray)
    ts = CreateRegMatrix(tsStructArray{i}, nPredictions);
    %ts = trimTimeSeries(ts);
    rawTs = MergeTimeSeries(rawTs, ts.x);
    X = [X; ts.X];
    Y = [Y; ts.Y];    
end

ts.X = X;
ts.Y = Y;
ts.x = rawTs;
if numel(ts.x) > 1
    ts.name = [ts.name, '_all'];
else
    ts.name = [ts.name, '_', ts.legend{1}, '_all'];
end
end



function ts = MergeTimeSeries(old_ts, new_ts)

% concat new time series with old
ts = cellfun(@(x, y) vertcat(x, y), old_ts, new_ts, 'UniformOutput', false);

end