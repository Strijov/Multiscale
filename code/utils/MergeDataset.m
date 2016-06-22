function ts = MergeDataset(tsStructArray, nPredictions)

% Merges tsStructures from the data set into a single working struct
% The reason to merge structures instead of raw time series is to merge
% struts so that the matrices do not intersect

X = [];
Y = [];
rawTs = cell(1, numel(tsStructArray{1}));

for i = 1:numel(tsStructArray)
    rawTs = MergeTimeSeries(rawTs, {tsStructArray{i}.x});
end

% Prenormalization:
norm_div = zeros(1, numel(rawTs));
norm_subt = zeros(1, numel(rawTs));
for i = 1:numel(rawTs)
    [normalizedTs, norm_div(i), norm_subt(i)] = NormalizeTS(rawTs{i});
    %tsStructArray{i}.x = normalizedTs;
end

for i = 1:numel(tsStructArray)
    % normalize inputs to CreateRegMatrix:
    %ts_x = cellfun(@(x,a,b) (x - repmat(a, size(x)))./repmat(b, size(x)), ...
    %    {tsStructArray{i}.x}, norm_div, norm_subt, 'UniformOutput', 0);
    ts_x = renormalize({tsStructArray{i}.x}, norm_div, norm_subt);
    [tsStructArray{i}.x] = deal(ts_x{:});
    % suppress normalzation in createRegMatrix
    ts = CreateRegMatrix(tsStructArray{i}, nPredictions, false);
    % denormalize time series:
    %ts_x = cellfun(@(x,a,b) (x.*repmat(b, size(x)) + repmat(a, size(x))), ...
    %    {tsStructArray{i}.x}, norm_div, norm_subt, 'UniformOutput', 0);
    ts_x = denormalize({tsStructArray{i}.x}, norm_div, norm_subt);
    rawTs = MergeTimeSeries(rawTs, ts_x);
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
