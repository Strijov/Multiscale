function dataStruct = createSimpleDataStruct(func, noiseLevel, segmLength, nHistPoints)

nTs = 1;
nSegments = 100;

tsSegment = feval(func, segmLength);
tsSegment = tsSegment(:);

% copy segments and add noise:
ts = repmat(tsSegment, nSegments, 1);
ts = ts + randn(size(ts))*noiseLevel;

deltaTr = segmLength;
deltaTp = nHistPoints*deltaTr;
time = (1:numel(ts))';

deltaTr = num2cell(deltaTr);
deltaTp = num2cell(deltaTp);
dataStruct = struct('x', ts, 'time', time, ...
            'legend', arrayfun(@num2str, 1:nTs, 'UniformOutput', 0),...
            'deltaTp', deltaTp, 'deltaTr', deltaTr,...
            'name', ['linear_segments_', num2str(noiseLevel)], 'readme', 'randomly generated data',...
            'dataset', 'ExampleData');


end