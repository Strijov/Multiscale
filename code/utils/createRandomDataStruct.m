function dataStruct = createRandomDataStruct(nTs, nSamples)

if nargin < 1
    nTs = randi(10);
end

if nargin < 2
    nSamples = round(rand()*500) + 100;
end

deltaTr = randi(10, [1, nTs]);
deltaTp = 2*deltaTr;
ts = arrayfun(@(x) randn(nSamples*x, 1), deltaTr, 'UniformOutput', 0);
time = arrayfun(@(x) linspace(1, nSamples, nSamples*x)', deltaTr, 'UniformOutput', 0);

deltaTr = num2cell(deltaTr);
deltaTp = num2cell(deltaTp);
dataStruct = struct('x', ts, 'time', time, ...
            'legend', arrayfun(@num2str, 1:nTs, 'UniformOutput', 0),...
            'deltaTp', deltaTp, 'deltaTr', deltaTr,...
            'name', 'random_data', 'readme', 'randomly generated data',...
            'dataset', 'RandomData');


end