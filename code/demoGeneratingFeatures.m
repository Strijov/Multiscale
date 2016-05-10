%Unit tests
%Reading data from file(s).
addpath(genpath(cd));
filename = 'data\orig\SL2.xls';
sheet = 'Arkusz1';
xlRange = 'D3:AA1098';
ts0 = xlsread(filename,sheet,xlRange);
ts0 = reshape(ts0', numel(ts0), 1);

tmp = xlsread('data\orig\weatherdata.csv', 1, 'E2:J1093');
ts{1} = ts0;
for i = [1:size(tmp, 2)]
    ts{i+1} = tmp(:, i);
end
ts_length = 155;
legend = {'Consumption', 'Max Temperature','Min Temperature','Precipitation','Wind','Relative Humidity','Solar'};
time_step = {1, 24,24,24,24,24,24};
self_deltaTp = {6*24,6,6,6,6,6,6};
self_deltaTr = {24,1,1,1,1,1,1};
tmp1 = linspace(numel(ts{2}), numel(ts{2}) - 7 * ts_length, ts_length+1);
tmp2 = linspace(numel(ts{1}), numel(ts{1}) - 24 * 7 * ts_length, ts_length+1);
time_points = {tmp2,tmp1,tmp1,tmp1,tmp1,tmp1, tmp1};
inputStructTS = struct('x', ts, 'time_step', time_step, 'legend', legend, 'deltaTp', self_deltaTp, 'deltaTr', self_deltaTr, 'time_points', time_points, 'normalization', []);

%Constructing regression matrix.
[workStructTS] = CreateRegMatrix(inputStructTS);
cla
%Generating extra features:
generator_names = {'SSA', 'NW', 'Cubic', 'Conv'};
for i = [1:numel(generator_names)]
    generator = generator_names(i);
    structWithNewFeatures{i} = GenerateFeatures(workStructTS, generator);
    figure(i)
    pcolor(structWithNewFeatures{i}.matrix)
    title(generator_names(i))
end
structWithNewFeatures
'done'