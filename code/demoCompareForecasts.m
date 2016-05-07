%Data preparation. Time series length are fixed and precomputed.
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
ts_legend = {'Consumption', 'Max Temperature','Min Temperature','Precipitation','Wind','Relative Humidity','Solar'};
time_step = {1, 24,24,24,24,24,24};
self_deltaTp = {6*24,6,6,6,6,6,6};
self_deltaTr = {24,1,1,1,1,1,1};
tmp1 = linspace(numel(ts{2}), numel(ts{2}) - 7 * ts_length, ts_length+1);
tmp2 = linspace(numel(ts{1}), numel(ts{1}) - 24 * 7 * ts_length, ts_length+1);
time_points = {tmp2,tmp1,tmp1,tmp1,tmp1,tmp1, tmp1};
inputStructTS = struct('x', ts, 'time_step', time_step, 'legend', ts_legend, 'deltaTp', self_deltaTp, 'deltaTr', self_deltaTr, 'time_points', time_points);
[workStructTS] = CreateRegMatrix(inputStructTS);

names_vector = cellstr(char('VAR', 'Neural_network', 'SVR'));
model_draft = struct('name', [], 'params', [], 'tuned_func', [], 'error', [], 'unopt_flag', true, 'forecasted_y', []);
m = size(workStructTS.matrix,1);
alpha_coeff = 0;
K = 1;
for i = [1:3]
    model(i) = model_draft;
    model(i).name = names_vector{i};
    [MAPE_target, model(i), real_y] = ComputeForecastingErrors(workStructTS, K, m, alpha_coeff, model(i));
end


% VAR results are not plotted because it's unstable on samples [MxN] where
% M < N, just like our case. Feature selection is vital for it.
figure(1)
cla
plot(real_y(1:24), 'LineWidth', 2);
hold on
grid on
MAPE_full = zeros(3,1);
MAPE_target = zeros(3,1);
AIC = zeros(3,1);
for i = [2:3]
    plot(model(i).forecasted_y(1:24));
end
legend({'Real', 'NN', 'SVR'},'Location','NorthWest');

for i = [1:3]
    epsilon_target = (model(i).forecasted_y(1:24) - real_y(1:24));
    MAPE_target(i) = sqrt((1/24)*norm(epsilon_target));
    epsilon_full = (model(i).forecasted_y - real_y);
    MAPE_full(i) = sqrt(1/workStructTS.deltaTr)*norm(epsilon_full);
    AIC(i) = 2*workStructTS.deltaTp + size(workStructTS.matrix, 1) * log(norm(epsilon_full));
end
table(MAPE_target, MAPE_full, AIC, 'RowNames', names_vector)

