%Data preparation. Time series length are fixed and precomputed.
filename = 'data\orig\SL2.xls';
sheet = 'Arkusz1';
xlRange = 'D3:AA366';
ts0 = xlsread(filename,sheet,xlRange);
ts0 = reshape(ts0', numel(ts0), 1);
tmp = xlsread('data\orig\weatherdata.csv', 1, 'E2:J365');
ts{1} = ts0;
for i = [1:size(tmp, 2)]
    ts{i+1} = tmp(:, i);
end
ts_length = 300;
legend = {'Consumption', 'Max Temperature','Min Temperature','Precipitation','Wind','Relative Humidity','Solar'};
time_step = {1, 24,24,24,24,24,24};
self_deltaTp = {6*24,6,6,6,6,6,6};
self_deltaTr = {24,1,1,1,1,1,1};
tmp1 = linspace(numel(ts{2}), numel(ts{2}) - ts_length, ts_length+1);
tmp2 = linspace(numel(ts{1}), numel(ts{1}) - 24 * ts_length, ts_length+1);
time_points = {tmp2,tmp1,tmp1,tmp1,tmp1,tmp1, tmp1};
structTS = struct('x', ts, 'time_step', time_step, 'legend', legend, 'deltaTp', self_deltaTp, 'deltaTr', self_deltaTr, 'time_points', time_points, 'normalization', []);
[matrix, deltaTp, deltaTr] = CreateRegMatrix(structTS);


m = 250;
alpha_coeff = 0;
K = 10;

model_draft = struct('name', [], 'params', [], 'tuned_func', [], 'error', [], 'unopt_flag', true, 'forecasted_y', []);
names_vector = cellstr(char('VAR', 'Neural_network'));

%Creatung array of model structures 
%TODO: real_y should be pre-created, now is being re-created on each
%iteration
for i = 1:numel(names_vector)
    model(i) = model_draft;
    model(i).name = char(names_vector(i));
    [RMSE, model(i), real_y] = ComputeForecastingErrors(matrix, K, m, alpha_coeff, model(i), deltaTp, deltaTr);
end
figure(1)
for i = 1:numel(names_vector)
    plot(model(i).forecasted_y);
    hold on;
end
grid on;
plot(real_y, 'LineWidth', 1.5)
legend('VAR', 'Neural net', 'Real')

%varyind deltaTp
deltaTp_bounds = [24:144];
self_deltaTr = 24;
m = 250;
alpha_coeff = 0;
K = 1;
error_vec = zeros(1, numel(deltaTp_bounds));
for i = 1:numel(deltaTp_bounds)
    self_deltaTp = deltaTp_bounds(i);
    [matrix] = CreateRegMatrix(ts0, time_points, self_deltaTp, self_deltaTr);
    [RMSE, model(i), real_y] = ComputeForecastingErrors(matrix, K, m, alpha_coeff, model(1), self_deltaTp, self_deltaTr);
    error_vec(i) = RMSE;
end
figure(2)
plot(deltaTp_bounds, error_vec)
grid on;
xlabel 'deltaTp'
ylabel 'RMSE'

%varyind deltaTr
deltaTr_bounds = [2:48];
self_deltaTp = 144;
m = 250;
alpha_coeff = 0;
K = 1;
error_vec = zeros(1, numel(deltaTr_bounds));
for i = 1:numel(deltaTr_bounds)
    self_deltaTr = deltaTr_bounds(i);
    [matrix] = CreateRegMatrix(ts0, time_points, self_deltaTp, self_deltaTr);
    [RMSE, model(i), real_y] = ComputeForecastingErrors(matrix, K, m, alpha_coeff, model(1), self_deltaTp, self_deltaTr);
    error_vec(i) = RMSE;
end


figure(3)
plot(deltaTr_bounds, error_vec)
grid on;
xlabel 'deltaTr'
ylabel 'RMSE'

%varyind both deltaTp and deltaTr
deltaTr_bounds = [2:48];
deltaTp_bounds = [24:196];
m = 250;
alpha_coeff = 0;
K = 1;
error_mat = zeros(numel(deltaTp_bounds), numel(deltaTr_bounds));
for i = 1:numel(deltaTp_bounds)
    for j = 1:numel(deltaTr_bounds)
    self_deltaTr = deltaTr_bounds(j);
    self_deltaTp = deltaTp_bounds(i);
    [matrix] = CreateRegMatrix(ts0, time_points, self_deltaTp, self_deltaTr);
    [RMSE, model(i), real_y] = ComputeForecastingErrors(matrix, K, m, alpha_coeff, model(1), self_deltaTp, self_deltaTr);
    error_mat(i,j) = RMSE;
    end
end
figure(4)
surf(deltaTr_bounds, deltaTp_bounds, error_mat)
ylabel 'deltaTp'
xlabel 'deltaTr'
zlabel 'RMSE'
colorbar

% 
% 
% %varying m
% deltaTp = 144;
% deltaTr = 24;
% m_bounds = [150:270];
% alpha_coeff = 0;
% K = 1;
% error_vec = zeros(1, numel(m_bounds));
% for i = 1:numel(m_bounds)
%     m = m_bounds(i);
%     [RMSE, model(1), real_y] = ComputeForecastingErrors(matrix, K, m, alpha_coeff, model(1), deltaTp, deltaTr);
%     error_vec(i) = RMSE;
% end
% 
% 
% figure(3)
% plot(m_bounds, error_vec)
% grid on;
% xlabel 'm'
% ylabel 'RMSE'
% 
% %varying detlaTr x m
% deltaTp_bounds = [24:144];
% m_bounds = [150:270];
% alpha_coeff = 0;
% K = 1;
% error_mat = zeros(numel(deltaTp_bounds), numel(m_bounds));
% for i = 1:numel(deltaTp_bounds)
%     deltaTp = deltaTp_bounds(i);
%     deltaTr = 168 - deltaTp;
%     for j = 1:numel(m_bounds)
%         m = m_bounds(j);
%         [RMSE, model(1), real_y] = ComputeForecastingErrors(matrix, K, m, alpha_coeff, model(1), deltaTp, deltaTr);
%         error_mat(i,j) = RMSE;
%     end
% end
% figure(4)
% 
% surfc(deltaTp_bounds, m_bounds, error_mat)
% xlabel 'deltaTp'
% ylabel 'm'
% zlabel 'RMSE'
% colorbar
