%ts0 = csvread('data\tsEnergyConsumption.csv', 0, 1, [0,1,365*24,1]);

filename = 'data\orig\SL2.xls';
sheet = 'Arkusz1';
xlRange = 'D3:AA366';

ts0 = xlsread(filename,sheet,xlRange);
ts0 = reshape(ts0', numel(ts0), 1);
time_points = linspace(numel(ts0), numel(ts0) - 24 * 300, 301);

deltaTp = 144;
deltaTr = 24;
m = 250;
alpha_coeff = 0;
K = 10;
model_draft = struct('name', [], 'params', [], 'tuned_func', [], 'error', [], 'unopt_flag', true, 'forecasted_y', []);
names_vector = cellstr(char('VAR', 'Neural_network'));
[matrix] = CreateRegMatrix(ts0, time_points, deltaTp, deltaTr);
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
m = 250;
alpha_coeff = 0;
K = 1;
error_vec = zeros(1, numel(deltaTp_bounds));
for i = 1:numel(deltaTp_bounds)
    deltaTp = deltaTp_bounds(i);
    deltaTr = 168 - deltaTp;
    [RMSE, model(i), real_y] = ComputeForecastingErrors(matrix, K, m, alpha_coeff, model(1), deltaTp, deltaTr);
    error_vec(i) = RMSE;
end


figure(2)
plot(deltaTp_bounds, error_vec)
grid on;
xlabel 'deltaTp'
ylabel 'RMSE'

%varying m
deltaTp = 144;
deltaTr = 24;
m_bounds = [150:270];
alpha_coeff = 0;
K = 1;
error_vec = zeros(1, numel(m_bounds));
for i = 1:numel(m_bounds)
    m = m_bounds(i);
    [RMSE, model(1), real_y] = ComputeForecastingErrors(matrix, K, m, alpha_coeff, model(1), deltaTp, deltaTr);
    error_vec(i) = RMSE;
end


figure(3)
plot(m_bounds, error_vec)
grid on;
xlabel 'm'
ylabel 'RMSE'

%varying detlaTr x m
deltaTp_bounds = [24:144];
m_bounds = [150:270];
alpha_coeff = 0;
K = 1;
error_mat = zeros(numel(deltaTp_bounds), numel(m_bounds));
for i = 1:numel(deltaTp_bounds)
    deltaTp = deltaTp_bounds(i);
    deltaTr = 168 - deltaTp;
    for j = 1:numel(m_bounds)
        m = m_bounds(j);
        [RMSE, model(1), real_y] = ComputeForecastingErrors(matrix, K, m, alpha_coeff, model(1), deltaTp, deltaTr);
        error_mat(i,j) = RMSE;
    end
end
figure(4)

surfc(deltaTp_bounds, m_bounds, error_mat)
xlabel 'deltaTp'
ylabel 'm'
zlabel 'RMSE'
colorbar
