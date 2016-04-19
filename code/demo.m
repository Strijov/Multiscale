ts0 = csvread('data\tsEnergyConsumption.csv', 0, 1, [0,1,365*24,1]);
deltaTp = 144;
deltaTr = 24;
time_points = linspace(8671, 8671 - 24 * 300, 301);

[matrix] = CreateRegMatrix(ts0, time_points, deltaTp, deltaTr);
m = 250;
N = 10;
alpha_coeff = 0;
model = struct('name', [], 'params', [], 'tuned_func', [], 'error', []);
model.name = 'Neural_network';
[RMSE] = ComputeForecastingErrors(matrix, N, m, alpha_coeff, model, deltaTp, deltaTr);
