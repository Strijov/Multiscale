%ts0 = csvread('data\tsEnergyConsumption.csv', 0, 1, [0,1,365*24,1]);

filename = 'data\orig\SL2.xls';
sheet = 'Arkusz1';
xlRange = 'D3:AA366';

ts0 = xlsread(filename,sheet,xlRange);
ts0 = reshape(ts0', numel(ts0), 1);

deltaTp = 144;
deltaTr = 24;
time_points = linspace(numel(ts0), numel(ts0) - 24 * 300, 301);

[matrix] = CreateRegMatrix(ts0, time_points, deltaTp, deltaTr);
m = 250;
N = 10;
alpha_coeff = 0;
model = struct('name', [], 'params', [], 'tuned_func', [], 'error', [], 'unopt_flag', true);
model.name = 'Neural_network';
[RMSE] = ComputeForecastingErrors(matrix, N, m, alpha_coeff, model, deltaTp, deltaTr);

%Here just training every model once and forecasting 24 hours
names_vector = cellstr(char('VAR', 'Neural_network'));
[trainX, trainY, testX, testY, val_x, val_y] = FullSplit(matrix, alpha_coeff, deltaTp, deltaTr);
for i = 1:numel(names_vector)
    model.name = char(names_vector(i));
    model = OptimizeModelParameters(trainX, trainY, model);
    forecast_y = ComputeForecast(val_x, model);
    hold on;
    plot(forecast_y);
end
grid on;
plot(val_y, 'LineWidth', 1.5)
legend('VAR', 'Neural net', 'Real')


