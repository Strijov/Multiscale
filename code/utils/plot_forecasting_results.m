function plot_forecasting_results(real_y, model, time_ticks_plot, max_error)
MAX_ERROR = 10^4;

if nargin < 4
    max_error = MAX_ERROR;
end

idx_models = extractfield(model, 'error') < max_error;
model = model(idx_models);

figure;
cla
plot(real_y(time_ticks_plot), 'LineWidth', 2);
hold on
grid on

for i = 1:numel(model)
    plot(model(i).forecasted_y(time_ticks_plot), 'linewidth', 2);
end
legend(['Data', extractfield(model, 'name')], 'Location', 'NorthWest');
xlabel('Time, $t$', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Forecasts', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;


end