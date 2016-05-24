function [figname, caption] = plot_forecasting_results(real_y, model, time_ticks_plot, max_error, figname)
MAX_ERROR = 10^4;

caption = '';

if nargin < 4
    max_error = MAX_ERROR;
end

idx_models = extractfield(model, 'error') < max_error;
if ~all(idx_models)
    model_names = extractfield(model, 'name');
    disp(['Error exceeds ', num2str(max_error), ' for the following models: ', ...
        model_names, '.']);
    disp(['The forecasts for ', strjoin(model_names(~idx_models), ', '), ' are not displayed.']);
end
model = model(idx_models);

h = figure;
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
if exist('figname', 'var')
    figname = strcat('res_', figname);
    saveas(h, fullfile('fig', figname), 'eps');
    close(h);
    caption = strcat('Forecasting results for\t', regexprep(figname, '_', '.'), '.\t', ...
                        strjoin(model_names, ', '));
end

end