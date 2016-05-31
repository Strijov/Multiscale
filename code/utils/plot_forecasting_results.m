function [figname, caption] = plot_forecasting_results(real_y, model, time_ticks_plot, max_error, figname, folder)
MAX_ERROR = 10^4;

caption = '';

if nargin < 4
    max_error = MAX_ERROR;
end

model_names =  {model().name};
idx_models = [model().testError] < max_error;
max_errors_str = '';
if ~all(idx_models)
    max_errors_str = [max_errors_str, '\tError exceeds ', num2str(max_error), ' for the following models: ', ...
        strjoin(model_names(~idx_models), ', '), '.'];
    disp(max_errors_str);
    max_errors_str = [max_errors_str, 'The forecasts for ', strjoin(model_names(~idx_models), ', '), ' are not displayed.'];
    disp(max_errors_str);
end
model = model(idx_models);
model_names = model_names(idx_models);

h = figure;
plot(real_y(time_ticks_plot), 'LineWidth', 2);
hold on
grid on


for i = 1:numel(model)
    plot(model(i).forecasted_y(time_ticks_plot), 'linewidth', 2);
end
legend(['Data', model_names], 'Location', 'NorthWest');
xlabel('Time, $t$', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Forecasts', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;
if exist('figname', 'var')
    caption = strcat('Forecasting results for\t', ...,
                        regexprep(regexprep(figname, '_', '.'), '\\', '/'), '.\t', ...
                        strjoin(model_names, ', '), max_errors_str);
    figname = fullfile('fig', folder, strcat('res_', figname, '.eps'));
    saveas(h, figname, 'epsc');
    close(h);
    
end

end