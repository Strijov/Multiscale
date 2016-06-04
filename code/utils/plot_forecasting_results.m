function [figname, caption, figname_m, caption_m] = plot_forecasting_results(...
                                                    ts, model, time_ticks_plot,... 
                                                    max_error,...
                                                    folder,...
                                                    string)
MAX_ERROR = 10^4;
if nargin < 4
    max_error = MAX_ERROR;
end
TIME_FRC_RATIO = 0.25;
if nargin < 5
   folder = 'fig'; 
end
if nargin < 6
   string = ''; 
end

% plot frc by model
ls = {'k--', 'k:', 'k-', 'k-.'};
figname_m = cell(1, numel(model));
caption_m = cell(1, numel(model));
for i = 1:numel(model)
   [figname_m{i}, caption_m{i}] = plot_model_forecast(ts, model(i), TIME_FRC_RATIO, ls{i}, folder, string);    
end

% plot all foreasts on one figure
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
plot(ts.x(time_ticks_plot), 'LineWidth', 2);
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
caption = strcat('Forecasting results for\t', ...,
                    regexprep(regexprep(ts.name, '_', '.'), '\\', '/'), '.\t', ...
                    strjoin(model_names, ', '), max_errors_str);
figname = fullfile(folder, ts.dataset, strcat('res_', ts.name, string, '.eps'));
saveas(h, figname, 'epsc');
close(h);


    

end