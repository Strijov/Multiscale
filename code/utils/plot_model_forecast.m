function [figname, caption] = plot_model_forecast(ts, model, time_frc_ratio, ls)

time =  1 + ts.deltaTp:numel(ts.x);
min_time_frc = fix(max(time - ts.deltaTp)*(1 - time_frc_ratio));
time_frc = time(min_time_frc:end - ts.deltaTp);



h = figure;
plot(time, ts.x(time), 'b-', 'linewidth', 1.2);
hold on;
%# vertical line
plot(time_frc + ts.deltaTp, model.forecasted_y(time_frc), ls, 'linewidth', 1.2);
ylim = get(gca,'ylim');
line([min_time_frc, min_time_frc], ylim, 'LineStyle', '-', 'Color', 'k');

legend({'Time series', model.name}, 'Location', 'NorthWest');
xlabel('Time, $t$', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('TS forecast', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;
caption = strcat('Forecasting results for\t', ...,
                    regexprep(regexprep(ts.name, '_', '.'), '\\', '/'), ',\t', ...
                    model.name, '.\t');
figname = fullfile('fig', ts.dataset, strcat('res_', ts.name, '_', model.name, '.eps'));
saveas(h, figname, 'epsc');
close(h);



end