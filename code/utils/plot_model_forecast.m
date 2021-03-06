function [figname, caption] = plot_model_forecast(ts, model, nTs, time_frc_ratio, ls, folder, string)

deltaTp = ts.deltaTp(nTs);
yBlocks = [0, cumsum(ts.deltaTr)]*size(ts.Y, 2)/sum(ts.deltaTr);
time =  1: numel(ts.Y(:, yBlocks(nTs)+1:yBlocks(nTs+1)));
min_time_frc = fix(max(time - deltaTp)*(1 - time_frc_ratio));
time_frc = time(min_time_frc:end);



h = figure;
plot(time + deltaTp, ts.x{nTs}(time), 'b-', 'linewidth', 1.2);
hold on;
%# vertical line
plot(time_frc + deltaTp, model.forecasted_y{nTs}(time_frc), ls, 'linewidth', 1.2);
ylim = get(gca,'ylim');
line([min_time_frc + deltaTp, min_time_frc + deltaTp], ylim, 'LineStyle', '-', 'Color', 'k');

legend({'Time series', model.name}, 'Location', 'NorthWest');
xlabel('Time, $t$', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('TS forecast', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;
caption = strcat('Forecasting results for\t', ...,
                    regexprep(regexprep(ts.name, '_', '.'), '\\', '/'), ',\t', ...
                    model.name , '.\t');
figname = fullfile(folder, ts.dataset, strcat('res_', ts.name, '_', ...
                            regexprep(ts.legend{nTs}, ' ', '_'), '_',...
                            regexprep(model.name, ' ', '_'), string, '.eps'));
saveas(h, figname, 'epsc');
close(h);



end