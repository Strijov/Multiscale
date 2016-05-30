function [figname, caption] = plot_energy_ts(ts)

MAX_PERIODS = 10;

%[figname{1}, caption] = plot_ts(ts);

h = figure;
n_periods = min(floor(numel(ts.x)/ts.deltaTr), MAX_PERIODS);
idx = repmat([1:ts.deltaTr]', 1, n_periods) + repmat((0:n_periods-1)*ts.deltaTr, ts.deltaTr, 1);
plot(ts.x(idx), 'LineWidth', 2);

xlabel('Time, $t$', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Data', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
figname = fullfile(ts.dataset, strcat('data_segms_', ts.name, '.eps'));
saveas(h, fullfile('fig', figname), 'epsc');
close(h);
caption = strcat('\tPeriods of electricity time series\t', regexprep(ts.name, '_', '.'),'\t');




end