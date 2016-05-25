function [figname, caption] = plot_energy_ts(ts)

MAX_TIME = 2000;
MAX_PERIODS = 10;


max_time = min(MAX_TIME, size(ts.x, 1));

h = figure;
plot(ts.x(1:max_time), 'LineWidth', 2);

xlabel('Time, $t$', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Data', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;

figname{1} = strcat('data_', ts.name, '.eps');
saveas(h, fullfile('fig', figname{1}), 'epsc');
close(h);
caption = strcat(ts.readme,'\t(a) Target time series\t', regexprep(ts.name, '_', '.'),'\t');


h = figure;
n_periods = min(floor(numel(ts.x)/ts.deltaTr), MAX_PERIODS);
idx = repmat([1:ts.deltaTr]', 1, n_periods) + repmat(0:n_periods-1, ts.deltaTr, 1);
plot(ts.x(idx), 'LineWidth', 2);

xlabel('Time, $t$', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Data', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
figname{2} = strcat('data_segms_', ts.name, '.eps');
saveas(h, fullfile('fig', figname{2}), 'epsc');
close(h);
caption = strcat(caption, '(b) Periods of electricity time series\t', regexprep(ts.name, '_', '.'),'\t');




end