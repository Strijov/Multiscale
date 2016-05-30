function [figname, caption] = plot_ts(ts)

MAX_TIME = 2000;
MAX_PERIODS = 20;
max_time = min(MAX_TIME, size(ts.x, 1));

h = figure;
plot(ts.x(end-max_time+1:end), 'LineWidth', 2);

xlabel('Time, $t$', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Data', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;

figname{1} = fullfile('fig', ts.dataset, strcat('data_', ts.name, '.eps'));
saveas(h, figname{1}, 'epsc');
close(h);
caption{1} = strcat(ts.readme,'\tTarget time series\t', regexprep(ts.name, '_', '.'),'.\t');

h = figure;
plot(ts.matrix(1:MAX_PERIODS, end-ts.deltaTr+1:end)', 'LineWidth', 2);

xlabel('Time, $t$', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Data', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
figname{2} = fullfile(ts.dataset, strcat('data_segms_', ts.name, '.eps'));
saveas(h, fullfile('fig', figname{2}), 'epsc');
close(h);
caption{2} = strcat('\tTarget segments of the time series\t', regexprep(ts.name, '_', '.'),'.\t');



%{
if strcmp(ts.dataset, 'EnergyWeather')  
    figname = {figname, ''};
    caption = {caption, ''};
   [figname{2}, caption{2}] = plot_energy_ts(ts);    
end
%}
end