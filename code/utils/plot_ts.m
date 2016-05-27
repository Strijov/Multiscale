function [figname, caption] = plot_ts(ts)

MAX_TIME = 2000;
max_time = min(MAX_TIME, size(ts.x, 1));

h = figure;
plot(ts.x(1:max_time), 'LineWidth', 2);

xlabel('Time, $t$', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Data', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;

figname = fullfile('fig', ts.dataset, strcat('data_', ts.name, '.eps'));
saveas(h, figname, 'epsc');
close(h);
caption = strcat(ts.readme,'\tTarget time series\t', regexprep(ts.name, '_', '.'),'.\t');

if strcmp(ts.dataset, 'EnergyWeather')  
    figname = {figname, ''};
    caption = {caption, ''};
   [figname{2}, caption{2}] = plot_energy_ts(ts);    
end

end