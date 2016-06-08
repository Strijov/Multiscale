function [figname, caption] = plot_residuals_stats(testRes, trainRes, ...
                                                  ts, model, folder, string)
% Plot evolution of res mean and std by split

figname = '';
caption = '';
if nargin < 6
   string = '';
end
if nargin < 5   
   folder = '';
end

fig = figure;
errorbar(1:size(trainRes, 2), mean(trainRes, 1), std(trainRes, [], 1));
hold on;
errorbar(1:size(testRes, 2), mean(testRes, 1), std(testRes, [], 1));
legend({'Train', 'Test'}, 'Location', 'NorthWest');
xlabel('N. of forecasted point', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Residuals mean', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;
if nargin >= 4
caption = strcat('Residual mean and standard deviation to residuals of \t', ...
                    model.name, ...
                    regexprep(regexprep(ts.name, '_', '.'), '\\', '/'), '.\t');
figname = fullfile(folder, ts.dataset, strcat('stats_', ts.name, string, '.eps'));
saveas(fig, figname, 'epsc');
close(fig)
end

end