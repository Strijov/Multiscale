function [figname, caption] = plot_residuals_npdf(testRes, trainRes, trainPD, testPD, ...
                                                  ts, model, folder, string)
% Plot normal pdf and QQ-plots for train and test residuals
figname = {'',''};
caption = {'',''};
if nargin < 8
   string = '';
end
if nargin < 7   
   folder = '';
end

allRes = sort([trainRes(:); testRes(:)]);

fig = figure;
hold on;
plot(allRes, pdf(trainPD, allRes), 'linewidth', 2);
plot(allRes, pdf(testPD, allRes), 'linewidth', 2);
legend({'Train', 'Test'}, 'Location', 'NorthWest');
xlabel('Residual values', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Normal pdf', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;
if nargin >= 6
caption{1} = strcat('Normal probability density functions fitted to residuals of \t', ...
                    model.name, ...
                    regexprep(regexprep(ts.name, '_', '.'), '\\', '/'), '.\t');
figname{1} = fullfile(folder, ts.dataset, strcat('npdf_', ts.name, string, '.eps'));
saveas(fig, figname{1}, 'epsc');
close(fig)
end

fig = figure;
hold on;
h1 = qqplot(trainRes, trainPD);
h2 = qqplot(testRes, testPD);
set(h1(1),'marker','o','markersize',4,'markeredgecolor',[0 0 1]);
set(h1(2),'linewidth',2,'color',[0 0 1]);
set(h1(3),'linewidth',2,'color',[0 0 1]);
set(h2(1),'marker','x','markersize',4,'markeredgecolor',[1 0 0]);
set(h2(2),'linewidth',2,'color',[1 0 0]);
set(h2(3),'linewidth',2,'color',[1 0 0]);
ylim = get(gca,'ylim');
xlim = get(gca,'xlim');
xlim(1) = min(xlim(1), ylim(1));
xlim(2) = max(xlim(2), ylim(2));
h3 = line(xlim, xlim, 'LineStyle', '-', 'Color', 'k');
h = legend([h1(1), h2(1), h3], {'Train', 'Test', '$x=y$'}, 'Location', 'NorthWest');
set(h,'Interpreter','latex');
xlabel('Quantiles of normal distrubution', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Sample quanties', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
title('');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;
if nargin >= 6
caption{2} = strcat('Quntile-qquantile plot for probability density functions fitted to residuals of \t', ...
                    model.name, ...
                    regexprep(regexprep(ts.name, '_', '.'), '\\', '/'), '.\t');
figname{2} = fullfile(folder, ts.dataset, strcat('qq_', ts.name, string, '.eps'));
saveas(fig, figname{2}, 'epsc');
close(fig)
end


end