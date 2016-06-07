function plot_residuals_npdf(testRes, trainRes, trainPD, testPD)
% Plot normal pdf and QQ-plots for train and test residuals

allRes = sort([trainRes(:); testRes(:)]);

figure;
hold on;
plot(allRes, pdf(trainPD, allRes), 'linewidth', 2);
plot(allRes, pdf(testPD, allRes), 'linewidth', 2);
legend({'Train', 'Test'}, 'Location', 'NorthWest');
xlabel('Residual values', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Normal pdf', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;

figure;
hold on;
h1 = qqplot(trainRes, trainPD);
h2 = qqplot(testRes, testPD);
set(h1(1),'marker','o','markersize',4,'markeredgecolor',[0 0 0]);
set(h1(2),'linewidth',2,'color',[0 0 1]);
set(h1(3),'linewidth',2,'color',[0 0 1]);
set(h2(1),'marker','x','markersize',4,'markeredgecolor',[0 0 0]);
set(h2(2),'linewidth',2,'color',[1 0 0]);
set(h2(3),'linewidth',2,'color',[1 0 0]);
legend([h1(1), h2(1)], {'Train', 'Test'}, 'Location', 'NorthWest');
xlabel('Quantiles of normal distrubution', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Sample quanties', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
title('');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;


end