function plot_residuals_stats(testRes, trainRes)
% Plot evolution of res mean and std by split

figure;
errorbar(1:size(trainRes, 2), mean(trainRes, 1), std(trainRes, [], 1));
hold on;
errorbar(1:size(testRes, 2), mean(testRes, 1), std(testRes, [], 1));
legend({'Train', 'Test'}, 'Location', 'NorthWest');
xlabel('N. of forecasted point', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Residuals mean', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;

end