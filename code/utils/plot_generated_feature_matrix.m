function plot_generated_feature_matrix(X, generator_names)

figure;
pcolor(X)
title(strjoin(generator_names, ' , '));
xlabel('Features', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Samples', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')


end