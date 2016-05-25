function [fname, caption] = plot_generated_feature_matrix(X, generator_names, fname)

caption = '';

h = figure;
pcolor(X)
title_txt = strjoin(generator_names, ', ');
title(title_txt);
xlabel('Features', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Samples', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
if exist('fname', 'var')
    fname = strcat('generation_', fname, '.eps');
    saveas(h, fullfile('fig', fname), 'epsc');
    close(h);
    caption = strcat('Feature generation results for\t', regexprep(fname, '_', '.'), '.\t', ...
                        title_txt);
end

end