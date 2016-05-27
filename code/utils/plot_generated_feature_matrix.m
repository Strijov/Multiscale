function [fname, caption] = plot_generated_feature_matrix(X, generator_names, fname, folder)

caption = '';

h = figure;
imagesc(X);
title_txt = strjoin(generator_names, ', ');
title(title_txt);
xlabel('Features', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Samples', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
if exist('fname', 'var')
    caption = strcat('Feature generation results for\t', regexprep(regexprep(fname, '_', '.'), '\\', '/'), ...
        '.\t', title_txt);
    fname = fullfile('fig', folder, strcat('generation_', fname, '.eps'));
    saveas(h, fname, 'epsc');
    close(h);
    
end

end