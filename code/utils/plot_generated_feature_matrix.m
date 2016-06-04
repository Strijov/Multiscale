function [fname, caption] = plot_generated_feature_matrix(ts, generator_names, folder, string)

if nargin < 3
   folder = 'fig'; 
end
if nargin < 4
   string = ''; 
end

h = figure;
imagesc(ts.X);
title_txt = strjoin(generator_names, ', ');
title(title_txt);
xlabel('Features', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Samples', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
caption = strcat('Feature generation results for\t', regexprep(regexprep(ts.name, '_', '.'), '\\', '/'), ...
    '.\t', title_txt);
fname = fullfile(folder, ts.dataset, strcat('generation_', ts.name, string, '.eps'));
saveas(h, fname, 'epsc');
close(h);

end