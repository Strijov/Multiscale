function [figname, caption] = plot_pca_results(res, feature_names, ts, folder, string)


if nargin < 3
   ts.name = '';
   ts.dataset = '';
end

if nargin < 4
   folder = 'fig'; 
end
if nargin < 5
   string = ''; 
end


feature_names = strjoin(feature_names,', ');
h = figure;
plot(cumsum(res.var_ratio), 'k-', 'linewidth', 2);
%title('Total percentage of variance explained');
legend(feature_names, 'Location', 'SouthEast');
xlabel('Number of components', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Percentage of explained variance', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;
caption{1} = strcat('Dimensionality reduction with PCA for\t', ...,
                    regexprep(regexprep(ts.name, '_', '.'), '\\', '/'), ',\t', ...
                    feature_names, '.\t');
figname{1} = fullfile(folder, ts.dataset, strcat('var_ratio_', ts.name, string, '.eps'));
saveas(h, figname{1}, 'epsc');
close(h)

h = figure;
imagesc(ts.X);
%title('Total percentage of variance explained');
legend(feature_names, 'Location', 'SouthEast');
xlabel('Number of component', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Samples', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
hold off;
caption{2} = strcat('Features reduced with PCA for\t', ...,
                    regexprep(regexprep(ts.name, '_', '.'), '\\', '/'), ',\t', ...
                    feature_names, '.\t');
figname{2} = fullfile(folder, ts.dataset, strcat('newX_', ts.name, string, '.eps'));
saveas(h, figname{2}, 'epsc');
close(h)

end