function plotRandSearchResuls(X, Z, names, fname)

[~, i] = min(Z);
disp('Rand search objctive, parameters:')
disp([Z(i), X(i, :)])

% Use diverging coormap:
rgb = [ ...
    94    79   162
    50   136   189
   102   194   165
   171   221   164
   230   245   152
   255   255   191
   254   224   139
   253   174    97
   244   109    67
   213    62    79
   158     1    66  ] / 255;



fig = figure;
colormap(rgb);
if size(X, 2) == 3
scatter3(X(:, 1),X(:, 2), X(:, 3), 400, Z-min(Z), '.');
zlabel(names{3},  'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
else
scatter(X(:, 1),X(:, 2), 400, Z-min(Z), '.');
end
xlabel(names{1},  'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel(names{2},  'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
colorbar;
saveas(fig, fname, 'fig');
close(fig);


idxPairs = combntns(1:size(X, 2), 2);
for idx = idxPairs'
    plotInterpolated(X(:, idx(1)), X(:, idx(2)), Z, names(idx), fname);
end

end