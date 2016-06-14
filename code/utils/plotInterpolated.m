function plotInterpolated(X, Y, Z, names, fname)

F = scatteredInterpolant([X, Y], Z);
[Xq,Yq] = meshgrid(linspace(min(X), max(X), 50), linspace(min(Y), max(Y), 50));
Zq = F(Xq,Yq);


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
%surf(Xq,Yq,Zq);
imagesc(Zq);
colorbar;
xlabel(names{1},  'FontSize', 20, 'FontName', 'Times','Interpreter','latex');
ylabel(names{2},  'FontSize', 20, 'FontName', 'Times','Interpreter','latex');
%zlabel('RMSE',  'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')

saveas(fig, [fname,regexprep(names{1}, '\$', ''), '_',...
                                regexprep(names{2}, '\$', '')], 'fig');
close(fig);



end