function fid = horizontal_res_table(fid, report)

algos = report.algos;
headers = report.headers;
report = report.res;
n_data = numel(report);

res = [];
for i = 1:numel(report)
    res = [res, report{i}.errors];
end
%res = horzcat(res);
column_specs = repmat({'c'}, 1, numel(headers));
column_specs = strjoin(column_specs, '|');
column_specs = repmat({column_specs}, 1, n_data);
column_specs = strjoin(column_specs, '||');

header_cols = strjoin(repmat( {strjoin(headers, ' & ')}, 1,  n_data), ' & ');

fprintf(fid,'\\begin{table}\n');
fprintf(fid, strcat('\\begin{tabular}{||c||', column_specs,'||}\n'));
fprintf(fid,'\\hline\n');

for i = 1:n_data
   fprintf(fid, strcat(' & \\multicolumn{', num2str(numel(headers)) ,'}{|c||}{',...
                       regexprep(report{i}.data, '_', '.'),'}' ));
end
fprintf(fid, '\\\\ \n'); 
fprintf(fid,'\\hline\n');
fprintf(fid, strcat(' & ',  header_cols, '\\\\ \n'));
fprintf(fid,'\\hline\n');

for n_row = 1:size(res, 1)
    row_str = cellfun(@(x) sprintf('   %.3f',x), num2cell(res(n_row, :)), 'UniformOutput', false);
    fprintf(fid, strcat(algos{n_row}, ' & ', strjoin(row_str, ' & '), '\\\\ \n')); 
    fprintf(fid,'\\hline\n');    
end
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,'\\end{table}\n');

end