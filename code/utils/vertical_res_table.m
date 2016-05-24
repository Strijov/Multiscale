function fid = vertical_res_table(fid, report)

algos = report.algos;
headers = report.headers;
report = report.res;
n_data = numel(report);

%res = horzcat(res);
column_specs = repmat({'c'}, 1, numel(headers));
column_specs = strjoin(column_specs, '|');

header_cols = strcat('Data & Models & ', strjoin(headers, ' & '));

fprintf(fid,'\\begin{table}\n');
fprintf(fid, strcat('\\begin{tabular}{|p{2cm}|c|', column_specs,'|}\n'));
fprintf(fid,'\\hline\n');

fprintf(fid, strcat(header_cols, '\\\\ \n'));
fprintf(fid,'\\hline\n');

for i = 1:n_data
    %data_name = regexprep(report{i}.data, '_', ' ');
    fprintf(fid, strcat('\\multirow{', num2str(numel(algos)) ,'}{*}{',...
                       num2str(i),'}' ));
    for n_row = 1:numel(algos)               
        row_str = cellfun(@(x) sprintf('   %.3f',x), num2cell(report{i}.errors(n_row, :)), 'UniformOutput', false);
        fprintf(fid, strcat(' & ', algos{n_row}, ' & ', strjoin(row_str, ' & '), '\\\\ \n')); 
        fprintf(fid,strcat('\\cline{2-', num2str(numel(headers)+2), '}\n') );  
    end
    fprintf(fid,'\\hline\n');
end
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,'\\end{table}\n');

end