function fid = horizontal_res_table(fid, report)
% Writes results of model comparison into the file, specified with fid  
% Inputs: 
% report is a structure that contains necessary information to cimstrut the
% table: 
%   report.algos {1 x n_models} = names of compared models. Models are arranged horizontally 
%   report.headers {1 x n_headers} = names of the quality criteria, arranged horizontally
%   report.res = array of res structures
%       res.data = string, name of the data set
%       res.errors [n_models x n_headers] = values of the quality criteria

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