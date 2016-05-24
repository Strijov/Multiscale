function fid = vertical_res_table(fid, report)
% Writes results of model comparison into the file, specified with fid  
% Inputs: 
% report is a structure that contains necessary information to cimstrut the
% table: 
%   report.algos {1 x n_models} = names of compared models. Models are arranged vertically 
%   report.headers {1 x n_headers} = names of the quality criteria, arranged horiontally
%   report.res = array of res structures
%       res.data = string, name of the data set
%       res.errors [n_models x n_headers] = values of the quality criteria

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