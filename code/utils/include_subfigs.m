function fid = include_subfigs(fid, report)

report = report.res;

for n_data = 1:length(report)
    fprintf(fid,'\\begin{figure}\n');
    fprintf(fid, '\\centering\n');
    for i = 1:numel(report{n_data}.names)
        fprintf(fid, strcat('\\subfloat[]{\\includegraphics[width=0.5\\textwidth]{',...
        report{n_data}.names{i}, '.eps}}\n'));
    end
    fprintf(fid, strcat('\\caption{',  strjoin(report{n_data}.captions, '. ()'), '.}\n'));
    fprintf(fid,'\\end{figure}\n');
    fprintf(fid,'\n\n');    
end

end