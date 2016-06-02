function fid = include_subfigs(fid, report)
% Inputs figures, generated during the experiments, into the file, specified with fid  
% Inputs: 
% report.res.figs  contains necessary information to cimstrut the
% table: 
%   figs.names {1 x n_experiments} = names of figures to display 
%   figs.captions {1 x n_experiments} = the corresponding captions
%   

report = report.res;

for n_data = 1:length(report)
    
    for i = 1:numel(report{n_data}.figs)
        fprintf(fid,'\\begin{figure}\n');
        fprintf(fid, '\\centering\n');
        str = include_figs_from_list(report{n_data}.figs(i).names, report{n_data}.figs(i).captions);
        fprintf(fid, str);
        %fprintf(fid, strcat('\\caption{',  strjoin(report{n_data}.figs(i).captions, '. ()'), '.}\n'));
        fprintf(fid,'\\end{figure}\n');
        fprintf(fid,'\n\n');  
    end
      
end

end

function str = include_figs_from_list(fig_list, captions)
    str = '';
    % Magic:
    widths = [0.5, 0.45, 0.35, 0.45];
    eof_str = {'', '', '',  '';...
               '', '', '',  '\\\\';...
               '', '', '\\\\', '';
               '', '', '',   '\\\\'};
    
    subfigs = 'abcdefghijklmnop';
    if ~iscell(fig_list)
        fig_list = regexprep(fig_list, '\\', '/');
        str = strcat(str, '\\includegraphics[width=0.5\\textwidth]{',...
        fig_list, '}\n');
    else
        width = num2str(widths(numel(fig_list)));
        for i = 1:numel(fig_list) 
            fig_list{i} = regexprep(fig_list{i}, '\\', '/');
           str = strcat(str, '\\subfloat[]{\\includegraphics[width=', ...
               width, '\\textwidth]{',...
            fig_list{i}, '}}', eof_str{i, numel(fig_list)},'\n');
        end
    end
    
    if ~iscell(captions)%numel(captions) ~= numel(fig_list)
       str = strcat(str, '\\caption{', captions, '}\n');
    else
        str = strcat(str, '\\caption{');
        for i = 1:numel(captions)
            str = strcat(str, '(', subfigs(i),')\t', captions{i}, '\t');
        end
        str = strcat(str, '}\n');
    end

end