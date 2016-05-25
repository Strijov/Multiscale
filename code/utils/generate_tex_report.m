function generate_tex_report(report_struct, tex_filename)
% Generates report in latex format, using functions specified in
% report_struct.handles. Options: @horizontal_res_table, @vertical_res_table, 
% @include_subfigs. 
% Inputs:
% report_struct. The structure of report_struct depends on the
% expriment to be reported. For more information look inside specific
% functions
% tex_filename = name of the output latex file

REPORT_FOLDER = 'reports';
tex_filename = fullfile(REPORT_FOLDER, tex_filename);

strbeg = [  '\\documentclass[12pt]{article}\n', ...
            '\\extrafloats{100}\n',...
            '\\usepackage{a4wide}\n', ...
            '\\usepackage{multicol, multirow}\n', ...
            '\\usepackage[cp1251]{inputenc}\n',...
            '\\usepackage[russian]{babel}\n',...
            '\\usepackage{amsmath, amsfonts, amssymb, amsthm, amscd}\n',...
            '\\usepackage{graphicx, epsfig, subfig, epstopdf}\n',...
            '\\usepackage{longtable}\n', ...
            '\\graphicspath{ {../fig/} }\n',...                      
            '\\begin{document}\n\n'];
strend =    '\\end{document}';

fid = fopen(tex_filename,'w+');
fprintf(fid,strbeg);


for i = 1:numel(report_struct.handles)
   fid = feval(report_struct.handles{i}, fid, report_struct);    
end


fprintf(fid, strend);
fclose(fid);

end


