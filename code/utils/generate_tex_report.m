function generate_tex_report(report_struct, tex_filename)
% 
% %% fignames, captions, table_results
% \usepackage{ecltree}
% \drawwith{\dottedline{1}}
% \setlength{\GapDepth}{1mm}
% \setlength{\GapWidth}{2mm}
% then insert the tex string

REPORT_FOLDER = 'reports';
tex_filename = fullfile(REPORT_FOLDER, tex_filename);

strbeg = [  '\\documentclass[12pt]{article}\n', ...
            '\\usepackage{a4wide}\n', ...
            '\\usepackage{multicol}\n', ...
            '\\usepackage[cp1251]{inputenc}\n',...
            '\\usepackage[russian]{babel}\n',...
            '\\usepackage{amsmath, amsfonts, amssymb, amsthm, amscd}\n',...
            '\\usepackage{graphicx, epsfig, subfig, epstopdf}\n',...
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


